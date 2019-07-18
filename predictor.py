from __future__ import print_function

import cv2
import io
from PIL import Image
from flask import Flask, request, jsonify, Response

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import numpy as np
import torch

from lib.model.faster_rcnn.resnet import resnet
from lib.model.rpn.bbox_transform import bbox_transform_inv, keep_detections
from lib.model.utils.blob import im_list_to_blob
from lib.model.utils.config import cfg
from model.roi_layers import nms
from overlap_suppression import overlap_suppression

# The flask app for serving predictions
app = Flask(__name__)
app.secret_key = 'super secret key'


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that makes a prediction for the input data.
class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance,
        loading it if it's not already loaded."""
        if cls.model is None:
            load_name = '/opt/ml/model/faster-rcnn.pt'
            checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
            print('checkpoint keys:', checkpoint.keys())
            old_classes = ['__background__', 'text',
                           'structured_data', 'graphical_chart', 'title']
            classes = checkpoint.get('classes', old_classes)
            model = resnet(classes, 'resnet101')
            model.create_architecture()

            # If the model was trained with DataParallel we need to remove the
            # "module." prefix from the parameter names.
            state_dict = {}
            for key, value in checkpoint['model'].items():
                if key.startswith('module.'):
                    state_dict[key[7:]] = value
                else:
                    state_dict[key] = value
            
            model.load_state_dict(state_dict)
            model.classes = classes
            model.cuda()
            model.eval()
            cls.model = model

        return cls.model

    @classmethod
    def predict(cls, im_in):
        """For the input, do the predictions and return them.
        Args:
            im_in (a PIL image): The data on which to do the predictions."""

        assert len(im_in.shape) == 3, "RGB images only"

        if cls.model is None:
            cls.model = cls.get_model()

        with torch.no_grad():
            im_blob, im_scales = _get_image_blob(im_in)
            assert len(im_scales) == 1, "Only single-image batch implemented"

            # Convert to BRG
            im_data = torch.from_numpy(im_blob).permute(0, 3, 1, 2).cuda()

            # im_info stores (height, width, scale)
            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
            im_info = torch.from_numpy(im_info_np).cuda()

            # When predicting on new images the ground truth boxes are unknown.
            # We can set them to 0.
            gt_boxes = torch.zeros(1, 1, 5).cuda()
            num_boxes = torch.zeros(1).cuda()

            # Call the model

            # rois has shape (batch_size, RPN_POST_NMS_TOP_N, 5)
            # The last dimension holds (example_idx, x, y, x2, y2).
            # example idx is always 1 at inference time since batch_size==1

            # cls_prob has shape
            # (batch_size, RPN_POST_NMS_TOP_N, num_classes + 1)
            # The first probability is for the background class

            # bbox_delta has shape
            # (batch_size, RPN_POST_NMS_TOP_N, 4 * num_classes)
            # The last dim holds the bounding box deltas
            # (dx_0, dy_0, dw_0, dh_0,
            #  dx_1, dy_1, dw_1, dh_1...)
            # Where the subscript is the class index.
            rois, cls_prob, bbox_delta, _, _, _, _, _ = cls.model(im_data, im_info, gt_boxes, num_boxes)

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                # box_deltas = bbox_delta.data

                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    bbox_delta = bbox_delta.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()

                    bbox_delta = bbox_delta.view(1, -1, 4 * len(cls.model.classes))

                # Apply the deltas to the ROIS to get the predicted boxes.
                pred_boxes = bbox_transform_inv(rois.data[:, :, 1:5], bbox_delta, 1)

            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(rois.data[:, :, 1:5], (1, cls_prob.shape[1]))

            # pred boxes has shape
            # (max_num_detections=300, num_coordinates=4 * num_classes)
            # The coords in the last dimension are in the format
            # (x_0, y_0, x2_0, y2_0, x_1, y_1, x2_1, y2_1,...)
            # Scale the predicted boxes so they line up with the input image
            pred_boxes = (pred_boxes/im_scales[0]).squeeze(0)
            cls_prob = cls_prob.squeeze(0)

            iou_thresh = 0.3
            all_detections = []

            max_class_probs, max_class_idxs = torch.max(cls_prob, dim=1)

            # Assume that the 0th class is __background__
            for j in range(1, len(cls.model.classes)):
                # Get the indices of boxes where the maximum probability is in the jth column
                class_idxs = torch.nonzero(max_class_idxs == j).view(-1)

                # Get the probabilities of the boxes that were predicted to be class j.
                class_probs = max_class_probs[class_idxs]

                if class_idxs.numel() > 0:
                    # Sort the class j boxes in order of decreasing probability
                    _, order = torch.sort(class_probs, 0, True)

                    class_probs = class_probs[order]

                    class_boxes = pred_boxes[class_idxs, j * 4:(j + 1) * 4][order]

                    # Concat box coords with ALL class probabilities
                    # class_detections has shape
                    # (class_count, num_coords + num_classes = 8)
                    class_detections = torch.cat((class_boxes, cls_prob[class_idxs, :]), 1)

                    # Perform non-max suppression for class j
                    # keep holds a list of indices of boxes that survived nms
                    keep = nms(class_boxes, class_probs, iou_thresh).view(-1).long()
                    class_detections = class_detections[keep, :]

                    # Only keep predicted boxes if they are entirely inside the
                    # page bounds.
                    keep2 = keep_detections(class_detections, im_info[0, 1]/im_scales[0], im_info[0, 0]/im_scales[0])
                    class_detections = class_detections[keep2, :]

                    all_detections.append(class_detections)

            if all_detections:
                all_detections = torch.cat(all_detections, 0)

                # Iterate through all of the detected boxes in order of
                # decreasing confidence. Kill any boxes that overlap with more
                # confident boxes.
                ovr_sup_thresh = 0.2
                keep3 = overlap_suppression(all_detections.cpu(), ovr_sup_thresh)
                all_detections = all_detections[keep3.long()]
                result2 = all_detections.cpu().numpy().tolist()
            else:
                result2 = []

            raw_detections = []
            for j in range(len(cls.model.classes)):
                class_idxs = torch.nonzero(max_class_idxs == j).view(-1)
                class_probs = max_class_probs[class_idxs]
                if class_idxs.numel() > 0:
                    _, order = torch.sort(class_probs, 0, True)
                    class_boxes = pred_boxes[class_idxs][:, j * 4:(j + 1) * 4]
                    class_probs = max_class_probs[class_idxs]

                    # Concat most probable box coords with ALL class probabilities
                    class_detections = torch.cat((class_boxes, cls_prob[class_idxs]), 1)[order]

                    # Perform non-max suppression for class j
                    # keep holds a list of indices of boxes that survived nms
                    keep = nms(class_boxes[order, :], class_probs[order], iou_thresh)
                    class_detections = class_detections[keep.view(-1).long()]
                    raw_detections.append(class_detections)
            raw = torch.cat(raw_detections, 0).cpu().numpy().tolist()

            #################################################################
            # THIS BLOCK PRODUCES THE CONTENTS OF THE DEPRECATED 'pred' field
            #################################################################
            result = dict()
            thresh = 0.05
            for j in range(1, len(cls.model.classes)):
                # Indices where class j probability exceeds the detection
                # threshold
                inds = torch.nonzero(cls_prob[:, j] > thresh).view(-1)

                # if there is detection for class j
                if inds.numel() > 0:
                    cls_scores = cls_prob[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]

                    # Perform non-max suppression for class j
                    # keep holds a list of indices of boxes that survived nms
                    keep = nms(cls_boxes[order, :], cls_scores[order], iou_thresh)
                    cls_dets = cls_dets[keep.view(-1).long()]

                    result[cls.model.classes[j]] = cls_dets.cpu().numpy().tolist()
            #################################################################

            return {'pred': result, 'predicted_boxes': result2, 'raw_boxes': raw, 'box_labels': list(cls.model.classes)}


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy.
    We declare this sample container to be healthy
    if we can load the model successfully."""
    model = ScoringService.get_model()
    health = model is not None  # You can insert a health check here
    status = 200 if health else 404
    return Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():

    try:
        data = io.BytesIO(request.data)
        image = Image.open(data)
    except Exception:
        return Response(response='Data could not be deserialized as an image',
                        status=415, mimetype='text/plain')
    im_arr = np.array(image)
    result = ScoringService.predict(im_arr)
    return jsonify(result)


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
    im (ndarray): a color image in BGR order
    Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
      At test time there can only be one scale (i.e. len(cfg.TEST.SCALES) == 1)
      so the pyramid has height of 1.
    """
    im_orig = im.astype(np.float32, copy=True)[:, :, :3]
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None,
                        fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
