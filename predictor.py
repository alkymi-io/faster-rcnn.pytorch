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
from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes, keep_detections
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
            # load_name = '/opt/ml/model/faster-rcnn.pt'
            load_name = 'faster-rcnn.pt'
            checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
            classes = checkpoint['classes']
            model = resnet(classes, 'resnet101')
            model.create_architecture()

            # If the model was trained with DataParallel we need to remove the
            # "module." prefix from the parameter names.
            checkpoint = {key[7:]: value
                          for key, value in checkpoint['model'].items()}
            
            model.load_state_dict(checkpoint)
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
        thresh = 0.05

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

            # bbox_pred has shape
            # (batch_size, RPN_POST_NMS_TOP_N, 4 * num_classes)
            # The last dim holds the bounding box deltas
            # (dx_0, dy_0, w_0, h_0,
            #   ddx_1, dy_1, w_1, h_1...)
            # Where the subscript is the class index.
            rois, cls_prob, bbox_pred, _, _, _, _, _ = cls.model(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()

                    box_deltas = box_deltas.view(1, -1, 4 * len(cls.model.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)

                # Clip boxes so that they fit inside the image bounds
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            # Scale the predicted boxes so they line up with the input image
            pred_boxes /= im_scales[0]

            scores = scores.squeeze()

            pred_boxes = pred_boxes.squeeze()
            result = dict()
            iou_thresh = 0.3
            all_detections = []

            max_class_scores, max_class_idxs = torch.max(scores, dim=1)
            for j in range(len(cls.model.classes)):
                class_idxs = torch.nonzero(max_class_idxs == j).view(-1)
                class_scores = max_class_scores[class_idxs]
                if class_idxs.numel() > 0:
                    _, order = torch.sort(class_scores, 0, True)
                    class_boxes = pred_boxes[class_idxs][:, j * 4:(j + 1) * 4]
                    class_scores = max_class_scores[class_idxs]

                    # Concat box coords with ALL class probabilities
                    class_detections = torch.cat((class_boxes, scores[class_idxs]), 1)[order]

                    # Perform non-max suppression for class j
                    # keep holds a list of indices of boxes that survived nms
                    keep = nms(class_boxes[order, :], class_scores[order], iou_thresh)
                    class_detections = class_detections[keep.view(-1).long()]

                    # Only keep predicted boxes if the are entirely inside the
                    # page bounds.
                    # Might want to relax this.
                    keep2 = keep_detections(class_detections, im_info.data)
                    class_detections = class_detections[keep2]
                    all_detections.append(class_detections)

            all_detections = torch.cat(all_detections, 0)

            # Iterate through all of the detected boxes in order of decreasing
            # confidence. Kill any boxes that overlap with more confident boxes.
            keep3 = overlap_suppression(all_detections.cpu())
            all_detections = all_detections[keep3.long()]
            result2 = all_detections.cpu().numpy().tolist()

            for j in range(1, len(cls.model.classes)):
                # Indices where class j probability exceeds the detection threshold
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)

                # if there is detection for class j
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]

                    # Perform non-max suppression for class j
                    # keep holds a list of indices of boxes that survived nms
                    keep = nms(cls_boxes[order, :], cls_scores[order], iou_thresh)
                    cls_dets = cls_dets[keep.view(-1).long()]

                    result[cls.model.classes[j]] = cls_dets.cpu().numpy().tolist()

            return {'pred': result, 'pred2': result2}


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
