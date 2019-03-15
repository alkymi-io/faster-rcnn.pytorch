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
from torch.autograd import Variable
from torch import nn

from lib.model.faster_rcnn.resnet import resnet
from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from lib.model.utils.blob import im_list_to_blob
from lib.model.utils.config import cfg
from lib.model.nms.nms_wrapper import nms


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
            # load_name = 'faster-rcnn.pt'

            checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
            cls.classes = checkpoint['classes']
            model = resnet(cls.classes, 'resnet101')
            model.create_architecture()

            model = nn.DataParallel(model)
            model.load_state_dict(checkpoint['model'])
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

            blobs, im_scales = _get_image_blob(im_in)
            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            im_data = Variable(torch.from_numpy(im_blob).permute(0, 3, 1, 2))

            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
            im_info = Variable(torch.from_numpy(im_info_np))

            gt_boxes = Variable(torch.zeros(1, 1, 5))
            num_boxes = Variable(torch.zeros(1))
            rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = cls.model(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4 * len(cls.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= im_scales[0]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()

            result = dict()
            for j in range(1, len(cls.classes)):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    result[cls.classes[j]] = cls_dets.cpu().numpy().tolist()
            return {'pred': result,
                    'metrics': {'rpn_loss_cls': rpn_loss_cls,
                                'rpn_loss_box': rpn_loss_box,
                                'RCNN_loss_cls': RCNN_loss_cls,
                                'RCNN_loss_bbox': RCNN_loss_bbox,
                                'rois_label': rois_label}}


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
