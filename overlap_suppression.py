from __future__ import absolute_import

import numpy as np
import torch


def overlap_suppression(dets, thresh):
    """
    Return indices of boxes that do not overlap with a more confident box.

    :param dets:
    :param thresh:
    :return:
    """
    dets = dets.numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    confidence = dets[:, 4:].max(axis=1)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort boxes by decreasing confidence
    order = confidence.argsort()[::-1]

    keep = []
    while order.size > 0:
        # Add the most confident box to a list of boxes to be kept
        i = order.item(0)
        keep.append(i)

        # Calculate the area of overlap between the most confident box and all
        # the others
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # Calculate the fractional overlap relative to the smaller area
        ovr = inter / np.minimum(areas[order[0]], areas[order[1:]])

        # Get the indices where the fractional overlap does not exceed the
        # threshold. Shift the indices to account for the most confident box.
        idxs = np.where(ovr < thresh)[0] + 1

        # Filter the bounding boxes. This keeps only boxes that do not
        # significantly overlap with the most confidant box. This process
        # always removes the most confidant box.
        order = order[idxs]

    return torch.IntTensor(keep)


