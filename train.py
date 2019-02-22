#!/usr/bin/env python

from __future__ import print_function

import os
import json
import sys
import time
from argparse import ArgumentParser
from shutil import copy
import traceback
from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

from lib.model.faster_rcnn.resnet import resnet
from lib.model.utils.net_utils import adjust_learning_rate
from lib.model.utils.config import cfg
from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


def train():

    if args.env == 'sagemaker':
        prefix = '/opt/ml/'
        model_path = os.path.join(prefix, 'model')
        param_path = os.path.join(prefix, 'input/config/hyperparameters.json')
        output_path = os.path.join(prefix, 'output')

    elif args.env == 'local':
        model_path = './save/%s/' % str(int(time.time()))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        param_path = './hyperparameters.json'
        copy(param_path, model_path)
        output_path = './output'
    else:
        raise ValueError

    print('Starting the training.')
    # try:
    with open(param_path, 'r') as tc:
        trainingParams = json.load(tc)
    batch_size = int(trainingParams.get('batch_size', None))
    epochs = int(trainingParams.get('epochs', None))
    lr = float(trainingParams.get('lr', None))
    imdb_name = trainingParams.get('imdb_name', None)  # 'fakepages'
    num_workers = int(trainingParams.get('num_workers', None))
    lr_patience = int(trainingParams.get('lr_decay_step', 2))
    lr_decay_gamma = float(trainingParams.get('lr_decay_gamma', 0.1))
    early_stopping_patience = float(trainingParams.get('patience', 4))

    imdb_train, roidb_train, ratio_list_train, ratio_index_train = combined_roidb(imdb_name + '_train')
    train_size = len(roidb_train)
    sampler_batch_train = sampler(train_size, batch_size)
    dataset_train = roibatchLoader(roidb_train, ratio_list_train, ratio_index_train, batch_size, imdb_train.num_classes, training=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    imdb_val, roidb_val, ratio_list_val, ratio_index_val = combined_roidb(imdb_name + '_validation', training=False)
    val_size = len(roidb_val)
    sampler_batch_val = sampler(val_size, batch_size)
    dataset_val = roibatchLoader(roidb_val, ratio_list_val, ratio_index_val, batch_size, imdb_val.num_classes)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    classes = np.asarray(['__background__',
                          'text',
                          'structured_data',
                          'graphical_chart'])

    fasterRCNN = resnet(classes, 'resnet101', pretrained=True, class_agnostic=False)
    fasterRCNN.create_architecture()
    fasterRCNN.cuda()

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
                params += [{'params': [value], 'lr': lr}]

    optimizer = torch.optim.Adam(params)
    best_val_loss = 10000
    if args.load_model:
        print('Loading weights from %s ...' % args.load_model)
        checkpoint = torch.load(args.load_model)
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_val_loss = checkpoint.get('best_val_loss', 10000)
    fasterRCNN.train()

    train_log_path = os.path.join(model_path, 'train_log.csv')
    with open(train_log_path, 'a+') as f:
        f.write('epoch,loss,rpn_cls,rpn_box,rcnn_cls,rcnn_box\n')
    val_log_path = os.path.join(model_path, 'val_log.csv')
    with open(val_log_path, 'a+') as f:
        f.write('epoch,loss,rpn_cls,rpn_box,rcnn_cls,rcnn_box\n')

    train_log = []
    val_log = []

    patience_count = 0

    for epoch in range(epochs):

        train_loss_tot = 0
        train_rpn_cls_tot = 0
        train_rpn_box_tot = 0
        train_rcnn_cls_tot = 0
        train_rcnn_box_tot = 0
        num_examples_train = 0

        train_start_time = time.time()

        t = tqdm(dataloader_train)
        for batch_idx, (im_data, im_info, gt_boxes, num_boxes) in enumerate(t):
            im_data = Variable(im_data.cuda())
            im_info = Variable(im_info.cuda())
            gt_boxes = Variable(gt_boxes.cuda())
            num_boxes = Variable(num_boxes.cuda())

            rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            # train_loss = rpn_loss_cls + rpn_loss_box + RCNN_loss_cls + RCNN_loss_bbox
            #
            # num_examples_train += im_data.shape[0]
            # train_loss_tot += train_loss.data.cpu().numpy() * im_data.shape[0]
            # train_rpn_cls_tot += rpn_loss_cls.data.cpu().numpy() * im_data.shape[0]
            # train_rpn_box_tot += rpn_loss_box.data.cpu().numpy() * im_data.shape[0]
            # train_rcnn_cls_tot += RCNN_loss_cls.data.cpu().numpy() * im_data.shape[0]
            # train_rcnn_box_tot += RCNN_loss_bbox.data.cpu().numpy() * im_data.shape[0]
            #
            # fasterRCNN.zero_grad()
            # optimizer.zero_grad()
            # train_loss.backward()
            # clip_gradient(fasterRCNN, 1.)
            # optimizer.step()
            # t.set_description(desc='Batch Loss: %f' % train_loss)

        train_end_time = time.time()

        epoch_log_train = {'epoch': epoch,
                           'loss': train_loss_tot/num_examples_train,
                           'rpn_cls': train_rpn_cls_tot/num_examples_train,
                           'rpn_box': train_rpn_box_tot/num_examples_train,
                           'rcnn_cls': train_rcnn_cls_tot/num_examples_train,
                           'rcnn_box': train_rcnn_box_tot/num_examples_train}
        with open(train_log_path, 'a') as f:
            f.write(','.join([str(v) for v in epoch_log_train.values()]) + '\n')

        print("[epoch %2d] train loss: %.4f, lr: %.2e" % (epoch, epoch_log_train['loss'], lr))
        print("\t\t\ttrain time cost: %f" % (train_end_time - train_start_time))
        print("\t\t\ttrain rpn_cls: %.4f, train rpn_box: %.4f, train rcnn_cls: %.4f, train rcnn_box %.4f"
              % (epoch_log_train['rpn_cls'], epoch_log_train['rpn_box'], epoch_log_train['rcnn_cls'], epoch_log_train['rcnn_box']))

        train_log.append(epoch_log_train)

        val_loss_tot = 0
        val_rpn_cls_tot = 0
        val_rpn_box_tot = 0
        val_rcnn_cls_tot = 0
        val_rcnn_box_tot = 0
        num_examples_val = 0

        val_start_time = time.time()

        with torch.no_grad():
            for im_data, im_info, gt_boxes, num_boxes in tqdm(dataloader_val):
                im_data = Variable(im_data.cuda())
                im_info = Variable(im_info.cuda())
                gt_boxes = Variable(gt_boxes.cuda())
                num_boxes = Variable(num_boxes.cuda())

                rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
                val_loss = rpn_loss_cls + rpn_loss_box + RCNN_loss_cls + RCNN_loss_bbox

                num_examples_val += im_data.shape[0]
                val_loss_tot += val_loss.cpu().numpy() * im_data.shape[0]
                val_rpn_cls_tot += rpn_loss_cls.cpu().numpy() * im_data.shape[0]
                val_rpn_box_tot += rpn_loss_box.cpu().numpy() * im_data.shape[0]
                val_rcnn_cls_tot += RCNN_loss_cls.cpu().numpy() * im_data.shape[0]
                val_rcnn_box_tot += RCNN_loss_bbox.cpu().numpy() * im_data.shape[0]

        val_end_time = time.time()

        epoch_log_val = {'epoch': epoch,
                         'loss': val_loss_tot / num_examples_val,
                         'rpn_cls': val_rpn_cls_tot / num_examples_val,
                         'rpn_box': val_rpn_box_tot / num_examples_val,
                         'rcnn_cls': val_rcnn_cls_tot / num_examples_val,
                         'rcnn_box': val_rcnn_box_tot / num_examples_val}

        with open(val_log_path, 'a') as f:
            f.write(','.join([str(v) for v in epoch_log_val.values()]) + '\n')

        print("[epoch %2d] val loss: %.4f, lr: %.2e" % (epoch, epoch_log_val['loss'], lr))
        print("\t\t\tval time cost: %f" % (val_end_time - val_start_time))
        print("\t\t\tval rpn_cls: %.4f, val rpn_box: %.4f, val rcnn_cls: %.4f, val rcnn_box %.4f"
              % (epoch_log_val['rpn_cls'], epoch_log_val['rpn_box'], epoch_log_val['rcnn_cls'],  epoch_log_val['rcnn_box']))

        val_log.append(epoch_log_val)

        if epoch_log_val['loss'] < best_val_loss:
            patience_count = 0

            print('val loss improved from %.4f to %.4f' % (best_val_loss, epoch_log_val['loss']))
            best_val_loss = epoch_log_val['loss']
            model_file_path = os.path.join(model_path, 'faster-rcnn.pt')
            torch.save({
                'epoch': epoch + 1,
                'model': fasterRCNN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'classes': classes
            }, model_file_path)

            model_file_size = os.path.getsize(model_file_path)

            print("saved model file:", model_file_path)
            print('model file size:', model_file_size)
        else:
            patience_count += 1

        if patience_count >= lr_patience:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma
        if patience_count == early_stopping_patience:
            return

    # except Exception as e:
    #     # Write out an error file. This will be returned as the failureReason in the
    #     # DescribeTrainingJob result.
    #     trc = traceback.format_exc()
    #     with open(os.path.join(output_path, 'failure'), 'w') as s:
    #         s.write('Exception during training: ' + str(e) + '\n' + trc)
    #     # Printing this causes the exception to be in the training job logs, as well.
    #     print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
    #     # A non-zero exit code causes the training job to be marked as Failed.
    #     sys.exit(255)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', dest='env', help='training environment (local or sagemaker)', default='sagemaker', type=str)
    parser.add_argument('--load_model', help='path to model file to load for resuming training', type=str)
    parser.add_argument('--cfg', help='Optional config file', type=str)

    args = parser.parse_args()

    cfg.TRAIN.USE_FLIPPED = False
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
