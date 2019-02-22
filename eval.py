import sagemaker
from PIL import Image
import requests
from io import BytesIO
import os
import json
import numpy as np
import xmltodict
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser


def main(name):
    predictor = sagemaker.predictor.RealTimePredictor('faster-rcnn-ep')

    results = []

    path = '/home/ubuntu/persist/data_gen/data/pageseg_examples4/validation_annotation/'
    filenames = os.listdir(path)
    labels = ['text', 'graphical_chart', 'structured_data']
    tp_counts = dict()

    for filename in tqdm(filenames):
        hash_key = os.path.splitext(filename)[0]
        hash_key, page_num = hash_key.split('_')
        s3_url = 's3://%s/%s/source.pdf' % ('va-clean', hash_key)
        params = {'page': page_num, 's3_url': s3_url}
        try:
            page_image_response = requests.get('https://pdf-service.alkymi.cloud/v2/getPageImage', params=params)
            prediction_response = predictor.predict(page_image_response.content)
        #     prediction_response = requests.post('http://54.173.93.209:8080/invocations', data=page_image_response.content).content
            pred = json.loads(prediction_response)['pred']
            img_bytes = BytesIO(page_image_response.content)
            img = Image.open(img_bytes)
        except KeyboardInterrupt:
            break
        except:
            continue

        file_path = os.path.join(path, filename)
        with open(file_path, 'r') as f:
            xml_data = xmltodict.parse(f.read(), force_list={'object'})

        for label in labels:
            for obj in xml_data['annotation']['object']:
                if obj['name'] == label:
                    tp_counts[label] = tp_counts.get(label, 0) + 1

            for pred_box in pred.get(label, []):
                pred_mask = np.zeros(img.size)
                pred_mask[int(pred_box[0]):int(pred_box[2]), int(pred_box[1]):int(pred_box[3])] = 1
                iou_max = 0
                result = None
                for obj in xml_data['annotation']['object']:
                    if obj['name'] == label:
                        gt_box = obj['bndbox']
                        gt_mask = np.zeros(img.size)
                        gt_mask[int(float(gt_box['xmin'])):int(float(gt_box['xmax'])), int(float(gt_box['ymin'])):int(float(gt_box['ymax']))] = 1

                        intersection = np.logical_and(gt_mask, pred_mask).sum()
                        union = np.logical_or(gt_mask, pred_mask).sum()
                        iou = intersection/union if union else 0.0
                        if iou > iou_max:
                            iou_max = iou
                    result = (pred_box[4], iou_max, label)
                results.append(result)

    labels = ['text', 'structured_data', 'graphical_chart']
    label_aps = dict()
    for label in labels:
        iou_threshes = np.arange(0.5, 1.0, 0.05)
        aps = []
        for iou_thresh in iou_threshes:
            results_df = pd.DataFrame(results, columns=['confidence', 'max_iou', 'label'])
            results_df = results_df.sort_values('confidence', ascending=False)
            results_df = results_df[results_df.label == label]
        #     results_df = results_df[results_df.confidence > .5]
            results_df = results_df.reset_index(drop=True)
            results_df['cum_sum'] = (results_df.max_iou > iou_thresh).cumsum()
            results_df['precision'] = results_df.apply(lambda row: row.cum_sum/(row.name+1), axis=1)
            results_df['recall'] = results_df.cum_sum/tp_counts[label]

            precisions = [1.0]
            for rec in np.arange(.1, 1.1, .1):
                p = results_df.precision[results_df.recall>=rec].head(1)
                precisions.append(float(p) if len(p) else 0.0)
            ap = sum(precisions)/11
            aps.append(ap)
        label_aps[label] = aps

    with open(f'./results/{name}.txt', 'w') as f:
        for label, aps in label_aps.items():
            f.write(label + '\n')
            f.write(str(sum(aps)/len(aps)) + '\n')
            f.write(json.dumps(aps) + '\n\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str)
    args = parser.parse_args()
    main(args.name)
