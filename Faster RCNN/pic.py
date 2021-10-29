import torch
import torchvision
from PIL import Image
import cv2
from faster_rcnn_utils.engine import evaluate
from faster_rcnn_utils.AIZOODataset import AIZOODataset
from faster_rcnn_utils.transforms import get_transform
from faster_rcnn_utils import utils
from torchvision.transforms import functional as F
import os
import time
import numpy as np
import argparse
import datetime

def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0].astype(int)
    y1 = dets[:, 1].astype(int)
    x2 = dets[:, 2].astype(int)
    y2 = dets[:, 3].astype(int)
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, default="testing/input/images", help="path to images directory")
    parser.add_argument("--output_path", type=str, default="testing/output/images", help="output image directory")
    parser.add_argument("--model_def", type=str, default="config/yolov3_mask.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_self.pth",help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/mask_dataset.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    # Output directory
    os.makedirs(opt.output_path, exist_ok=True)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("正在使用的设备是",device)

    num_classes = 3
    BATCH_SIZE = 1

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
    model.load_state_dict(torch.load("checkpoints_faster_rcnn/yolov3_ckpt_0__'1'.pth"))
    model.to(device)
    model.eval()

    # ckecking for GPU for Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    t_size = cv2.getTextSize(" ", cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    for imagename in os.listdir(opt.input_file_path):

        print("\n"+imagename+"_______")
        image_path = os.path.join(opt.input_file_path, imagename)
        org_img = cv2.imread(image_path)
        i_height, i_width = org_img.shape[:2]
        x = y = i_height if i_height > i_width else i_width
        img = np.zeros((x, y, 3), np.uint8)
        start_new_i_height = int((y - i_height) / 2)
        start_new_i_width = int((x - i_width) / 2)
        img[start_new_i_height: (start_new_i_height + i_height) ,start_new_i_width: (start_new_i_width + i_width) ] = org_img
        img = cv2.resize(img, (opt.img_size, opt.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img) / 255
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = torch.Tensor(img).to(device)

        with torch.no_grad():
            detections = model(img)
    
        outputs = [{k: v.to(device) for k, v in t.items()} for t in detections]
        id2class = {0: 'No Mask', 1: 'Mask'}
    
        # For each detection in detections
        detection = outputs[0]
        mul_constant = x / opt.img_size
        if detection is not None:
            # for [x1, y1, x2, y2], conf,  cls_pred in detection:
            boxes = detection['boxes'].cpu().detach().numpy().astype(int)
            labels = detection['labels'].cpu().detach().numpy()
            scores = detection['scores'].cpu().detach().numpy()
            all = np.c_[boxes,scores]
            keep = py_cpu_nms(all, 0.1)
            all = np.c_[boxes, labels,scores]
            for i in keep:
                if all[i][5]<0.2:
                    continue
                x1 = int(all[i][0])
                y1 = int(all[i][1])
                x2 = int(all[i][2])
                y2 = int(all[i][3])

                if all[i][4] ==2:
                    labels = 1
                else:
                    labels = 0
                    
                x1 = int(x1 * mul_constant - start_new_i_width)
                y1 = int(y1 * mul_constant - start_new_i_height)
                x2 = int(x2 * mul_constant - start_new_i_width)
                y2 = int(y2 * mul_constant - start_new_i_height)
    
                # Bounding box making and setting Bounding box title
                if int(labels) == 0:
                    # WITH_MASK
                    cv2.rectangle(org_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    # WITHOUT_MASK
                    cv2.rectangle(org_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
                cv2.putText(org_img, id2class[int(labels)] + ": %.2f" % all[i][5], (x1, y1 + t_size[1]),
                            cv2.FONT_HERSHEY_PLAIN, 1,
                            [225, 255, 255], 2)
        out_filepath = os.path.join(opt.output_path, imagename)
        cv2.imwrite(out_filepath,org_img)

        print("Done....")

    cv2.destroyAllWindows()
    
    
    








