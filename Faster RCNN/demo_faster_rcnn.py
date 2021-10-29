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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3_mask.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_self.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/mask_dataset.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--frame_size", type=int, default=416, help="size of each image dimension")

    opt = parser.parse_args()
    print(opt)


    # test_path = './demo'
    def py_cpu_nms(dets, thresh):
        """Pure Python NMS baseline."""
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


    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("正在使用的设备是",device)
    # 3 classes, background, face，face_mask
    num_classes = 3
    BATCH_SIZE = 1

    # get the model using our helper function
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
    model.load_state_dict(torch.load("checkpoints_faster_rcnn/yolov3_ckpt_0__'1'.pth"))
    # move model to the right device
    model.to(device)
    model.eval()

    # ckecking for GPU for Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # camara capture
    # "http://192.168.43.19:8080/video"
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Cannot capture source'

    print("\nPerforming object detection:")

    # Video feed dimensions
    _, frame = cap.read()
    v_height, v_width = frame.shape[:2]

    # For a black image
    x = y = v_height if v_height > v_width else v_width

    # Putting original image into black image
    start_new_i_height = int((y - v_height) / 2)
    start_new_i_width = int((x - v_width) / 2)

    # For accommodate results in original frame
    mul_constant = x / opt.frame_size

    # for text in output
    t_size = cv2.getTextSize(" ", cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

    frames = fps = 0
    start = time.time()

    while _:

        # frame extraction => resizing => [BGR -> RGB] => [[0...255] -> [0...1]] => [[3, 416, 416] -> [416, 416, 3]]
        #                       => [[416, 416, 3] => [416, 416, 3, 1]] => [np_array -> tensor] => [tensor -> variable]

        # frame extraction
        _, org_frame = cap.read()
        # resizing to [416 x 416]

        # Black image
        frame = np.zeros((x, y, 3), np.uint8)

        frame[start_new_i_height: (start_new_i_height + v_height),
        start_new_i_width: (start_new_i_width + v_width)] = org_frame

        # resizing to [416x 416]
        frame = cv2.resize(frame, (opt.frame_size, opt.frame_size))
        # [BGR -> RGB]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # [[0...255] -> [0...1]]
        frame = np.asarray(frame) / 255
        # [[3, 416, 416] -> [416, 416, 3]]
        frame = np.transpose(frame, [2, 0, 1])
        # [[416, 416, 3] => [416, 416, 3, 1]]
        frame = np.expand_dims(frame, axis=0)
        # [np_array -> tensor]
        frame = torch.Tensor(frame).to(device)

        # plt.imshow(frame[0].permute(1,2,0))
        # plt.show()

        # [tensor -> variable]

        # Get detections
        with torch.no_grad():
            detections = model(frame)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in detections]
        # boxes = outputs[0]['boxes'].cpu().detach().numpy().astype(int)
        # labels = outputs[0]['labels'].cpu().detach().numpy()
        # scores = outputs[0]['scores'].cpu().detach().numpy()
        # all = np.c_[boxes,scores]
        # keep = py_cpu_nms(all, 0.05)
        id2class = {0: 'No Mask', 1: 'Mask'}

        # For each detection in detections
        detection = outputs[0]
        if detection is not None:
            # for [x1, y1, x2, y2], conf,  cls_pred in detection:
            boxes = detection['boxes'].cpu().detach().numpy().astype(int)
            labels = detection['labels'].cpu().detach().numpy()
            print("lable",labels)
            scores = detection['scores'].cpu().detach().numpy()
            all = np.c_[boxes,scores]
            keep = py_cpu_nms(all, 0.02)
            all = np.c_[boxes, labels,scores]
            for i in keep:
                if all[i][5]<0.6:
                    continue
                x1 = int(all[i][0])
                y1 = int(all[i][1])
                x2 = int(all[i][2])
                y2 = int(all[i][3])
                print(all[i][4])
                if all[i][4] ==2:
                    labels = 1


                else:
                    labels = 0

                print("x1的值",x1)
                # boxes = outputs[0]['boxes'].cpu().detach().numpy().astype(int)
                # labels = outputs[0]['labels'].cpu().detach().numpy()
                # scores = outputs[0]['scores'].cpu().detach().numpy()
                # Accommodate bounding box in original frame
                x1 = int(x1 * mul_constant - start_new_i_width)
                y1 = int(y1 * mul_constant - start_new_i_height)
                x2 = int(x2 * mul_constant - start_new_i_width)
                y2 = int(y2 * mul_constant - start_new_i_height)

                # Bounding box making and setting Bounding box title
                if (int(labels) == 0):
                    # WITH_MASK
                    cv2.rectangle(org_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    # WITHOUT_MASK
                    cv2.rectangle(org_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                cv2.putText(org_frame, id2class[int(labels)] + ": %.2f" % all[i][5], (x1, y1 + t_size[1]),
                            cv2.FONT_HERSHEY_PLAIN, 1,
                            [225, 255, 255], 2)

        # CURRENT TIME SHOWING
        now = datetime.datetime.now()
        current_time = now.strftime("%H:%M:%S")

        # FPS PRINTING
        cv2.rectangle(org_frame, (0, 0), (175, 20), (0, 0, 0), -1)
        cv2.putText(org_frame, current_time + " FPS : %3.2f" % (fps), (0, t_size[1] + 2),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    [255, 255, 255], 1)

        frames += 1
        fps = frames / (time.time() - start)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN - 5, cv2.WINDOW_FULLSCREEN)

        cv2.imshow('frame', org_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




