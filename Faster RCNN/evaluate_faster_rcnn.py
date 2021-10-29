import torch
import torchvision
from faster_rcnn_utils.engine import evaluate
from faster_rcnn_utils.AIZOODataset import AIZOODataset
from faster_rcnn_utils.transforms import get_transform
from faster_rcnn_utils import utils
import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'



def eva_rcnn(model,data_loader_val, device):
    test_path = './val'

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model.to(device)

    # evaluate on the test dataset
    coco_evaluator=evaluate(model, data_loader_val, device=device)
    mAP_one=None
    for coco_Evaluator in coco_evaluator.coco_eval.values():
        mAP_one=np.mean([coco_Evaluator.eval['precision'][:,0:101:10,0,0,2].mean(),coco_Evaluator.eval['precision'][:,0:101:10,1,0,2].mean()])
    return mAP_one