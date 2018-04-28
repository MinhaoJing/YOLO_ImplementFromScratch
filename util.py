# coding=utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2

def predict_transform(prediction, input_dim, anchors, num_classes, CUDA):
