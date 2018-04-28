# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.split('#')[0] for x in lines]
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []
    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].strip()
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks.append(block)

    return blocks


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    filters_list = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        # 如果block的类型是卷积层
        if (x["type"] == "convolutional"):
            filters = int(x["filters"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            pad = x["pad"]
            activation = x["activation"]

            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            if pad == '1':
                padding = ((kernel_size - 1) * stride) // 2
            else:
                padding = 0
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, padding, bias=bias)
            module.add_module("cnv_{}".format(index), conv)
            if batch_normalize:
                module.add_module("batch_norm_{}".format(index), nn.BatchNorm2d(filters))
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)
        # 如果block的类型是上采样层
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)
        # 如果block的类型是route
        elif (x["type"] == "route"):
            layers = x["layers"].split(',')
            start = int(layers[0])
            try:
                end = int(layers[1])
            except:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{}".format(index), route)
            # 这里的意思应该是end和start的索引值都不应该大于该route层的索引值
            if end < 0:
                filters = filters_list[index + start] + filters_list[index + end]
            else:
                filters = filters_list[index + start]
        # 如果block的类型是shortcut
        elif (x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
        # 如果block的类型是yolo
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        # 这里需要特别注意一下，如果遇到upsample和shortcut层的话,其实filters的数量是不会改变的,
        # 而如果以此循环中filter没有被赋值，则其值保持上依次循环的值不变
        prev_filters = filters
        filters_list.append(filters)
    return (net_info, module_list)


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)
    def forward(self, x, CUDA):
        modules=self.blocks[1:]
        # 用来feature maps(用于shortcut层和route层),
        # key是层的索引值,value值是相应的feature maps
        outputs={}
        # 这是一个标志,用于yolo层部分
        write=0
        for i,module in modules:
            module_type=module["type"]
            if module_type=="convolutional" or module_type == "upsample":
                x=self.module_list[i](x)
            elif module_type == "route":
                layers = module["layers"]
                layers=[int(x) for x in layers]
                if layers[0]>0:
                    layers[0]-=i
                if len(layers)==1:
                    x=outputs[i+layers[0]]
                else:
                    if layers[1]>0:
                        layers[1]-=i
                    feat_map0=outputs[i+layers[0]]
                    feat_map1 = outputs[i + layers[1]]

                    x=torch.cat((feat_map0,feat_map1),1)
            elif module_type== "shortcut":
                from_=int(module["from"])
                x=outputs[i-1]+outputs[i+from_]
            elif module_type=="yolo"
                pass
                
        outputs[i]=x



if __name__ == "__main__":
    blocks = parse_cfg("./cfg/yolov3.cfg")
    net_info,modules = create_modules(blocks)
    print(modules)
