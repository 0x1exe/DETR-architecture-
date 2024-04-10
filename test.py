import torch
from .utils import * 
from .transforms import * 
from .transformer import *
from .backbone import *
from .detr import DETR
import os
from dataset import *
from torch.utils.data import DataLoader

base = Backbone('resnet18',True,False,False)
pos_encoding = PositionalEncoding(base.num_channels // 2)
backbone = Joiner(base,pos_encoding)
backbone.num_channels=base.num_channels
transformer = Transformer()
model = DETR(
    backbone,
    transformer,
    num_classes=2,
    num_queries=100,
    )

normalize = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_train = CheetahDetection(TRAIN,os.path.join(DATA_PATH,'train.json'),transforms = normalize)

sampler_train = torch.utils.data.RandomSampler(dataset_train)
batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, 16 , drop_last=True)
data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,collate_fn=collate_fn)

for data,target in data_loader_train:
  out = model(data)
  break