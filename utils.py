import torch
import copy
import itertools
import torch.nn as nn
import cv2 


def convert_bbox(bbox):
  x_center,y_center,width,height,_=bbox
  x1=x_center - width//2
  x2=x_center + width//2
  y1=y_center - height//2
  y2=y_center + height//2
  return x1,y1,x2,y2

def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):
  def __init__(self,tensor,mask=None):
    self.tensor=tensor
    self.mask=mask
  def decompose(self):
    return self.tensor,self.mask
  def to(self,device):
    cast_tensor = self.tensor.to(device)
    if self.mask is not None:
      assert self.mask is not None
      cast_mask=self.mask.to(device)
    else:
      cast_mask=None
    return NestedTensor(cast_tensor,cast_mask)

def nested_tensor_from_tensor_list(tensor_list:List[torch.Tensor]):
  if tensor_list[0].ndim == 3:
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    b,c,h,w = batch_shape
    device = tensor_list[0].device
    dtype = tensor_list[0].dtype
    tensor = torch.zeros(batch_shape,dtype=dtype,device=device)
    mask = torch.ones((b,h,w),dtype=torch.bool,device=device)

    for img,pad_img,m in zip(tensor_list,tensor,mask):
      pad_img[:img.shape[0],:img.shape[1],:img.shape[2]].copy_(img)
      mask[:img.shape[1],:img.shape[2]] = False
  else:
    raise ValueError('Not supported!')
  return NestedTensor(tensor,mask)

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

def stack_layers(layer,m):
  return nn.ModuleList([copy.deepcopy(layer) for i in range(m)])

def visualize_bboxes(annots,n):
  annots = dict(itertools.islice(annots.items(), n))
  for image_path, boxes in annots.items():
    image = cv2.imread(TRAIN+'/'+image_path)
    for box in boxes:
      x1, y1, x2, y2 = convert_bbox(box)
      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow(image)

def read_n_images(annots,n):
  annots = parse_file(annots)
  annots = dict(itertools.islice(annots.items(), n))
  images=[]
  for image_path,_ in annots.items():
    image=torch.tensor(cv2.imread(TRAIN+'/'+image_path)).permute(2,0,1)
    images.append(image)
  return images