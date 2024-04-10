DATA_PATH  = '/content/drive/MyDrive/Thermal-Cheetah-1'
TRAIN = DATA_PATH + '/train'
TEST = DATA_PATH + '/test'

import json

annotations = {}

with open(os.path.join(TRAIN,'_annotations.txt'), 'r') as file:
    for line in file:
        line_parts = line.strip().split(' ')
        image_name = line_parts[0]
        bboxes = []
        for bbox_str in line_parts[1:]:
            bbox = [int(x) for x in bbox_str.split(',')]
            bboxes.append(bbox)
        annotations[image_name] = bboxes

json_data=json.dumps(annotations, indent=2)
with open(os.path.join(TRAIN,'_annotations.json'), 'w') as json_file:
    json_file.write(json_data)

annotations = {}

with open(os.path.join(TEST,'_annotations.txt'), 'r') as file:
    for line in file:
        line_parts = line.strip().split(' ')
        image_name = line_parts[0]
        bboxes = []
        for bbox_str in line_parts[1:]:
            bbox = [int(x) for x in bbox_str.split(',')]
            bboxes.append(bbox)
        annotations[image_name] = bboxes

json_data=json.dumps(annotations, indent=2)
with open(os.path.join(TEST,'_annotations.json'), 'w') as json_file:
    json_file.write(json_data)


from PIL import Image

MAX_N = 10
categories = [
    {
        "supercategory": "none",
        "name": "cheetah",
        "id": 0
    },
    {
        "supercategory": "none",
        "name": "human",
        "id": 1
    }
]

phases = ["train", "test"]
for phase in phases:
  root_path = "/content/drive/MyDrive/Thermal-Cheetah-1/{}".format(phase)
  annotations = json.load(open('/content/drive/MyDrive/Thermal-Cheetah-1/{}/_annotations.json'.format(phase)))
  json_file = "/content/drive/MyDrive/Thermal-Cheetah-1/{}.json".format(phase)

  file_list = [i for i in os.listdir(root_path) if not i.endswith('.txt')  and not i.endswith('.json')]


  res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

  annot_count = 0
  image_id = 0
  processed = 0
  for file_idx,image_name in enumerate(file_list):
    file_path = image_name
    bboxes = annotations[file_path]
    num_bboxes = len(bboxes)

    if num_bboxes > MAX_N:
      continue

    img_path = os.path.join(root_path, file_path)
    img = Image.open(img_path)
    img_w, img_h = img.size

    img_elem = {"file_name": img_path,
                "height": img_h,
                "width": img_w,
                "id": image_id}

    res_file["images"].append(img_elem)

    for i in range(num_bboxes):
      w,h = bboxes[i][2] , bboxes[i][3]
      xmin = int(bboxes[i][0] - w/2)
      ymin = int(bboxes[i][1] - h/2)
      xmax = int(bboxes[i][0] + w/2)
      ymax = int(bboxes[i][1] + h/2)
      area = w * h
      poly = [[xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]]

      category_id = int(bboxes[i][-1])

      annot_elem = {
          "id": annot_count,
          "bbox": [
              float(xmin),
              float(ymin),
              float(w),
              float(h)
              ],
          "segmentation": list([poly]),
          "image_id": image_id,
          "ignore": 0,
          "category_id": category_id,
          "iscrowd": 0,
          "area": float(area)
          }
      res_file["annotations"].append(annot_elem)
      annot_count += 1
    image_id += 1
    processed += 1

  with open(json_file, "w") as f:
    json_str = json.dumps(res_file)
    f.write(json_str)
  print("Processed {} {} images...".format(processed, phase))
print("Done.")