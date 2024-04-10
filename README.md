# DETR [architecture implementation from scratch]
This is my attempt at implementing DETR acrhitecture from scratch with basis on official repo.
I used a custom detection dataset (Cheetah detection) in YoloV4 format and converted it to COCO, implemented the main components of an architecture.
I excluded some lines and parts which were redundant for my main goal, which is: explore the logic begind the model and attempt to write forward pass by myself of a custom dataset.
I didn't attemp to train the model due to hardware restrictions, but I achieved my main goal, as everything works and the shapes are as they should be.
