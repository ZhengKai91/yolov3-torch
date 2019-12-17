# Yolov3-note
## build_target
input: targets, which is a M*6 tensor. M is the number of gt boxes. each box is (b, cls, gx, gy, gw, gh). b is the index of images in batch this box belongs to. cls is the index of classes. gx, gy is the center of box normed to 0~1, gw, gh is the width and height of box normed to 0~1. 
The predicts is a tensor of size Nx3xgxgx85, which include all infos of Nx3xgxg achors. What build target need to do is to find out which anchors need to participate in computing loss, and theirs' coresponding target label. 
If we calculate the boxes one by one:
```python
    for gtbox in targets:
    gx, gy, gw, gh = gtbox * g
    gi , gj = long(gx, gy)
    b = gtbox[0]
    a = []
    for ai, anchor in enumerate(scaled_anchors):
        if iou_wh(ai, (gw, gh)) > iou_thres:
            a.append(ai)
    #a =[0, 1..] 
    for ai in a:
        tobj[b, ai, gj, gi] = 1
```
tobj is the mask of anchors. which is Nx3xgxg.
But we can do it more like python and more effiently. 
We have 3M anchor at most to take paticipate in computing loss. In origin YOLO, we only need to select M anchor. 
```python
    #orign YOLO, anchors index [b, a, gj, gi]
    b = target[0] # get M b
    gj, gi = long(gx, gy)# get M gj, gi
    # what left is a
    iou = torch.stack([wh_iou(x, gwh) for x in anchor_vec], 0)# compute 3xM ious at once
    iou, a = iou.max(0)# M anchor select
```
```python
    # if want to compute more anchor
    iou = torch.stack([wh_iou(x, gwh) for x in anchor_vec], 0)# compute 3xM ious at once
    iou = iou.view(-1) #3M ious
    a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1) # the index of anchor  s.t ious
    t = targets.repeat([na, 1])# 3M targets s.t ious
    ghw = ghw.repeat([na,1])# 3M scale h,w s.t ious
    t, a, gwh = select_anchors_by_iou()

    b = t[:, 0]
    gi , gj = t[:, 2:4]*g
    # now we have [b, a, gj, gi]
```