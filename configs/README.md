# 大图标注的文件，自动切图，保存对应的坐标

## 文件说明
1. cut_image.py(动态切图)
```text
将labelme标注的文件(.jpg, .json在同一个文件夹内)。
处理成小图(即一张大图切割成多张小图和标签,结果还是生成labelme格式的数据)

输入参数为:
参数1：原文件(原始labelme文件)(jpg/json)存放目录  参数2：目标文件(jpg/json)存放目录
(下一步骤: 运行labelme2coco.py生成coco格式数据)
```
2. labelme2coco.py
```text
step1: 将1.cut_image.py生成的labelme格式的数据转换成coco格式的数据
step2: 将原始的labelme格式数据转换成coco格式数据

输入参数为:
参数1：当前json文件路径  参数2：当前img图片路径  参数3：输出的coco路径
```
3. cut_image2.py(sahi切图)
+ 参考链接: https://blog.csdn.net/LuohenYJ/article/details/128538834
```text
使用sahi(切片辅助超推理库)切割大图生成小目标
使用之前将labelme格式数据转换成coco格式数据

输入参数为:
参数1：原图像大图的coco格式数据   参数2：输出图像的coco格式数据
(上一步骤：将原始的labelme数据通过labelme2coco.py转换成coco格式 luhou->coco_cut2)
该步骤： 将coco_cut2->coco_cut3
```
4. merge_coco.py
```text
合并动态coco与sahi的coco

参数1：动态的coco文件coco_cut1 参数2：sahi的coco文件coco_cut3
将 coco_cut1, coco_cut3 -> coco_cut4(最终文件)
```
5. coco2yolov5.py
```text
使用:
# train2017 训练集
# - 图片：目录软链到 images/
# - 标注：转换存储进 labels/*.txt
# - 物体类型：全部记录进 *.names
# - 图片列表：有物体标注的记录进 *.txt, 无的进 *.txt.ignored
python scripts/coco2yolov5.py \
--coco_img_dir $COCO_DIR/train2017/ \
--coco_ann_file $COCO_DIR/annotations/instances_train2017.json \
--output_dir $OUTPUT_DIR

# val2017 验证集
# - 物体类型：依照训练集的记录，保证顺序
python scripts/coco2yolov5.py \
--coco_img_dir $COCO_DIR/val2017/ \
--coco_ann_file $COCO_DIR/annotations/instances_val2017.json \
--output_dir $OUTPUT_DIR \
--obj_names_file $OUTPUT_DIR/train2017.names

格式：
coco2017_yolov5/
├── images
│   ├── train2017 -> /home/john/datasets/coco2017/train2017
│   └── val2017 -> /home/john/datasets/coco2017/val2017
├── labels
│   ├── train2017
│   └── val2017
├── train2017.names
├── train2017.txt
├── train2017.txt.ignored
├── val2017.txt
└── val2017.txt.ignored
```

## 生成文件说明
```text
coco_cut1: 动态选取框生成的coco文件
coco_cut2: 原始lebelme文件转换得到的coco文件
coco_cut3: 使用sahi滑动窗口生成的coco文件

coco_cut4: 将动态选取coco[coco_cut1]和sahi滑动生成的coco[coco_cut3]进行融合得到最终的coco文件
```