# 大图切小图训练
+ 根据labelme标注的情况，动态选择切图的尺寸(分割标注，可按需改造成检测标注)

```
input-datasets: labelme标注好的images/jsons文件放到这里
例如:
1.img
1.json
2.img
2.json

output-datasets:
切割好的img和json生成到这里

workspace:
中间生成的辅助文件夹包括coco格式和yolo格式，按需取用
```