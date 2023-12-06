# 统计标注中不同类别的面积，方便后续切割操作
import cv2
import json
import glob
import numpy as np


def computer_minwh(points):
    """
    求点的最小内接矩形
    """
    for i in range(len(points)):
        for j in range(2):
            points[i][j] = int(points[i][j])
    # 计算最小外接矩
    min_rect = cv2.minAreaRect(np.array(points))
    # rect_center_xy = list(min_rect[0])
    rect_wh = list((min_rect[1][0], min_rect[1][1]))
    # rect_angle = min_rect[2]
    return min(rect_wh)


def each_json_category_area(new_j, category_area_dict):
    shapes = new_j["shapes"]
    for shape in shapes:
        label = shape["label"]
        min_wh = computer_minwh(shape["points"])
        if label not in category_area_dict.keys():  # 不在，更新进去
            category_area_dict[label] = min_wh
        else:  # 在， 选最小的更新
            if category_area_dict[label] > min_wh:
                category_area_dict[label] = min_wh


def statis_category_areas_main(src_file_dir=""):
    """
    读取src目录下的json
    """
    src_jsons = glob.glob(src_file_dir + "/*.json")
    category_area_dict = dict()
    for src_json in src_jsons:
        # json读取
        try:
            with open(src_json, mode="r", encoding="utf-8") as f_src:
                src_j = json.loads(f_src.read())

                new_j = {}
                new_j['version'] = src_j['version']
                new_j['flags'] = src_j['flags']
                new_j['shapes'] = src_j['shapes']
                new_j['imagePath'] = src_j['imagePath']
                new_j['imageData'] = src_j['imageData']
                new_j['imageHeight'] = src_j['imageHeight']
                new_j['imageWidth'] = src_j['imageWidth']

            # 求该json中每个类别的面积
            each_json_category_area(new_j, category_area_dict)

        except Exception as e:
            print(f"\033[1;34m Error:{e}, 该{src_json}异常，进行异常处理\033[0m")
            continue
    # 根据最小的宽或高确定切割的大小，这里分为小 中 大 三个等级
    """
    小目标: 小于32*32个像素点 切割的尺寸为: 640*640
    中目标: 32*32 - 96*96   切割的尺寸为: 960*960
    大目标: 大于96*96        切割的尺寸为: 1280*1280
    """
    chop_size = {"small": (640, 640), "medium": (960, 960), "large": (1280, 1280)}
    # print(category_area_dict)
    smallest = float("inf")  # 一个最大的值
    for small in category_area_dict.keys():
        if category_area_dict[small] < smallest:
            smallest = category_area_dict[small]

    if smallest < 32.0:
        tag = "small"
    elif 32.0 < smallest < 96.0:
        tag = "medium"
    elif smallest > 96.0:
        tag = "large"
    return chop_size[tag]


if __name__ == '__main__':
    src_file_dir = r"../Images"
    statis_category_areas_main(src_file_dir=src_file_dir)
