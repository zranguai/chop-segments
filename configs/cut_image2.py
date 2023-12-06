# 使用sahi(切片辅助超推理库)切割大图生成小目标
import os
import cv2
import shutil

from configs.utils import slice_coco
from configs.utils import load_json


# 清空path目录下的所有目录和文件
def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


# 切割数据集
def cut_coco(coco_dict, coco_annotation_file_path, image_path, output_coco_annotation_file_name, output_dir,
             slice_height, slice_width):
    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=coco_annotation_file_path,
        image_dir=image_path,
        output_coco_annotation_file_name=output_coco_annotation_file_name,
        ignore_negative_samples=False,
        output_dir=output_dir,
        slice_height=slice_height,  # 切割出来图的尺度
        slice_width=slice_width,
        overlap_height_ratio=0.2,  # 重合度
        overlap_width_ratio=0.2,
        min_area_ratio=0.2,
        verbose=False
    )

    print("切分子图{}张".format(len(coco_dict['images'])))
    print("获得标注框{}个".format(len(coco_dict['annotations'])))
    return coco_dict, coco_path


# 可视化切出来的图
def visual_image(coco_dict, output_dir):
    for index, img in enumerate(coco_dict['images']):
        # img = Image.open(os.path.join(output_dir, img["file_name"]))
        img = cv2.imread(os.path.join(output_dir, img["file_name"]))
        for ann_ind in range(len(coco_dict["annotations"])):
            # 搜索与当前图像匹配的边界框
            if coco_dict["annotations"][ann_ind]["image_id"] == coco_dict["images"][index]["id"]:  # 该标签是该图片的
                xywh = coco_dict["annotations"][ann_ind]["bbox"]
                xyxy = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
                # 绘图
                # 画框
                cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), thickness=3,
                              lineType=cv2.LINE_8)
                # 画点
                segmentation_point = coco_dict["annotations"][ann_ind]["segmentation"][0]
                point_len = len(segmentation_point)
                for point_index in range(0, point_len, 2):
                    point = segmentation_point[point_index: point_index + 2]
                    cv2.circle(img, point, 2, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
                    # print(point)

                # ImageDraw.Draw(img, 'RGBA').rectangle(xyxy, width=5)
        # axarr[int(index / axarr_col), int(index % axarr_col)].imshow(img)
        cv2.imshow("cut_img", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def main(cut_mode,
         image_path = "./../cut_image/coco_cut2/train",# 切割的coco图片路径
         coco_annotation_file_path = "./../cut_image/coco_cut2/annotations/train_cut1.json",# 切割的coco的json路径
         chop_size=(640, 640)
         ):
    coco_path = os.path.dirname(os.path.dirname(image_path)) + '/coco_cut3' #"./cut_image/coco_cut3"
    clean_dir(coco_path)
    coco_path_annotations = os.path.join(coco_path, "annotations")
    clean_dir(coco_path_annotations)
    if "train" in cut_mode:
        # --------------------------训练集切割---------------------------
        print(f"\033[1;34m 开始切割训练集数据集\033[0m")

        coco_dict = load_json(coco_annotation_file_path)
        # 保存的coco数据集标注文件名
        output_coco_annotation_file_name = "train_2017"
        # 输出文件夹
        output_dir = coco_path + "/train"
        clean_dir(output_dir)

        # print(image_path)
        # 切分数据集
        coco_dict, coco_path = cut_coco(coco_dict,
                                        coco_annotation_file_path,
                                        image_path,
                                        output_coco_annotation_file_name,
                                        output_dir,
                                        slice_height=chop_size[0],
                                        slice_width=chop_size[1]
                                        )
        # 将json文件移动到coco_cut/annotations里面
        src_json_file = coco_path
        dst_json_file = os.path.join(coco_path_annotations, "instances_train2017.json")
        shutil.move(src_json_file, dst_json_file)
        # 可视化数据集(optional)
        # visual_image(coco_dict, output_dir)
    if "val" in cut_mode:
        # --------------------------验证集切割---------------------------
        print(f"\033[1;34m 开始切割验证集数据集\033[0m")
        image_path = r"./coco-ji/val2017"
        coco_annotation_file_path = r"./coco-ji/annotations/instances_val2017.json"
        coco_dict = load_json(coco_annotation_file_path)
        # 保存的coco数据集标注文件名
        output_coco_annotation_file_name = "val_2017"
        # 输出文件夹
        output_dir = "coco_cut/val2017"
        clean_dir(output_dir)
        # 切分数据集
        coco_dict, coco_path = cut_coco(coco_dict, coco_annotation_file_path, image_path,
                                        output_coco_annotation_file_name, output_dir)
        # 将json文件移动到coco_cut/annotations里面
        src_json_file = coco_path
        dst_json_file = os.path.join(coco_path_annotations, "instances_val2017.json")
        shutil.move(src_json_file, dst_json_file)
        # 可视化数据集(optional)
        # visual_image(coco_dict, output_dir)


if __name__ == '__main__':
    # cut_mode = ["train", "val"]
    cut_mode = ["train"]
    main(cut_mode)
