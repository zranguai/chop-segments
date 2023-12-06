# 合并动态coco与sahi的coco
import json
import os
import shutil
from glob import glob


# 清空path目录下的所有目录和文件
def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


def copy_file(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname=os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + fname)   # 复制文件
        # print("copy %s -> %s" % (srcfile, dstpath + fname))


def merge_coco(src_dir1, src_dir2, dst_file_dir):
    src_file_json1 = os.path.join(src_dir1, "annotations/train_cut1.json")
    src_file_json2 = os.path.join(src_dir2, "annotations/instances_train2017.json")
    f1 = open(src_file_json1, mode="r", encoding="utf-8")  # dynamic json(相对较少)
    f2 = open(src_file_json2, mode="r", encoding="utf-8")  # sahi json
    j_f1 = json.loads(f1.read())
    j_f2 = json.loads(f2.read())

    # 将不变的先保存到newj_f3里面
    newj_f3 = {}
    newj_f3["categories"] = j_f2["categories"]

    base_img_id = len(j_f2["images"])
    base_json_id = len(j_f2["annotations"])
    # 对j_f1的json文件进行改造
    for img in j_f1["images"]:
        img["id"] = img["id"] + base_img_id
    for anno in j_f1["annotations"]:
        anno["id"] = anno["id"] + base_json_id
        anno["image_id"] = anno["image_id"] + base_img_id

    # ===================sahi生成的images冗余过多，根据annotations进行过滤===============
    j_f2_new_images = list()
    j_f2_copy_img_files = list()  # 后面需要拷贝的图片名称
    image_id_lists = list()
    for anno in j_f2["annotations"]:
        image_id = anno["image_id"]
        # 防止重复包含数据
        if image_id not in image_id_lists:
            image_id_lists.append(image_id)
            j_f2_new_images.append(j_f2["images"][image_id - 1])
            j_f2_copy_img_files.append(j_f2["images"][image_id - 1]["file_name"])
    j_f2["images"] = j_f2_new_images  # images进行修改
    # ==============================================================================

    # 将j_f1的信息加到j_f2里面
    j_f2["images"].extend(j_f1["images"])
    j_f2["annotations"].extend(j_f1["annotations"])

    # 结果存到newj_f3中
    newj_f3["images"] = j_f2["images"]
    newj_f3["annotations"] = j_f2["annotations"]

    # 写进文件
    annotation_dir = os.path.join(dst_file_dir, "annotations")
    clean_dir(annotation_dir)  # 清空或者生成文件
    dst_file_json = os.path.join(annotation_dir, "instances_train2017.json")
    with open(dst_file_json, mode="w", encoding="utf-8") as out:
        out.write(json.dumps(newj_f3, ensure_ascii=False))

    # 拷贝文件  11月27号修改，拷贝图片应该按照instances_train2017来拷贝，要不然拷贝太多了
    src_train_dir1 = os.path.join(src_dir1, "train/")
    src_train_dir2 = os.path.join(src_dir2, "train/")
    dst_train_dir = os.path.join(dst_file_dir, "train/")
    clean_dir(dst_train_dir)  # 清空或者生成文件
    src_file1_list = glob(src_train_dir1 + "*")  # glob获得路径下所有文件，可根据需要修改
    src_file2_list = glob(src_train_dir2 + "*")

    for srcfile1 in src_file1_list:
        copy_file(srcfile1, dst_train_dir)

    # sahi切图的图片路径进行优化
    for j_f2_copy_img in j_f2_copy_img_files:
        dst_full_copy_img = os.path.join(src_train_dir2, j_f2_copy_img)
        # 拷贝图片
        copy_file(dst_full_copy_img, dst_train_dir)

    # for srcfile2 in src_file2_list:
    #     copy_file(srcfile2, dst_train_dir)


def main(dynimic_coco_dir = r"./coco_cut1/",
         sahi_coco_dir = r"./coco_cut3/",
         merge_coco_dir = "./coco_cut4"
         ):
    src_dir1 = dynimic_coco_dir
    src_dir2 = sahi_coco_dir

    dst_file_dir = merge_coco_dir
    clean_dir(dst_file_dir)  # 清空或者生成文件
    merge_coco(src_dir1, src_dir2, dst_file_dir)


if __name__ == '__main__':
    main()
