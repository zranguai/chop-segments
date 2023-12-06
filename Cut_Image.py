import shutil

import configs.cut_image as cut_image
import configs.labelme2coco as labelme2coco
import configs.cut_image2 as cut_image2
import configs.merge_coco as merge_coco
# import configs.coco2yolov5 as coco2yolov5  # 这个版本不用，只有box的转化，没有分割点的转化
import configs.general_json2yolo as coco2yolov5  # 这个版本box和seg转化都有
import configs.cut_chop_batch_img as cut_chop_image
import configs.statistic_category_areas as statistic_chop_size
import configs.yoloseg2labelme as yoloseg2labelme

import glob
import os
import os.path as osp
import argparse
import time


#清空文件夹下的文件
def delete_dir(str_dir):
    ls = os.listdir(str_dir)
    for name in ls:
        c_path = os.path.join(str_dir,name)
        if(os.path.isdir(c_path)):
            delete_dir(c_path)
        else:
            if c_path.endswith(".jpg") or c_path.endswith(".json") or c_path.endswith(".txt") or\
               c_path.endswith(".pth") or c_path.endswith(".onnx") or c_path.endswith(".torchscript") or\
               c_path.endswith(".pt") or c_path.endswith(".png") or c_path.endswith(".csv") or c_path.endswith(".0")or\
               c_path.endswith(".yaml"):
                os.remove(c_path)

    # if os.path.exists(str_dir): #改操作太过于危险
    #     shutil.rmtree(str_dir)


# copy一个文件夹下文件到例外一个文件
def copy_dir(img_dir, dst_dir):
    import shutil
    img_paths = glob.glob(img_dir + "/*")
    for img_path in img_paths:
        shutil.copy(img_path, dst_dir)


#裁图功能 --> 优化任意尺寸
class CutImage:
    def __init__(self, input_dir='./Common/mydata/data',output_dir = './Common/mydata/cut_image/txt_cut4'):
        self.sorce_dir = input_dir
        self.dst_dir = output_dir

    #动态切图
    def dynamic_cut_image(self,src_file_dir ='./test',
                          dst_file_dir = './dst_data'):
        #动态切图
        cut_image.clean_dir(dst_file_dir)
        # 获取文件名
        img_names = cut_image.img_file_list(src_file_dir)

        # ############################参数设置############################
        # 1.slide_window函数中的滑动步长
        step_eachx = 100
        step_eachy = 100
        # 2.merge_or_cut函数中的iou
        iou_conf = 0.1  # 两个框重合度较小就跳过
        # 3.main_process中的框的变化范围
        box_range = [[0, 640], [640, 960], [960, 1280], [1280, 1600], [1600, 1920], [1920, 2240], [2240, 2560],
                     [2560, 2880], [2880, 3200], [3200, 3520]]
        ################################################################
        cut_image.main_process(src_file_dir, dst_file_dir, img_names, step_eachx, step_eachy, iou_conf, box_range)

        print("Dynamic Cut Image ALL DONE")

    def get_chop_size(self,
                      src_file_dir="./test"):
        chop_size = statistic_chop_size.statis_category_areas_main(src_file_dir=src_file_dir)
        return chop_size

    def chop_cut_image(self,
                       src_file_dir='./test',
                       dst_file_dir='./dst_data',
                       chop_size=(640, 640),
                       shave=0):
        cut_image.clean_dir(dst_file_dir)
        cut_chop_image.chop_cut_main(src_file_dir=src_file_dir,
                                     dst_file_dir=dst_file_dir,
                                     chop_size=chop_size,
                                     shave=shave)

    #动态切图结果转coco格式
    def labelme2coco_fun(self,file_dir='../data/luhou',
                         dst_coco_annota_path='coco_cut2/annotations',
                         gen_json_name="train_cut1.json"):
        # print('waiting for labelme2coco...')
        labelme_json = glob.glob(file_dir + '/*.json')

        if not osp.exists(dst_coco_annota_path):
            # os.mkdir(dst_coco_annota_path)
            os.makedirs(dst_coco_annota_path)

        up_path = osp.dirname(dst_coco_annota_path)
        train_path = os.path.join(up_path, "train")
        if not osp.exists(train_path):
            os.mkdir(train_path)

        image_dir = file_dir
        dst_dir = train_path #./cut_image/coco_cut2/train
        labelme2coco.copyImage(image_dir,dst_dir)
        labelme2coco.labelme2coco(labelme_json, os.path.join(dst_coco_annota_path, gen_json_name))

    #sahi切图
    def sahi_cut_image(self, cut_model,
                       image_path="./cut_image/coco_cut2/train",
                       annotation_file_path="./cut_image/coco_cut2/annotations/train_cut1.json",
                       chop_size=(640, 640)):
        # dynamic cut image result
        # print('start sahi cut image.....')
        cut_image2.main(cut_model, image_path, annotation_file_path, chop_size=chop_size)

    #合并动态coco于sahi的coco
    def merge_coco_data(self,dynimic_coco_dir = "./cut_image/coco_cut2/",
                        sahi_coco_dir = "./cut_image/coco_cut3/",
                        merge_coco_dir = "./cut_image/coco_cut4"):
        # print('start merge coco data....')

        merge_coco.main(dynimic_coco_dir,sahi_coco_dir,merge_coco_dir)

    #数据格式转换
    def coco2yolo(self,coco_img_dir = "./cut_image/coco_cut4/train",
                    coco_ann_file = "./cut_image/coco_cut4/annotations",
                    output_dir = "./cut_image/txt_cut4"):

        # print('start coco2yolov5....')
        label_id_map = coco2yolov5.convert_coco_json(coco_ann_file, use_segments=True, cls91to80=False, output_dir=output_dir,
                                      coco_img_dir=coco_img_dir)
        return label_id_map
        # coco2yolov5.coco2yolo(coco_img_dir, coco_ann_file, output_dir)  # 这个版本不用

    # txt2labelme
    def txt2yolo(self,
                 txt_dir="",
                 img_dir="",
                 json_dir="",
                 label_id_map=dict()):
        cut_image.clean_dir(json_dir)
        convertor = yoloseg2labelme.YOLO2Labelme(txt_dir, img_dir, json_dir, label_id_map)
        convertor.convert()

    #参数配置文件
    def parse_args(self,sorce_dir ='./mydata/data',
                   dynamic_dir = './mydata/dst_data',
                   dst_dir = './Common/mydata/cut_image/txt_cut4'):
        #手动获取当前运行文件的文件名，赋值给prog,如果为空，内部会有问题
        current_path = os.path.abspath(__file__)
        current_path= os.path.basename(current_path)

        parser = argparse.ArgumentParser(prog =current_path)
        parser.add_argument("--sorce_dir",
                            type=str,
                            default= sorce_dir,
                            help="sorce dir, include json and jpg file")

        parser.add_argument("--dynamic_dir",
                            type=str,
                            default=dynamic_dir,
                            help="dynamic cut image dir,include json and jpg file")

        parser.add_argument("--dst_coco_dir",
                            type=str,
                            default=dst_dir,
                            help="out dir,include coco txt and jpg file")

        args = parser.parse_args()
        return args


#裁图功能-对外调用接口
def cutImageSdk(sorce_dir, dst_dir):
    # instance object
    CutImageSDK = CutImage(sorce_dir,dst_dir)
    if not osp.exists("workspace"):
        os.mkdir("workspace")
    dynamic_dir = os.path.join(os.path.dirname(CutImageSDK.sorce_dir), 'workspace/dst_data')
    if not osp.exists(dynamic_dir):
        os.mkdir(dynamic_dir)
    if not osp.exists(dst_dir):
        os.mkdir(dst_dir)
    # get params
    args = CutImageSDK.parse_args(sorce_dir, dynamic_dir, dst_dir)

    # #1.动态切图处理数据  暂时不用
    # CutImageSDK.dynamic_cut_image(args.sorce_dir,args.dynamic_dir)

    # 0. 统计数据集，判断切割的大小
    print(f"\033[1;32m step0: 统计判断切割大小... \033[0m")
    chop_size = CutImageSDK.get_chop_size(args.sorce_dir)
    print(f"\033[1;31m 切割的大小: {chop_size} \033[0m")

    # 1. chop_cut_image
    print(f"\033[1;32m step1: chop_cut_image... \033[0m")
    CutImageSDK.chop_cut_image(args.sorce_dir, args.dynamic_dir, chop_size=chop_size, shave=0)
    base_dir = osp.dirname(args.sorce_dir)#获取基础文件路径
    time.sleep(0.01)

    #2.将1 的处理结果转为coco格式
    print(f"\033[1;33m waiting for labelme2coco...(将chop_cut_image结果[labelme格式]转为coco格式, 方便合并结果) \033[0m")
    dst_coco_annota_path = os.path.join(base_dir, 'workspace/coco_cut2/annotations')
    CutImageSDK.labelme2coco_fun(args.dynamic_dir,dst_coco_annota_path, gen_json_name="train_cut1.json")
    time.sleep(0.01)

    #3.sahi切图
    # sahi切图从原图开始切(先原始数据转化成coco格式，然后在开始切)
    print(f"\033[1;33m waiting for labelme2coco...(将原始数据[labelme格式]转为coco格式，方便sahi切割) \033[0m")
    sahi_coco_annota_path = os.path.join(base_dir, "workspace/sahi_cut_temp/annotations")
    CutImageSDK.labelme2coco_fun(sorce_dir, sahi_coco_annota_path, gen_json_name="sahi_cut1.json")
    up_path = osp.dirname(sahi_coco_annota_path)
    cut_mode = ["train", "val"]
    image_path = os.path.join(up_path, "train")
    annotation_file_path = os.path.join(up_path, "annotations/sahi_cut1.json")
    print(f"\033[1;32m step2: sahi_cut_image \033[0m")
    CutImageSDK.sahi_cut_image(cut_mode[0], image_path, annotation_file_path, chop_size=chop_size)
    time.sleep(0.01)

    #4.合并动态coco于sahi的coco
    dynimic_coco_dir = os.path.join(base_dir, "workspace/coco_cut2")
    sahi_coco_dir = os.path.join(base_dir, "workspace/coco_cut3")
    merge_coco_dir = os.path.join(base_dir, "workspace/coco_cut4")
    print(f"\033[1;33m merge chop results[coco格式] and sahi results[coco格式] to coco【用于coco格式数据集训练, mask-rcnn等】... \033[0m")
    CutImageSDK.merge_coco_data(dynimic_coco_dir,sahi_coco_dir,merge_coco_dir)
    time.sleep(0.01)

    #5.coco2yolov5
    print(f"\033[1;33m convert coco results to yolo results【用于yolo格式数据集训练，yolov8-seg等】... \033[0m")
    coco_img_dir = os.path.join(merge_coco_dir, "train")
    coco_ann_file = os.path.join(merge_coco_dir, "annotations")
    # output_dir = args.dst_coco_dir  #base_dir + "/workspace/txt_cut4"
    output_dir = os.path.join(base_dir, "workspace/txt_cut4")
    label_id_map = CutImageSDK.coco2yolo(coco_img_dir, coco_ann_file, output_dir)
    time.sleep(0.01)

    #6. txt转化为labelme，方便可视化和后续训练操作
    print(f"\033[1;32m step3: txt转换为labelme格式【用于可视化&训练】... \033[0m")
    time.sleep(0.01)
    txt_dir = os.path.join(output_dir, r"labels\train2017")
    img_dir = coco_img_dir
    json_dir = dst_dir
    CutImageSDK.txt2yolo(txt_dir, img_dir, json_dir, label_id_map)

    #7.清理文件夹
    print(f"\033[1;32m step4: 移动&清理文件夹... \033[0m")
    copy_dir(img_dir, dst_dir)
    delete_dir(dynamic_dir)
    delete_dir(dynimic_coco_dir)  # 清理中间辅助文件夹
    delete_dir(sahi_coco_dir)
    delete_dir(up_path)

    print(f"\033[1;35m finash all... \033[0m")
    return 0


if __name__ == '__main__':
    src_dir = r"input-datasets"
    dst_dir = r"output-datasets"
    cutImageSdk(src_dir, dst_dir)
