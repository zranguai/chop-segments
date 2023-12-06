import json
import cv2
import os
import base64
import shutil


# 清空path目录下的所有目录和文件
def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


def img_file_list(path):
    img_names = []
    for img_name in os.listdir(path):
        if img_name[-4:] == ".jpg":
            img_names.append(img_name)
    return img_names


# 让cut_for_box窗口在原图上滑动，只要目标的所有点都在cut_for_box窗口内那么就选择该窗口
def slide_window(cut_for_box, img_width, img_height, point_minx, point_miny, point_maxx, point_maxy, step_eachx, step_eachy):
    step_each_x = step_eachx
    step_each_y = step_eachy
    for step_y in range(0, img_width, step_each_y):
        for step_x in range(0, img_height, step_each_x):  # 屏幕向右为x轴
            my_cut_box = [[step_x, step_y], [cut_for_box + step_x, cut_for_box + step_y]]
            # 如果my_cut_box包含所有点，那么就是需要的my_cut_box
            if (point_minx > step_x and point_miny > step_y and point_maxx < (
                    cut_for_box + step_x) and point_maxy < (cut_for_box + step_y)):
                return my_cut_box
    # TODO: 这里返回值需做异常处理
    my_cut_box = [[int(point_minx - 10), int(point_miny - 10)], [int(point_minx + cut_for_box - 10), int(point_miny + cut_for_box - 10)]]
    return my_cut_box


# 在原图上进行滑动操纵(选择包含目标最多的作为切割框)，并把要切割的(left_top, right_bottom)坐标写进shapes
def slide_window_complex(cut_box, img_width, img_height, shapes, step_eachx, step_eachy):
    step_each_x = step_eachx
    step_each_y = step_eachy

    for shap_index, shape in enumerate(shapes):
        # 根据最小外接矩形确定窗口的大小
        # debug here cut_for_box可能进不去该if条件
        cut_for_box = cut_box[-1][1]  # 选择为当前cut_box里面最大的框
        for box in cut_box:
            if max(shape["min_box"]) > box[0] and max(shape["min_box"]) < box[1]:
                cut_for_box = box[1]  # 确定切的框的宽高

        for step_y in range(0, img_width, step_each_y):
            for step_x in range(0, img_height, step_each_x):
                my_cut_box = [[step_x, step_y], [cut_for_box + step_x, cut_for_box + step_y]]  # 滑动窗口的框
                # 滑动窗口的框和目标框之间的处理逻辑
                # 找出目标的中心点落在滑动框里面的滑动窗口
                point_minx, point_miny = shape["ltrb"][0]
                point_maxx, point_maxy = shape["ltrb"][1]
                if (point_minx > step_x and point_miny > step_y and point_maxx < (
                    cut_for_box + step_x) and point_maxy < (cut_for_box + step_y)):
                    shape["my_cut_box"] = my_cut_box  # 滑动窗口赋值到shape上去
        # print(shape)
        # todo: debug here 如果shape没有"my_cut_box"这个属性，进行添加
        if "my_cut_box" not in shape:
            my_cut_box = [[int(shape["ltrb"][0][0]), int(shape["ltrb"][0][1])], [int(shape["ltrb"][0][0]+cut_for_box), int(shape["ltrb"][0][1]+cut_for_box)]]
            shape["my_cut_box"] = my_cut_box
    return shapes


def compute_iou(i_box, j_box):
    x1min, y1min = i_box[0]
    x1max, y1max = i_box[1]
    x2min, y2min = j_box[0]
    x2max, y2max = j_box[1]

    # 计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交部分的坐标
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    # 计算iou
    iou = intersection / union
    return iou


def merge_or_cut(complex_shapes, iou_conf):
    long_complex_shapes = []  # 太长太宽
    normal_complex_shape = []  # 正常尺度
    # print(complex_shapes)
    len_complex_shapes = len(complex_shapes)
    max_w_h = 1500
    for cs_index in range(len_complex_shapes):
        # 太长或者太宽的先剔除
        if max(complex_shapes[cs_index]["min_box"]) > max_w_h:
            long_complex_shapes.append(complex_shapes[cs_index])
        else:
            normal_complex_shape.append(complex_shapes[cs_index])
    # 每个目标以中心点开始 扩大自己的滑动框的大小观察是否包含其他目标
    len_normal_complex_shape = len(normal_complex_shape)
    # TODO: 写太长太宽的逻辑(sahi包括了)
    # TODO: 写len_normal_complex小于2的异常处理
    # 将重合的放在一起  长度为3，(0, 1) (0, 2) (1, 2)
    for i in range(len_normal_complex_shape):
        for j in range(i+1, len_normal_complex_shape):
            # 取出第i个信息
            i_shape = normal_complex_shape[i]
            # 取出第j个信息
            j_shape = normal_complex_shape[j]
            # print(i_shape)
            # print(j_shape)
            # 判断j的最小外接矩形是否在i的滑动框里面，在的话将j的滑动框my_cut_box设置和i一样
            # 1.计算iou, iou>0说明这两个可以重合
            # todo:这一步会报错 KeyError: 'my_cut_box'
            ij_iou = compute_iou(i_shape["my_cut_box"], j_shape["my_cut_box"])
            if ij_iou <= iou_conf:  # 不重合/重合度较小，跳过
                continue
            else:  # 重合，更改my_cut_box信息(根据ltrb, my_cut_box等信息进行重新修改my_cut_box的信息)
                # 找出i, j的最大滑动窗口
                max_slide = max((i_shape["my_cut_box"][1][0] - i_shape["my_cut_box"][0][0]), (j_shape["my_cut_box"][1][0] - j_shape["my_cut_box"][0][0]))
                # 如果j的ltrb都在i滑动框里面，直接使用i的my_cut_box
                if (j_shape["ltrb"][0][0] > i_shape["my_cut_box"][0][0]) and (j_shape["ltrb"][0][1] > i_shape["my_cut_box"][0][1]) and (j_shape["ltrb"][1][0] < i_shape["my_cut_box"][1][0]) and (j_shape["ltrb"][1][1] < i_shape["my_cut_box"][1][1]):
                    j_shape["my_cut_box"] = i_shape["my_cut_box"]
                else:
                    # 1.找出这两个最大的ltrb位置
                    min_l = min(i_shape["ltrb"][0][0], j_shape["ltrb"][0][0])
                    min_t = min(i_shape["ltrb"][0][1], j_shape["ltrb"][0][1])
                    max_r = max(i_shape["ltrb"][1][0], j_shape["ltrb"][1][0])
                    max_b = max(i_shape["ltrb"][1][1], j_shape["ltrb"][1][1])
                    max_ltrb = max((max_r - min_l), (max_b - min_t))
                    # # 滑动窗口在640尺度
                    # if max_ltrb > 0 and max_ltrb <= 640:
                    #     my_cutcut_box = [[int(min_l) - 5, int(min_t) - 5], [int(min_l) + 645, int(min_t) + 645]]
                    # # 滑动窗口为960尺度
                    # elif max_ltrb > 640 and max_ltrb <= 960:
                    #     my_cutcut_box = [[int(min_l) - 5, int(min_t) - 5], [int(min_l) + 965, int(min_t) + 965]]
                    # # 滑动窗口为1024尺度
                    # elif max_ltrb > 960 and max_ltrb <= 1024:
                    #     my_cutcut_box = [[int(min_l) - 5, int(min_t) - 5], [int(min_l) + 1029, int(min_t) + 1029]]
                    # # 滑动窗口为2048尺度
                    # elif max_ltrb > 1024 and max_ltrb <= 2048:
                    #     my_cutcut_box = [[int(min_l) - 5, int(min_t) - 5], [int(min_l) + 2053, int(min_t) + 2053]]

                    # 滑动窗口在640尺度
                    if max_ltrb > 0 and max_ltrb <= 740:
                        my_cutcut_box = [[int(min_l) - 5, int(min_t) - 5], [int(min_l) + 745, int(min_t) + 745]]
                    # 滑动窗口为960尺度
                    elif max_ltrb > 740 and max_ltrb <= 1060:
                        my_cutcut_box = [[int(min_l) - 5, int(min_t) - 5], [int(min_l) + 1065, int(min_t) + 1065]]
                    # 滑动窗口为1024尺度
                    elif max_ltrb > 1060 and max_ltrb <= 1124:
                        my_cutcut_box = [[int(min_l) - 5, int(min_t) - 5], [int(min_l) + 1129, int(min_t) + 1129]]  #这里写错了
                    # 滑动窗口为2048尺度
                    elif max_ltrb > 1124 and max_ltrb <= 2048:
                        my_cutcut_box = [[int(min_l) - 5, int(min_t) - 5], [int(min_l) + 2053, int(min_t) + 2053]]


                    # 2.更改i_shape, j_shape里面的my_cutcut_box
                    i_shape["my_cut_box"] = my_cutcut_box
                    j_shape["my_cut_box"] = my_cutcut_box

    # print(long_complex_shapes)
    # print(normal_complex_shape)

    return long_complex_shapes, normal_complex_shape


# 根据new_j的标注文件进行切割图片并生成对应的json
def cut_image(dst_file_dir, img_file, img_mat, new_j):
    # 处理new_j文件, 根据my_cut_box的不同决定切割几个出来
    cut_j = new_j
    shapes = new_j["shapes"]
    cut_box_diff = 1  # 准备切的生成的份数
    my_cut_box = shapes[0]["my_cut_box"]
    shapes_id = 0
    # 查看有多少个不同的my_cut_box
    for shape_index, shape in enumerate(shapes):
        if shape["my_cut_box"] != my_cut_box:
            cut_box_diff += 1
            shapes_id += 1
            shape["shapes_id"] = shapes_id  # id相同的合并到同一个地方去
            my_cut_box = my_cut_box
        else:
            shape["shapes_id"] = shapes_id
    # print(new_j)
    # 根据生成的份数准备1.图片和json名字，2.json文件
    img_paths = []
    json_paths = []
    json_files = []
    for diff_index in range(cut_box_diff):
        img_na = img_file[:-4] + str(diff_index) + img_file[-4:]
        json_na = img_na.replace(".jpg", ".json")
        json_file = {}
        # 相同的部分放进json_file里面
        json_file['version'] = new_j['version']  # no change
        json_file['flags'] = new_j['flags']  # no change
        json_file['shapes'] = []  # list类型 change
        json_file['imagePath'] = img_na  # no change
        json_file['imageData'] = ""  # change
        json_file['imageHeight'] = 0  # change 后续根据宽高切割imageData
        json_file['imageWidth'] = 0  # change
        dst_img_full_path = os.path.join(dst_file_dir, img_na)
        dst_json_full_path = os.path.join(dst_file_dir, json_na)

        img_paths.append(dst_img_full_path)
        json_paths.append(dst_json_full_path)
        json_files.append(json_file)

    # 遍历shapes,将shapes_id相同的放进同一个shapes里面
    for_hw_shapeid = 0
    for shape_index, shape in enumerate(shapes):
        # 把shape里面的点根据切割框左上角也进行相对的改变
        box_minx, box_miny = shape["my_cut_box"][0]
        for point in shape["points"]:
            point[0] -= box_minx
            point[1] -= box_miny
        # todo:观察是否需要去掉'min_box','ltrb','my_cut_box'等信息
        json_files[shape["shapes_id"]]["shapes"].append(shape)

        # width_ = shape["my_cut_box"][1][0] - shape["my_cut_box"][0][0]
        # height_ = shape["my_cut_box"][1][1] - shape["my_cut_box"][0][1]
        #
        # json_files[shape["shapes_id"]]["imageHeight"] = height_
        # json_files[shape["shapes_id"]]["imageWidth"] = width_

    # 正式切割
    # print(json_files)
    # print(1)
    for diff_index in range(cut_box_diff):
        dst_img_full_path = img_paths[diff_index]
        dst_json_full_path = json_paths[diff_index]

        new_json = json_files[diff_index]
        slide_minx, slide_miny = new_json["shapes"][0]["my_cut_box"][0]
        slide_maxx, slide_maxy = new_json["shapes"][0]["my_cut_box"][1]
        img_new_mat = img_mat[slide_miny:slide_maxy, slide_minx:slide_maxx]

        # 图片的宽度和高度
        new_json["imageWidth"] = img_new_mat.shape[1]
        new_json["imageHeight"] = img_new_mat.shape[0]

        try:
            cv2.imwrite(dst_img_full_path, img_new_mat)  # 目标图片
        except:
            print(f"{img_file} is false")
            continue

        f = open(dst_img_full_path, 'rb')
        base64_encode = base64.b64encode(f.read()).decode('utf-8')
        new_json['imageData'] = base64_encode

        with open(dst_json_full_path, mode="w", encoding="utf-8") as out:
            out.write(json.dumps(new_json, ensure_ascii=False))  # 目标json
        print(f"{img_file} is done")


def main_process(src_file_dir, dst_file_dir, img_names, step_eachx, step_eachy, iou_conf, box_range):
    for img_index, img_file in enumerate(img_names):
        src_img_full_path = os.path.join(src_file_dir, img_file)
        src_json_full_path = os.path.join(src_file_dir, img_file.replace(".jpg", ".json"))
        dst_img_full_path = os.path.join(dst_file_dir, img_file)
        dst_json_full_path = os.path.join(dst_file_dir, img_file.replace(".jpg", ".json"))

        # 读取图片错误做异常处理
        try:
            img_mat = cv2.imread(src_img_full_path)  # 读取源图像，后续进行切割
        except:
            print(f"\033[1;31m {img_file}图片读取不到，进行异常处理\033[0m")
            continue
        # json文件缺失或者读取不到，做异常处理
        try:
            with open(src_json_full_path, mode="r", encoding="utf-8") as f_src:
                j = json.loads(f_src.read())

                new_j = {}
                new_j['version'] = j['version']  # no change
                new_j['flags'] = j['flags']  # no change
                new_j['shapes'] = j['shapes']  # change
                new_j['imagePath'] = j['imagePath']  # no change
                new_j['imageData'] = j['imageData']  # change
                new_j['imageHeight'] = j['imageHeight']  # change
                new_j['imageWidth'] = j['imageWidth']  # change

                # 得到图片的宽高
                img_width = new_j['imageWidth']
                img_height = new_j['imageHeight']

                shapes = new_j["shapes"]
                # 方案1: 该图没有标注，图片小于一定尺度。保留原图和原标注信息                min_w_h = max(img_mat.shape[0], img_mat.shape[1])
                less_640 = max(img_mat.shape[0], img_mat.shape[1])
                if len(shapes) == 0 or less_640 < 640:
                    cv2.imwrite(dst_img_full_path, img_mat)
                    with open(dst_json_full_path, mode="w", encoding="utf-8") as out:
                        out.write(json.dumps(new_j, ensure_ascii=False))
                    print(f"{src_json_full_path}为空标注, 或者大小小于640，将原图和json文件保存到了{dst_file_dir}文件夹")
                # 方案2: 该图只有一个标注，根据标注信息的宽高确定裁剪的尺度
                elif len(shapes) == 1:
                    cut_box = box_range  # 后面继续增加
                    # 找出最左上点 最右下点的坐标(该形状的最小内接矩形)
                    point_minx = 100000000000
                    point_miny = 100000000000
                    point_maxx = -1
                    point_maxy = -1
                    for point in shapes[0]["points"]:
                        # 最小最大x
                        if point[0] < point_minx:
                            point_minx = point[0]
                        if point[0] > point_maxx:
                            point_maxx = point[0]
                        # 最小最大y
                        if point[1] < point_miny:
                            point_miny = point[1]
                        if point[1] > point_maxy:
                            point_maxy = point[1]

                    min_box_width = point_maxx - point_minx
                    min_box_height = point_maxy - point_miny
                    # 中心点的位置
                    min_box_centerx = point_minx + min_box_width / 2
                    min_box_centery = point_miny + min_box_height / 2

                    # 选择滑动的框
                    for box in cut_box:
                        if max(min_box_width, min_box_height) > box[0] and  max(min_box_width, min_box_height) < box[1]:
                            cut_for_box = box[1]  # 确定切的框的宽高

                    # 让cut_for_box窗口在原图上滑动，只要目标的所有点都在cut_for_box窗口内那么就选择该窗口
                    slide_cut_box = slide_window(cut_for_box, img_width, img_height, point_minx, point_miny, point_maxx, point_maxy, step_eachx, step_eachy)
                    print(slide_cut_box)  # 滑动窗口结束
                    # 根据滑动窗口进行切割图片
                    print(f"{img_file} after slide_window")
                    slide_minx, slide_miny = slide_cut_box[0]
                    slide_maxx, slide_maxy = slide_cut_box[1]
                    # 改变point点的坐标(变成相对于640的坐标)
                    for point in shapes[0]["points"]:
                        point[0] -= slide_minx
                        point[1] -= slide_miny
                    # print(shapes[0]["points"])
                    img_new_mat = img_mat[slide_miny:slide_maxy, slide_minx:slide_maxx]
                    # new_j['imageWidth'] = slide_maxx - slide_minx
                    # new_j['imageHeight'] = slide_maxy - slide_miny
                    new_j["imageWidth"] = img_new_mat.shape[1]
                    new_j['imageHeight'] = img_new_mat.shape[0]
                    try:
                        cv2.imwrite(dst_img_full_path, img_new_mat)  # 切割的图片写入目标目录
                    except:
                        print(f"{img_file} is false")
                        continue
                    f = open(dst_img_full_path, 'rb')
                    base64_encode = base64.b64encode(f.read()).decode('utf-8')
                    new_j['imageData'] = base64_encode

                    with open(dst_json_full_path, mode="w", encoding="utf-8") as out:
                        out.write(json.dumps(new_j, ensure_ascii=False))
                    print(f"{img_file} is done")
                # 方案3: 较为复杂场景
                else:
                    # 得到所有类别的信息
                    cut_box = box_range  # 后面继续增加
                    # 遍历shapes,将最小外接矩形宽高信息, 左上角，右下角信息加入到shape里面
                    for sap_index, shape in enumerate(shapes):
                        # 找出最左上点 最右下点的坐标(该形状的最小内接矩形)
                        point_minx = 100000000000
                        point_miny = 100000000000
                        point_maxx = -1
                        point_maxy = -1
                        for point in shape["points"]:
                            # 最小最大x
                            if point[0] < point_minx:
                                point_minx = point[0]
                            if point[0] > point_maxx:
                                point_maxx = point[0]
                            # 最小最大y
                            if point[1] < point_miny:
                                point_miny = point[1]
                            if point[1] > point_maxy:
                                point_maxy = point[1]
                        min_box_width = point_maxx - point_minx
                        min_box_height = point_maxy - point_miny
                        min_box = [min_box_width, min_box_height]
                        shape["min_box"] = min_box
                        # shape["ltrb"] = [[point_minx, point_miny], [point_maxx, point_maxy]]
                        # 解决点再边框边缘
                        shape["ltrb"] = [[point_minx - 20, point_miny - 20], [point_maxx + 20, point_maxy + 20]]

                    # 复杂部分进行滑动窗口操作
                    complex_shapes = slide_window_complex(cut_box, img_width, img_height, shapes, step_eachx, step_eachy)
                    # 复杂部分进行处理，滑动框较小则合并,滑动框较大则切割(处理滑动窗口)
                    long_complex_shapes, normal_complex_shape = merge_or_cut(complex_shapes, iou_conf)
                    # 根据my_cut_box对图像，json文件进行切割处理
                    cut_image(dst_file_dir, img_file, img_mat, new_j)  # 因为是浅拷贝，所以new_j里面的值会自动改变
                    # 观察new_j和complex_shapes变没变
                    # print(new_j)
                    # print(complex_shapes)
                    # print(1)
        except:
            print(f"\033[1;34m {img_file}对应的json文件找不到，进行异常处理\033[0m")
            continue


if __name__ == '__main__':
    src_file_dir = "./test"  # 元素jpg和json的目录
    dst_file_dir = "./dst_data"  # 生成小图的jpg和json的目录
    # 运行前清空目录
    clean_dir(dst_file_dir)
    # 遍历文件夹得到文件名
    img_names = img_file_list(src_file_dir)
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
    main_process(src_file_dir, dst_file_dir, img_names, step_eachx, step_eachy, iou_conf, box_range)
    print("ALL DONE")
