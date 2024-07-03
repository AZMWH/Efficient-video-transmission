# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
import ffmpeg
import ffmpeg_cut
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# import matlab.engine

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from PIL import Image, ImageDraw


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    frame_room = []
    xyxy_video = []
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        if dnn:
            # check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        else:
            check_requirements(('onnx', 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            if dnn:
                net.setInput(img)
                pred = torch.tensor(net.forward())
            else:
                pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    frame_room.append(frame)

                # Write results
                xyxy_list = []
                dim_xyxy = reversed(det).shape[0]

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                        for xyxy_num in range(4):
                            xyxy_str = str(xyxy[xyxy_num])
                            xyxy_str = xyxy_str[7:len(xyxy_str) - 19]
                            xyxy_int = int(xyxy_str)
                            xyxy_list.append(xyxy_int)

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    v1, d1, v2, d2 = xyxy_list
                    if not xyxy_video:  # 如果xyxy_video数组为空，直接添加xyxy_list的值
                        xyxy_video = [v1, d1, v2, d2]
                    else:
                        # 更新xyxy_video数组的值
                        xyxy_video[0] = min(xyxy_video[0], v1)
                        xyxy_video[1] = min(xyxy_video[1], d1)
                        xyxy_video[2] = max(xyxy_video[2], v2)
                        xyxy_video[3] = max(xyxy_video[3], d2)
                    print(xyxy_video)
                # if xyxy_list:
                #     target_coor = np.array(xyxy_list)
                #     target_coor = target_coor.reshape(dim_xyxy, 4)  # 存储所有目标的xyxy
                #     print(target_coor)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    # 数据准备
    target_coor = np.array(xyxy_list)
    target_coor = target_coor.reshape(dim_xyxy, 4)  # 存储所有目标的xyxy
    print(frame_room)
    # ffmpeg_cut.cut_video(xyxy_video)
    ffmpeg.extract_frames(frame_room, xyxy_video)

    # print(target_coor)

    # coincide_list = []
    # coincide_list_2 = []
    # noncoincidence_list = []
    # x5, y5, x6, y6 = 0, 0, 0, 0
    # for target in range(dim_xyxy):  # 将重复的和不重复的分别列出
    #     contrast = [target_coor[target, 0], target_coor[target, 1], target_coor[target, 2], target_coor[target, 3]]
    #     wait_contrast = np.delete(target_coor, target, axis=0)
    #     for aim in range(dim_xyxy - 1):  # 每行矩阵都与其他行矩阵比较
    #         x1, y1, x2, y2 = contrast
    #         x3, y3, x4, y4 = wait_contrast[aim, :]
    #         if not (x3 >= x2 or x4 <= x1 or y3 >= y2 or y4 <= y1):  # 如果重合
    #             if coincide_list.count([x3, y3, x4, y4]) != 0:
    #                 index_1 = coincide_list.index([x3, y3, x4, y4])
    #             if coincide_list_2.count([x1, y1, x2, y2]) != 0:
    #                 index_2 = coincide_list_2.index([x1, y1, x2, y2])
    #             if coincide_list.count([x3, y3, x4, y4]) == 0 or coincide_list_2.count(
    #                     [x1, y1, x2, y2]) == 0 or index_1 != index_2:
    #                 coincide_list.append([x1, y1, x2, y2])
    #                 coincide_list_2.append([x3, y3, x4, y4])
    #         else:
    #             if x3 >= x2 or x4 <= x1 or y3 >= y2 or y4 <= y1:  # 如果没重合
    #                 if not (x5 == x1 and y5 == y1 and x6 == x2 and y6 == y2):
    #                     x5, y5, x6, y6 = x1, y1, x2, y2
    #                     noncoincidence_list.append([x5, y5, x6, y6])
    #
    # noncoincidence_list = [item for item in noncoincidence_list if item not in coincide_list]
    # noncoincidence_list = [item for item in noncoincidence_list if item not in coincide_list_2]
    # # 开切
    # print(noncoincidence_list)
    # if not noncoincidence_list:
    #     noncoincidence_list = target_coor
    # noncoincidence_list = np.array(noncoincidence_list)
    # coincide_list = np.array(coincide_list)
    # coincide_list_2 = np.array(coincide_list_2)
    #
    # CH_coor = []
    # for spot in range(len(coincide_list)):  # 找到两个矩形的重合部分
    #     upper_left_x = max(coincide_list[spot, 0], coincide_list_2[spot, 0])
    #     upper_left_y = max(coincide_list[spot, 1], coincide_list_2[spot, 1])
    #     lower_right_x = min(coincide_list[spot, 2], coincide_list_2[spot, 2])
    #     lower_right_y = min(coincide_list[spot, 3], coincide_list_2[spot, 3])
    #     CH_coor.append([upper_left_x, upper_left_y, lower_right_x, lower_right_y])
    # CH_coor = np.array(CH_coor)
    #
    # repeat_rectangle = []
    # xy_array = np.concatenate((coincide_list, coincide_list_2), axis=0)
    # position_1 = []
    # position_2 = []
    # position_3 = []
    # position_4 = []
    # position_5 = []
    # for number in range(len(coincide_list)):  # 将重叠的两个矩形切成五份
    #     x11, y11, x21, y21 = coincide_list[number]
    #     x31, y31, x41, y41 = coincide_list_2[number]
    #     x51, y51, x61, y61 = CH_coor[number]
    #     if repeat_rectangle.count([x11, y11, x21, y21]) == 0 and repeat_rectangle.count([x31, y31, x41, y41]) == 0:
    #         x1 = min(x11, x21, x31, x41)  # 1
    #         indices = np.where(xy_array == x1)
    #         y1 = int(xy_array[indices[0], indices[1] + 1])
    #         x2 = x51
    #         y2 = int(xy_array[indices[0], indices[1] + 3])
    #         x3 = x51  # 2
    #         y3 = min(y11, y21, y31, y41)
    #         x4 = x61
    #         y4 = y51
    #         x5 = x51  # 3
    #         y5 = y51
    #         x6 = x61
    #         y6 = y61
    #         x7 = x51  # 4
    #         y7 = y61
    #         x8 = x61
    #         y8 = max(y11, y21, y31, y41)
    #         x9 = x61  # 5
    #         x10 = max(x11, x21, x31, x41)
    #         indices2 = np.where(xy_array == x10)
    #         y10 = int(xy_array[indices2[0], indices[1] - 1])
    #         y9 = int(xy_array[indices2[0], indices[1] + 1])
    #
    #         repeat_rectangle.append([x11, y11, x21, y21])
    #         repeat_rectangle.append([x31, y31, x41, y41])
    #         position_1.append([x1, y1, x2, y2])
    #         position_2.append([x3, y3, x4, y4])
    #         position_3.append([x5, y5, x6, y6])
    #         position_4.append([x7, y7, x8, y8])
    #         position_5.append([x9, y9, x10, y10])
    #     # else:
    #
    # position_1 = np.array(position_1)
    # position_2 = np.array(position_2)
    # position_3 = np.array(position_3)
    # position_4 = np.array(position_4)
    # position_5 = np.array(position_5)
    # image = cv2.imread('data/video_target/frame_0026.jpg')
    # regions = []
    # if len(coincide_list) != 0:
    #     background = np.concatenate((noncoincidence_list, position_1, position_2, position_3, position_4, position_5),
    #                                 axis=0)  # 所有感兴趣目标的集合
    # else:
    #     background = noncoincidence_list
    # # print(background)
    #
    # img_kk = Image.open('data/video_target/frame_0026.jpg')
    #
    # # 创建ImageDraw对象
    # draw = ImageDraw.Draw(img_kk)
    #
    # # 在不同的位置画不同颜色的框
    # # 注意：位置是从左上角开始，以像素为单位
    # kk = 0
    # for kk in range(len(background)):
    #     draw.rectangle([(background[kk, 0], background[kk, 1]), (background[kk, 2], background[kk, 3])], outline='red',
    #                    width=5)
    #
    # # 保存图片
    # img_kk.save(r'C:\Users\He.Wang\Desktop\kk\1.jpg')
    #
    # for non_c in range(len(background)):
    #     regions.append((background[non_c, 0], background[non_c, 1], background[non_c, 2],
    #                     background[non_c, 3]))
    # mask = np.ones_like(image[:, :, 0], dtype=np.uint8)
    # for x, y, w, h in regions:
    #     cv2.rectangle(mask, (x, y), (w, h), (0, 0, 0), -1, cv2.LINE_AA)
    # remaining_image = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imwrite(r'C:\Users\He.Wang\Desktop\back\remaining_image.jpg', remaining_image)
    # print(background)
    #
    # for tar in range(len(background)):
    #     crop = image[background[tar, 1]:background[tar, 3], background[tar, 0]:background[tar, 2]]
    #     filepath = r'C:\Users\He.Wang\Desktop\tar\{},{},{},{}.jpg'.format(background[tar, 0], background[tar, 1],
    #                                                               background[tar, 2],
    #                                                               background[tar, 3])
    #     cv2.imwrite(filepath, crop)

    # 切目标
    # rect_width = 100
    # rect_height = 100
    # for tar in range(len(background)):
    #     x1 = background[tar, 0]
    #     y1 = background[tar, 1]
    #     x2 = background[tar, 2]
    #     y2 = background[tar, 3]
    #
    #     x_center = (x1 + x2) // 2
    #     y_center = (y1 + y2) // 2
    #
    #     x1 = x_center - rect_width // 2
    #     y1 = y_center - rect_height // 2
    #     x2 = x1 + rect_width
    #     y2 = y1 + rect_height
    #     if x1 < 0:
    #         a = 0 - x1
    #         x1 = 0
    #         x2 = x2 + a
    #
    #     crop = image[y1:y2, x1:x2]
    #     filepath = r'C:\Users\He.Wang\Desktop\target\{},{},{},{}.jpg'.format(x1, y1, x2, y2)
    #     cv2.imwrite(filepath, crop)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'runs/train/S+GOLD+BOTNET+CA200/weights/best.pt',
                        help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'video/text/text2.avi',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    # parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


# 调用matlab
# def matlab_for_OFDM():
#     eng = matlab.engine.start_matlab()
#     eng.cd('G:/matlabr2021b/bin/OFDM', nargout=0)
#     ratio = eng.new_bellhop_for_OFDM_fengzhuang()
#     print(ratio)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    # matlab_for_OFDM()
