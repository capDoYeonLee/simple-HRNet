import os
import sys
import argparse
import ast
import cv2
import time
import torch
from vidgear.gears import CamGear
import numpy as np

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations

def main(camera_id, filename, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution,
         single_person, use_tiny_yolo, disable_tracking, max_batch_size, disable_vidgear, save_video, video_format,
         video_framerate, device):
    global boxes
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')



    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    video_writer = None

    if filename is not None:
        rotation_code = check_video_rotation(filename)
        video = cv2.VideoCapture(filename)
        assert video.isOpened()
    else:
        rotation_code = None
        if disable_vidgear:
            video = cv2.VideoCapture(camera_id)
            assert video.isOpened()
        else:
            video = CamGear(camera_id).start()

    if use_tiny_yolo:
         yolo_model_def="./models/detectors/yolo/config/yolov3-tiny.cfg"
         yolo_class_path="./models/detectors/yolo/data/coco.names"
         yolo_weights_path="./models/detectors/yolo/weights/yolov3-tiny.weights"
    else:
         yolo_model_def="./models/detectors/yolo/config/yolov3.cfg"
         yolo_class_path="./models/detectors/yolo/data/coco.names"
         yolo_weights_path="./models/detectors/yolo/weights/yolov3.weights"

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        return_bounding_boxes=not disable_tracking,
        max_batch_size=max_batch_size,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device
    )

    if not disable_tracking:
        # here
        prev_boxes = None
        prev_pts = None
        prev_person_ids = None
        next_person_id = 0

    while True:
        t = time.time()

        if filename is not None or disable_vidgear:
            ret, frame = video.read()
            if not ret:
                break
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
        else:
            # here
            frame = video.read()
            if frame is None:
                break

        pts = model.predict(frame)   # model = Simple HRNet


        if not disable_tracking:
            # here
            boxes, pts = pts
            # boxes is numpy array (1,4)
            # pts   is numpy array (1,17,3) joint
            print(f" first pts shape is {pts.shape}")


        if not disable_tracking:
            # here
            if len(pts) > 0:
                if prev_pts is None and prev_person_ids is None:
                    # first pts mayme is '0'. > what is the pts.
                    person_ids = np.arange(next_person_id, len(pts) + next_person_id, dtype=np.int32)
                    next_person_id = len(pts) + 1
                else:
                    # and then here
                    boxes, pts, person_ids = find_person_id_associations(
                        boxes=boxes, pts=pts, prev_boxes=prev_boxes, prev_pts=prev_pts, prev_person_ids=prev_person_ids,
                        next_person_id=next_person_id, pose_alpha=0.2, similarity_threshold=0.4, smoothing_alpha=0.1,
                    )
                    next_person_id = max(next_person_id, np.max(person_ids) + 1)
            else:
                person_ids = np.array((), dtype=np.int32)

            prev_boxes = boxes.copy()
            prev_pts = pts.copy()
            prev_person_ids = person_ids

        else:

            person_ids = np.arange(len(pts), dtype=np.int32)

        # pts is
        landmarks = ["class"]

        for i, (pt, pid) in enumerate(zip(pts, person_ids)):
            frame, points = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=pid,
                                             points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                             points_palette_samples=10)
        #for num in range(pts.shape[1]):  # 일단 여기서 좌표값 저장하는데, 결과 한번 봐야 될 듯.   i > pts.shape[1]
        #    landmarks += ['x{}'.format(num), 'y{}'.format(num), 'z{}'.format(num)]

        import csv

        class_name = "Squat"
        pts = pts.tolist()


        # [test for elem in pts]
        # [
        test = []
        for elem in pts:
            test += elem

        list_to_excel_points = []
        for num in test:
            list_to_excel_points += num


        #list_to_excel_points.insert(0, class_name)

        # if not os.path.exists('test.csv'):
        #     with open('test.csv', mode='w', newline='') as f:
        #         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #         csv_writer.writerow(landmarks)
        #
        # else:
        #     with open('test.csv', mode='a', newline='') as f:
        #         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #         csv_writer.writerow(list_to_excel_points)




            # add code TODO
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df = pd.read_csv('test.csv')
        xis = df.drop('class', axis=1)  # features
        yis = df['class']  # target value

        x_train, x_test, y_train, y_test = train_test_split(xis, yis, test_size=0.3, random_state=1234)
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)


        import pickle
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        pipelines = {
            'lr': make_pipeline(StandardScaler(), LogisticRegression()),  # 분류. 일정 기준 넘기면 분류됨.
            'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        }


        # fit_models = {}
        # for algo, pipeline in pipelines.items():
        #     model = pipeline.fit(x_train, y_train)
        #     fit_models[algo] = model

        # with open('test.pkl', 'wb') as f:
        #     pickle.dump(fit_models['gb'], f)

        with open('test.pkl', 'rb') as f:
            body_model = pickle.load(f)

        aa = pd.DataFrame([list_to_excel_points])  # row = pose_row + face_row : x, y, z
        body_language_class = body_model.predict(aa)[0]
        body_language_prob = body_model.predict_proba(aa)[0]
        print(body_language_class, body_language_prob)



        fps = 1. / (time.time() - t)
        print('\rframerate: %f fps' % fps, end='')

        if has_display:
            cv2.imshow('frame.png', frame)
            k = cv2.waitKey(1)
            if k == 27:  # Esc button
                if disable_vidgear:
                    video.release()
                else:
                    video.stop()
                break
        else:
            cv2.imwrite('frame.png', frame)

        if save_video:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
                video_writer = cv2.VideoWriter('output.avi', fourcc, video_framerate, (frame.shape[1], frame.shape[0]))
            video_writer.write(frame)

    if save_video:
        video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", "-d", help="open the camera with the specified id", type=int, default=0)
    parser.add_argument("--filename", "-f", help="open the specified video (overrides the --camera_id option)",
                        type=str, default=None)
    parser.add_argument("--hrnet_m", "-m", help="network model - 'HRNet' or 'PoseResNet'", type=str, default='HRNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--use_tiny_yolo",
                        help="Use YOLOv3-tiny in place of YOLOv3 (faster person detection). Ignored if --single_person",
                        action="store_true")
    parser.add_argument("--disable_tracking",
                        help="disable the skeleton tracking and temporal smoothing functionality",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--disable_vidgear",
                        help="disable vidgear (which is used for slightly better realtime performance)",
                        action="store_true")  # see https://pypi.org/project/vidgear/
    parser.add_argument("--save_video", help="save output frames into a video.", action="store_true")
    parser.add_argument("--video_format", help="fourcc video format. Common formats: `MJPG`, `XVID`, `X264`."
                                                     "See http://www.fourcc.org/codecs.php", type=str, default='MJPG')
    parser.add_argument("--video_framerate", help="video framerate", type=float, default=30)
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                                         "Set to `cuda` to use all available GPUs (default); "
                                         "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
