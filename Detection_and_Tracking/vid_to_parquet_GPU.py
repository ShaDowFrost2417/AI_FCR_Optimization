import os
import numpy as np
import cupy as cp
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import supervision as sv
import datetime
from strong_sort.utils.parser import YamlParser
from strong_sort.strong_sort import StrongSORT
import pandas as pd

os.environ['CUDA_PATH'] = 'D:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1'
os.environ['CUDA_HOME'] = 'D:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1'

# for key, value in os.environ.items():
#     print(f"{key}: {value}")

def calculate_direction(initial_x, initial_y, final_x, final_y):
    # Calculate the displacement vector
    displacement_x = final_x - initial_x
    displacement_y = final_y - initial_y

    # Calculate the angle in radians using arctan2
    angle_rad = cp.arctan2(displacement_y, displacement_x)

    # Convert angle to degrees
    angle_deg = cp.degrees(angle_rad)

    # Ensure the angle is positive
    angle_deg = (angle_deg + 360) % 360

    return angle_deg


# Detection Settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.65

# Tracker Settings
reid_weights = os.path.join('.', 'strong_sort', 'weights', 'osnet_x0_25_msmt17.pt')
tracker_config = "strong_sort/configs/strong_sort.yaml"
cfg = YamlParser()
cfg.merge_from_file(tracker_config)
track_outputs = [None]

video_file_path = 'D:/Study/NTU/FYP/Dataset/Once_a_day/Use'

for video_file in os.listdir(video_file_path):
    print(video_file)

    # Global variable Initialization (as cuPy array to allow the GPU to be the one storing there runtime variables)
    Date = []
    Time = []
    fish_count = cp.empty((0,))
    COG = cp.empty((0,2))
    # var_x = cp.empty((0,))
    # var_y = cp.empty((0,))
    var_x = []
    var_y = []
    # individual_fish_velocity = cp.empty((0,2))
    individual_fish_velocity = []
    average_speed = cp.empty((0,))
    lingering = cp.empty((0,))

    # individual_fish_direction = cp.empty((0,2))
    individual_fish_direction = []
    average_direction = cp.empty((0,))

    previous_coords = {}

    # Get the time from video file name
    video_name = video_file.split('.mp4')[0]
    current_time = ' '.join(video_name.split('_'))
    current_time = current_time[:-3] + ':' + current_time[-2:]
    current_time = pd.to_datetime(current_time)

    # Setting Up Tracker
    tracker = StrongSORT(
        reid_weights,
        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        False,
        max_dist=cfg['STRONGSORT']['MAX_DIST'],
        max_iou_distance=cfg['STRONGSORT']['MAX_IOU_DISTANCE'],
        max_age=cfg['STRONGSORT']['MAX_AGE'],
        n_init=cfg['STRONGSORT']['N_INIT'],
        nn_budget=cfg['STRONGSORT']['NN_BUDGET'],
        mc_lambda=cfg['STRONGSORT']['MC_LAMBDA'],
        ema_alpha=cfg['STRONGSORT']['EMA_ALPHA']
    )

    if hasattr(tracker, 'model'):
        if hasattr(tracker.model, 'warmup'):
            tracker.model.warmup()

    # Video
    video_path = os.path.join(video_file_path, video_file)
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    fps = 20

    # Detection
    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    MODEL_PATH = os.path.join('.', 'custom-model-zebra1-100')
    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    model.to(DEVICE)

    # test_counter = 0
    while ret:
        # if test_counter >= 20:
        #     break
        # test_counter += 1
        start = datetime.datetime.now()
        try:
            with torch.no_grad():
                # load image and predict
                inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
                outputs = model(**inputs)

                # post-process
                target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
                results = image_processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=CONFIDENCE_THRESHOLD,
                    target_sizes=target_sizes,
                )[0]

            # annotate with non-max suppression (nms)
            try:
                detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_THRESHOLD)
            except:
                print("no_nms")
                detections = sv.Detections.from_transformers(transformers_results=results)

            # Variable initialization
            count = cp.array([0])
            previously_tracked_count = cp.array([0])
            total_velocity_in_frame = cp.array([0.0])
            total_direction_in_frame = cp.array([0.0])
            x_x_area_total = cp.array([0.0])  # x centroid multiplied by area of bounding box
            y_x_area_total = cp.array([0.0])  # y centroid multiplied by area of bounding box
            area_total = cp.array([0.0])
            linger_count = cp.array([0])
            positions = cp.empty((0,2))
            velocity_in_frame = cp.empty((0,2))
            direction_in_frame = cp.empty((0,2))

            try:
                # Tracking
                track_outputs[0] = tracker.update(detections.xyxy, detections.confidence, detections.class_id, image)

                # Iterate for every fish tracked inside the frame
                for index, track_output in enumerate(track_outputs[0]):
                    x1, y1, x2, y2, track_id = cp.asarray(track_output[0:5])  # bbox & track_id
                    track_id_CPU = track_output[4]

                    # Getting centroid of the bounding box
                    c_x, c_y = cp.round((x1 + x2) / 2), cp.round((y1 + y2) / 2)

                    ################ For STDev later ################
                    positions = cp.concatenate((positions, cp.array([[c_x, c_y]])))

                    ################ COG ################
                    area = cp.abs((x2 - x1) * (y2 - y1))
                    area_total[0] += area
                    x_x_area_total[0] += area * c_x
                    y_x_area_total[0] += area * c_y

                    # if track_id has already been recorded previously
                    # for properties that require prior tracking data (at least 2 bounding box data of the same track ID), i.e. velocity, lingering, direction

                    # Get indexes containing the same track ID
                    # track_id_history_index = cp.where(previous_coords[:, 0] == track_id)[0]

                    # Check if the same track ID is already recorded before in previous_coords
                    if track_id_CPU in previous_coords:
                        previously_tracked_count[0] += 1
                        prev_c_x, prev_c_y, prev_Time, prev_Dist = previous_coords[track_id_CPU]

                        prev_c_x = cp.asarray(prev_c_x)
                        prev_c_y = cp.asarray(prev_c_y)
                        # prev_Time = cp.asarray(prev_Time)
                        # prev_Dist = cp.asarray(prev_Dist)

                        ################ Velocity (50 frames Rolling) ################
                        dist = cp.sqrt((c_x - prev_c_x) ** 2 + (c_y - prev_c_y) ** 2)
                        latestDist = prev_Dist[-49:]
                        latestTime = prev_Time[-49:]
                        rollingDist = sum(latestDist) + dist
                        velocity = rollingDist / (sum(latestTime) + (1 / fps))

                        # update values
                        current_velocity_in_frame = cp.array([[float(track_id_CPU), float(velocity)]])
                        velocity_in_frame = cp.concatenate((velocity_in_frame, current_velocity_in_frame))
                        total_velocity_in_frame[0] += cp.asarray(velocity)
                        # prev_Time = cp.concatenate((prev_Time, cp.array([1 / fps])))
                        # prev_Dist = cp.concatenate((prev_Dist, cp.array([dist])))
                        prev_Time.append(1 / fps)
                        prev_Dist.append(dist)

                        ################ Swimming Direction (similar to Turning angle) ################
                        direction = calculate_direction(prev_c_x, prev_c_y, c_x, c_y)
                        direction_in_frame = cp.concatenate((direction_in_frame, cp.array([[track_id, direction]])))
                        total_direction_in_frame[0] += direction

                        ################ Linger ################
                        if dist <= 2:
                            linger_count[0] += 1

                        # previous_coords[track_id_history_index[0], :] = [track_id, c_x, c_y, prev_Time, prev_Dist]

                    else:  # Otherwise initialize the velocity
                        prev_Dist = [0]
                        prev_Time = [0]
                        # previous_coords = cp.concatenate((previous_coords, cp.array([[track_id, c_x, c_y, prev_Time, prev_Dist]])))

                    # input into box labels
                    c_x_CPU = int(c_x.get())
                    c_y_CPU = int(c_y.get())
                    previous_coords[track_id_CPU] = (c_x_CPU, c_y_CPU, prev_Time, prev_Dist)

                    # fish count inside the frame
                    count[0] += 1

            except:
                print("error")
                pass
        except:
            print("error")
            pass

        ##Append data for final dataframe
        # fish count
        fish_count = cp.concatenate((fish_count, count))
        # print(fish_count)

        # COG
        if area_total[0] != 0:
            COG = cp.concatenate((COG, cp.array([[cp.round(x_x_area_total[0] / area_total[0], 3), cp.round(y_x_area_total[0] / area_total[0], 3)]])))
        else:
            COG = cp.concatenate((COG, cp.array([[cp.nan, cp.nan]])))

        # Stdev (Spread of fish)
        #GPU (somehow cp.std() doesnt work)
        # if len(positions) != 0:
        #     stdev_x, stdev_y = cp.std(positions, axis=0)
        # else:
        #     stdev_x, stdev_y = cp.array([0]), cp.array([0])
        #
        # var_x = cp.concatenate((var_x, stdev_x))
        # var_y = cp.concatenate((var_y, stdev_y))

        #CPU
        if len(positions) != 0:
            stdev_x, stdev_y = np.std(positions.get(), axis=0)
        else:
            stdev_x, stdev_y = 0, 0

        var_x.append(stdev_x)
        var_y.append(stdev_y)

        # Velocity
        # individual_fish_velocity = cp.concatenate((individual_fish_velocity, velocity_in_frame), axis=0)
        temp_list = list(velocity_in_frame.get())
        individual_fish_velocity.append(temp_list)
        # print(individual_fish_velocity)
        # print(len(individual_fish_velocity))

        if previously_tracked_count[0] != 0:
            average_speed = cp.concatenate((average_speed, cp.array([cp.round(total_velocity_in_frame[0] / previously_tracked_count[0], 3)])))
        else:
            average_speed = cp.concatenate((average_speed, cp.array([0])))

        # Swim Direction
        # individual_fish_direction = cp.concatenate((individual_fish_direction, direction_in_frame))
        temp_list = list(direction_in_frame.get())
        individual_fish_direction.append(temp_list)
        if previously_tracked_count != 0:
            average_direction = cp.concatenate((average_direction, cp.array([cp.round(total_direction_in_frame[0] / previously_tracked_count[0], 3)])))
        else:
            average_direction = cp.concatenate((average_direction, cp.array([cp.nan])))

        # Lingering
        lingering = cp.concatenate((lingering, linger_count))

        # Time still in CPU since cp and np arrays cannot handle datetime datatype
        Date.append(current_time.date())
        Time.append(current_time.time())

        current_time += datetime.timedelta(seconds=1 / fps)

        # Proceed to read the next frame
        ret, image = cap.read()

    # Release the video capture and writer objects
    cap.release()

    #Move things back to CPU
    fish_count_CPU = list(fish_count.get())
    COG_CPU = list(COG.get())
    # var_x_CPU = var_x.get()
    # var_y_CPU = var_y.get()
    # individual_fish_velocity_CPU = list(individual_fish_velocity.get())
    average_speed_CPU = list(average_speed.get())
    lingering_CPU = list(lingering.get())
    # individual_fish_direction_CPU = list(individual_fish_direction.get())
    average_direction_CPU = list(average_direction.get())

    print(len(individual_fish_velocity))
    print(len(individual_fish_direction))

    # Save the dataframe
    df = pd.DataFrame(
        {'Date': Date, 'Time': Time, 'Fish_Count': fish_count_CPU, 'COG': COG_CPU, 'Pos_Var_x': var_x, 'Pos_Var_y': var_y,
         'Individual_Vel': individual_fish_velocity, 'Avg_Speed': average_speed_CPU, 'Lingering_Count': lingering_CPU,
         'Individual_Dir': individual_fish_direction, 'Avg_Dir': average_direction_CPU
         })

    save_path = os.path.join('D:/DETR/parquets', video_name + '.parquet')
    df.to_parquet(save_path)