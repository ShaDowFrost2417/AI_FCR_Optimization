import os
import numpy as np
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import supervision as sv
import datetime
from strong_sort.utils.parser import YamlParser
from strong_sort.strong_sort import StrongSORT
import pandas as pd

def calculate_direction(initial_x, initial_y, final_x, final_y):
    # Calculate the displacement vector
    displacement_x = final_x - initial_x
    displacement_y = final_y - initial_y

    # Calculate the angle in radians using arctan2
    angle_rad = np.arctan2(displacement_y, displacement_x)

    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)

    # Ensure the angle is positive
    angle_deg = (angle_deg + 360) % 360

    return angle_deg


#Detection Settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.65

#Tracker Settings
reid_weights = os.path.join('.','strong_sort','weights','osnet_x0_25_msmt17.pt')
tracker_config = "strong_sort/configs/strong_sort.yaml"
cfg = YamlParser()
cfg.merge_from_file(tracker_config)
track_outputs = [None]

video_file_path = 'D:/Study/NTU/FYP/Dataset/Twice_a_day/Use'

for video_file in os.listdir(video_file_path):
    print(video_file)

    #Global variable Initialization
    Date = []
    Time = []
    fish_count = []
    COG = []
    var_x = []
    var_y = []
    individual_fish_velocity = []
    average_speed = []
    lingering = []

    individual_fish_direction = []
    average_direction = []

    previous_coords = {}

    #Get the time from video file name
    video_name = video_file.split('.mp4')[0]
    current_time = ' '.join(video_name.split('_'))
    current_time = current_time[:-3] + ':' + current_time[-2:]
    current_time = pd.to_datetime(current_time)
    


    #Setting Up Tracker
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

    #Video
    video_path = os.path.join(video_file_path, video_file)
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    fps = 20


    #Detection
    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    MODEL_PATH = os.path.join('.', 'custom-model-zebra1-100')
    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    model.to(DEVICE)

    while ret:
        start = datetime.datetime.now()
        try:
            with torch.no_grad():
                #load image and predict
                inputs = image_processor(images = image, return_tensors = 'pt').to(DEVICE)
                outputs = model(**inputs)

                #post-process
                target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
                results = image_processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=CONFIDENCE_THRESHOLD,
                    target_sizes=target_sizes,
                )[0]

            #annotate with non-max suppression (nms)
            try:
                detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_THRESHOLD)
            except:
                print("no_nms")
                detections = sv.Detections.from_transformers(transformers_results=results)

            # box_annotator = sv.BoxAnnotator()

            # # Only take when class_id in acceptable_class
            # acceptable_class = [0, 1, 2, 3]
            # xyxy_list = []
            # confidence_list = []
            # class_id_list = []
            #
            # for xyxy, confidence, class_id, _ in detections:
            #     if class_id in acceptable_class:
            #         xyxy_list.append(xyxy)
            #         confidence_list.append(confidence)
            #         class_id_list.append(class_id)
            #
            # xyxy_list = np.array(xyxy_list)
            # class_id_list = np.array(class_id_list)
            # confidence_list = np.array(confidence_list)

            # Variable initialization
            count = 0
            previously_tracked_count = 0
            total_velocity_in_frame = 0
            total_direction_in_frame = 0
            x_x_area_total = 0  #x centroid multiplied by area of bounding box
            y_x_area_total = 0  #y centroid multiplied by area of bounding box
            area_total = 0
            linger_count = 0
            positions = []
            velocity_in_frame = []
            direction_in_frame = []

            try:
                # detections = sv.Detections(xyxy=xyxy_list, class_id=class_id_list, confidence=confidence_list)

                # Tracking
                track_outputs[0] = tracker.update(detections.xyxy, detections.confidence, detections.class_id, image)



                #Iterate for every fish tracked inside the frame
                for index, track_output in enumerate(track_outputs[0]):
                    x1,y1,x2,y2,track_id = track_output[0:5] #bbox & track_id

                    #Getting centroid of the bounding box
                    c_x, c_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    ################ For STDev later ################
                    positions.append((c_x, c_y))

                    ################ COG ################
                    area = np.abs((x2-x1)*(y2-y1))
                    area_total += area
                    x_x_area_total += area * c_x
                    y_x_area_total += area * c_y


                    # if track_id has already been recorded previously
                    # for properties that require prior tracking data (at least 2 bounding box data of the same track ID), i.e. velocity, lingering, direction
                    if track_id in previous_coords:
                        previously_tracked_count += 1
                        prev_c_x, prev_c_y, prev_Time, prev_Dist = previous_coords[track_id]

                        ################ Velocity (50 frames Rolling) ################
                        dist = np.sqrt((c_x - prev_c_x) ** 2 + (c_y - prev_c_y) ** 2)
                        latestDist = prev_Dist[-49:]
                        latestTime = prev_Time[-49:]
                        rollingDist = np.sum(latestDist) + dist
                        velocity = rollingDist / (np.sum(latestTime) + (1 / fps))

                        #update values
                        velocity_in_frame.append((track_id,velocity))
                        total_velocity_in_frame += velocity
                        prev_Time.append(1 / fps)
                        prev_Dist.append(dist)

                        ################ Swimming Direction (similar to Turning angle) ################
                        direction = calculate_direction(prev_c_x, prev_c_y, c_x, c_y)
                        direction_in_frame.append((track_id,direction))
                        total_direction_in_frame += direction

                        ################ Linger ################
                        if dist <= 2:
                            linger_count += 1

                    else: #Otherwise initialize the velocity
                        prev_Dist = [0]
                        prev_Time = [0]

                    # input into box labels
                    previous_coords[track_id] = (c_x, c_y, prev_Time, prev_Dist)

                    #fish count inside the frame
                    count += 1

            except:
                pass
        except:
            pass

        ##Append data for final dataframe
        # fish count
        fish_count.append(count)
        print(fish_count)

        # COG
        if area_total != 0:
            COG.append((np.round(x_x_area_total/area_total, 3), np.round(y_x_area_total/area_total, 3)))
        else:
            COG.append((np.nan, np.nan))

        #Stdev (Spread of fish)
        if positions != []:
            stdev_x, stdev_y = np.std(positions, axis=0)
        else:
            stdev_x, stdev_y = 0, 0

        var_x.append(stdev_x)
        var_y.append(stdev_y)

        #Velocity
        individual_fish_velocity.append(velocity_in_frame)

        if previously_tracked_count != 0:
            average_speed.append(np.round(total_velocity_in_frame/previously_tracked_count, 3))
        else:
            average_speed.append(0)

        #Swim Direction
        individual_fish_direction.append(direction_in_frame)
        if previously_tracked_count != 0:
            average_direction.append(np.round(total_direction_in_frame/previously_tracked_count, 3))
        else:
            average_direction.append(np.nan)

        #Lingering
        lingering.append(linger_count)

        #Time
        Date.append(current_time.date())
        Time.append(current_time.time())

        current_time += datetime.timedelta(seconds=1/fps)

        #Proceed to read the next frame
        ret, image = cap.read()

    # Release the video capture and writer objects
    cap.release()

    #Save the dataframe
    df = pd.DataFrame({'Date': Date, 'Time': Time, 'Fish_Count': fish_count, 'COG': COG, 'Pos_Var_x': var_x, 'Pos_Var_y': var_y,
                       'Individual_Vel': individual_fish_velocity, 'Avg_Speed': average_speed, 'Lingering_Count': lingering,
                       'Individual_Dir': individual_fish_direction, 'Avg_Dir': average_direction
                       })

    save_path = os.path.join('D:/DETR/parquets', video_name + '.parquet')
    df.to_parquet(save_path)





