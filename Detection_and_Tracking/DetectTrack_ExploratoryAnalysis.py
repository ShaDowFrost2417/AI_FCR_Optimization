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


def draw_centroids(frame, bboxes):
    black_frame = np.zeros_like(frame)
    centroids = [(int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)) for bbox in bboxes]
    if len(centroids) > 0:
        for centroid in centroids:
            cv2.circle(black_frame, centroid, 5, (255, 255, 255), -1)
    return black_frame

def generate_heatmap(image, bboxes, res):
    heatmap = np.zeros_like(frame[:,:,0], dtype=np.float32)

    if len(bboxes) > 0:
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(i) for i in bbox]
            heatmap[y1:y2, x1:x2] += 1

    # Apply Gaussian blur to the heatmap
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # Normalize the heatmap values between 0 and 255
    heatmap_range = np.max(heatmap) - np.min(heatmap)
    if heatmap_range != 0:
        heatmap = np.uint8(255 * (heatmap - np.min(heatmap)) / heatmap_range)
    else:
        heatmap = np.uint8(255 * (heatmap - np.min(heatmap)))

    # Apply the colormap to create a heatmap image
    heatmap_image = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    alpha = 1.0
    result_frame = cv2.addWeighted(image, 0, heatmap_image, alpha, 0)

    return result_frame

def generate_circular_heatmap_frame(frame, bboxes):
    # Same resolution as the frame
    heatmap = np.zeros_like(frame[:, :, 0], dtype=np.float32)

    for bbox in bboxes:
        x1, y1, x2, y2 = [int(i) for i in bbox]
        centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Create a circular mask centered at the bounding box centroid
        radius = min((x2 - x1) // 2, (y2 - y1) // 2)
        y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
        mask = (x - centroid_x) ** 2 + (y - centroid_y) ** 2 <= radius ** 2

        # Increment values in the circular region
        heatmap[mask] += 1

    # Gaussian blur with smaller kernel size
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)

    # Normalize the heatmap values between 0 and 255
    heatmap = np.uint8(255 * (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)))

    # Apply the colormap to create a heatmap image
    heatmap_image = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Full weight to the heatmap, no blending with the original frame
    alpha = 1.0
    result_frame = cv2.addWeighted(frame, 0, heatmap_image, alpha, 0)

    return result_frame

# Example usage:
# result_frame = generate_circular_heatmap_frame(frame, bboxes)
# cv2.imshow('Result Frame', result_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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

video_names = ['2023-11-26_19-53']

for video_name in video_names:
    # Video
    video_path = os.path.join('.', 'data', video_name+'.mp4')
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()

    # Adjust the frame rate and resolution as needed
    fps=20.0
    res=(640,480)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi or 'mp4v' for .mp4

    #Video annotated with detections
    # out_path = os.path.join('.', 'tracking_test', video_name+'_test.mp4')
    # out = cv2.VideoWriter(out_path, fourcc, fps, res)

    #Particle View
    out_path2 = os.path.join('.', 'processed', 'Twice_a_day', 'Particle', video_name+'_particle.mp4')
    out2 = cv2.VideoWriter(out_path2, fourcc, fps, res)

    #Heatmap
    # out_path3 = os.path.join('.', 'processed', 'Twice_a_day', 'Heatmap', video_name+'_heatmap.mp4')
    # out3 = cv2.VideoWriter(out_path3, fourcc, fps, res)
    #
    # out_path4 = os.path.join('.', 'processed', 'Twice_a_day', 'Heatmap', video_name+'_circleheatmap.mp4')
    # out4 = cv2.VideoWriter(out_path4, fourcc, fps, res)

    #Detection
    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    MODEL_PATH = os.path.join('.', 'custom-model-zebra1-100')
    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    model.to(DEVICE)

    data = {'track_id': [], 'bbox': []}
    df = pd.DataFrame(data)

    while ret:
        start = datetime.datetime.now()
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

        box_annotator = sv.BoxAnnotator()

        # Only take when class_id in acceptable_class
        acceptable_class = [0, 1, 2, 3]
        xyxy_list = []
        confidence_list = []
        class_id_list = []

        for xyxy, confidence, class_id, _ in detections:
            if class_id in acceptable_class:
                xyxy_list.append(xyxy)
                confidence_list.append(confidence)
                class_id_list.append(class_id)

        xyxy_list = np.array(xyxy_list)
        class_id_list = np.array(class_id_list)
        confidence_list = np.array(confidence_list)

        # print(np.array(xyxy_list))
        # print(np.array(class_id_list))
        # print(np.array(confidence_list))
        # print('----------')

        try:
            detections = sv.Detections(xyxy=xyxy_list, class_id=class_id_list, confidence=confidence_list)

            frame = box_annotator.annotate(scene=image,
                                           detections=detections,
                                           labels=[f"{model.config.id2label[class_id]}" for _, confidence, class_id, _ in
                                                   detections]
                                           )

            # Tracking
            track_outputs[0] = tracker.update(detections.xyxy, detections.confidence, detections.class_id, image)

            track_id_list = []
            bbox_list = []

            for index, track_output in enumerate(track_outputs[0]):
                bbox = track_output[0:4]
                track_id = track_output[4]
                cls = track_output[5]

                print(track_id)

                # Annotate tracking ID to frame for output video 'out'
                # top_left_bbox = (int(bbox[0]), int(bbox[1]))
                # cv2.putText(frame, f"ID : {track_id}", top_left_bbox, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Save track_id and bbox in a dataframe, export as .pt file (?)
                track_id_list.append(track_id)
                bbox_list.append(bbox)

            # out.write(frame) #Video annotated with detections
            out2.write(draw_centroids(frame,xyxy_list)) #Particle View
            # out3.write(generate_heatmap(frame, xyxy_list, res))  #Heatmap
            # out4.write(generate_circular_heatmap_frame(frame, xyxy_list))  # Heatmap


        except:
            # out.write(frame)  #Video annotated with detections
            out2.write(draw_centroids(frame, xyxy_list)) #Particle View
            # out3.write(generate_heatmap(frame, xyxy_list, res))  # Heatmap
            # out4.write(generate_circular_heatmap_frame(frame, xyxy_list))  # Heatmap

        df = df._append({"track_id":track_id_list, "bbox": bbox_list}, ignore_index=True)

        #Read the next frame and reiterate the while loop
        ret, image = cap.read()


    # Release the video capture and writer objects
    # out.release()
    out2.release()
    # out3.release()
    # out4.release()
    cap.release()

    # df.to_parquet(video_name + '.parquet', index=False)
    df.to_csv(video_name + '.csv', index=False)