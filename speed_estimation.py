import cv2
import torch
import numpy as np
import os
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def show_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

def show_counter(frame, title, class_names, vehicle_count, x_init):
    overlay = frame.copy()
    y_init = 100
    gap = 30
    alpha = 0.5
    cv2.rectangle(overlay, (x_init - 5, y_init - 35), (x_init + 200, 265), (0, 255, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, title, (x_init, y_init - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for vehicle_id, count in vehicle_count.items():
        y_init += gap
        vehicle_name = class_names[vehicle_id]
        vehicle_count_str = "%.3i" % count
        cv2.putText(frame, vehicle_name, (x_init, y_init), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, vehicle_count_str, (x_init + 135, y_init), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

def show_region(frame, points):
    for id, point in enumerate(points):
        start_point = (int(points[id - 1][0]), int(points[id - 1][1]))
        end_point = (int(point[0]), int(point[1]))
        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

def transform_points(perspective, points):
    if points.size == 0:
        return points
    reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(reshaped_points, perspective)
    return transformed_points.reshape(-1, 2)

def add_position_time(track_id, current_position, track_data):
    track_time = time.time()
    if track_id in track_data:
        track_data[track_id]['position'].append(current_position)
    else:
        track_data[track_id] = {'position': [current_position], 'time': track_time}

def calculate_speed(start, end, start_time):
    now = time.time()
    move_time = now - start_time
    distance = abs(end - start) / 10
    speed = (distance / move_time) * 3.6
    return speed

def speed_estimation(vehicle_position, speed_region, perspective_region, track_data, track_id, text):
    min_x, max_x = int(np.amin(speed_region[:, 0])), int(np.amax(speed_region[:, 0]))
    min_y, max_y = int(np.amin(speed_region[:, 1])), int(np.amax(speed_region[:, 1]))
    if (vehicle_position[0] in range(min_x, max_x)) and (vehicle_position[1] in range(min_y, max_y)):
        points = np.array([[vehicle_position[0], vehicle_position[1]]], dtype=np.float32)
        point_transform = transform_points(perspective_region, points)
        add_position_time(track_id, int(point_transform[0][1]), track_data)
        if len(track_data[track_id]['position']) > 5:
            start_position = track_data[track_id]['position'][0]
            end_position = track_data[track_id]['position'][-1]
            start_estimate = track_data[track_id]['time']
            speed = calculate_speed(start_position, end_position, start_estimate)
            text += f" - {speed:.2f} km/h"
    return text

def run_speed_estimation(video_input, weights_path, class_names, conf_threshold=0.5):
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print('❌ Error: Unable to open video source.')
        return

    os.makedirs('runs', exist_ok=True)
    video_name = os.path.basename(video_input).split('.')[0]
    output_path = f"runs/{video_name}_out.avi"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps_input, (width, height))

    tracker = DeepSort(max_age=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(weights_path)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    entered_vehicle_ids = []
    exited_vehicle_ids = []
    vehicle_class_ids = [1, 2, 3, 5, 7]
    vehicle_entry_count = {cls_id: 0 for cls_id in vehicle_class_ids}
    vehicle_exit_count = {cls_id: 0 for cls_id in vehicle_class_ids}

    entry_line = {'x1': 160, 'y1': 558, 'x2': 708, 'y2': 558}
    exit_line = {'x1': 1155, 'y1': 558, 'x2': 1718, 'y2': 558}
    offset = 20

    speed_region_1 = np.float32([[280, 460], [670, 470], [570, 820], [170, 800]])
    speed_region_2 = np.float32([[1030, 480], [1400, 470], [1550, 850], [1150, 860]])
    target_1 = np.float32([[0, 0], [150, 0], [150, 270], [0, 270]])
    target_2 = np.float32([[0, 0], [120, 0], [120, 270], [0, 270]])
    perspective_region_1 = cv2.getPerspectiveTransform(speed_region_1, target_1)
    perspective_region_2 = cv2.getPerspectiveTransform(speed_region_2, target_2)

    track_data = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    print("✅ Starting speed estimation...")

    while True:
        start_time = time.time()  # ✅ only this one needed
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.predict(frame, verbose=False)
        cv2.line(frame, (entry_line['x1'], entry_line['y1']), (exit_line['x2'], exit_line['y2']), (0, 127, 255), 3)
        show_region(frame, speed_region_1)
        show_region(frame, speed_region_2)

        detect = []
        result = results[0]
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            if conf < conf_threshold:
                continue
            detect.append([[x1, y1, x2 - x1, y2 - y1], conf, class_id])

        tracks = tracker.update_tracks(detect, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            class_id = track.get_det_class()
            color = colors[class_id]
            B, G, R = map(int, color)

            text = f"{track_id} - {class_names[class_id]}"
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            vehicle_position = (center_x, y2)
            text = speed_estimation(vehicle_position, speed_region_1, perspective_region_1, track_data, track_id, text)
            vehicle_position = (center_x, y1)
            text = speed_estimation(vehicle_position, speed_region_2, perspective_region_2, track_data, track_id, text)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if (center_x in range(entry_line['x1'], entry_line['x2'])) and (center_y in range(entry_line['y1'], entry_line['y1'] + offset)):
                if track_id not in entered_vehicle_ids and class_id in vehicle_class_ids:
                    vehicle_entry_count[class_id] += 1
                    entered_vehicle_ids.append(track_id)

            if (center_x in range(exit_line['x1'], exit_line['x2'])) and (center_y in range(exit_line['y1'] - offset, exit_line['y1'])):
                if track_id not in exited_vehicle_ids and class_id in vehicle_class_ids:
                    vehicle_exit_count[class_id] += 1
                    exited_vehicle_ids.append(track_id)

        show_counter(frame, "Vehicle Enter", class_names, vehicle_entry_count, 10)
        show_counter(frame, "Vehicle Exit", class_names, vehicle_exit_count, 1710)
        fps = 1 / max((time.time() - start_time), 1e-5)
        show_fps(frame, round(fps, 2))

        out.write(frame)

        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"✅ Done! Output video saved at: {output_path}")
