# System / OS imports
from pathlib import Path
import argparse
import os
import sys
import time

# SignalR imports
from signalrcore.hub_connection_builder import HubConnectionBuilder

# Displaying imports
from stream_loader import StreamLoader
from detection_dto import DetectionDto

# Ultralytics imports
import cv2
from ultralytics import YOLO
from ultralytics.utils import (LOGGER)
import supervision as sv

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

gestureCache = []
hub_connection = None
connected = False

def main(
        source=0, 
        weights="yolov8s.pt",
        show_preview=False, 
        process_connection=False):
    print(f'Arguments: source: {source}, weights: {weights}, show_preview: {show_preview}, process_connection: {process_connection}')
    LOGGER.info(f'ENTER THE DETECT FUNCTION')
    LOGGER.info(f'Arguments: source: {source}, weights: {weights}, show_preview: {show_preview}, process_connection: {process_connection}')

    annotator_bbox = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    print(f'ENTER THE DETECT FUNCTION')
    if process_connection:
        # for tests in container use this 
        # .with_url("http://controlapi:8001/movementHub") \ docker compose
        global hub_connection
        hub_connection = HubConnectionBuilder() \
        .with_url("http://localhost:8001/movementHub") \
        .with_automatic_reconnect({                     
            "type": "raw",                              
            "keep_alive_interval": 10,                   
            "reconnect_interval": 5,                    
            "max_attempts": 5                      
        }).build()
        hub_connection.start()
        hub_connection.on_error(lambda data: connection_error_handler(data))
        hub_connection.on_open(lambda: connection_opend(hub_connection))
        print(f'Printing HUB CONNECION STATE:')
        print(f"Connection state is: {hub_connection.transport.state.name}")   
        while connected is False:  
            print(f'waiting for connection to be established')
            time.sleep(1) 

    model = YOLO(weights)
    source = str(source)

    dataset = StreamLoader(source)
    for batch in dataset:
        path, im0s, vid_cap, s = batch

        frame = im0s[0].copy() 
        results = model(im0s[0], augment=False)
        detections = sv.Detections.from_yolov8(results[0])

        if len(detections) > 0:
            labels = [
                f"{model.model.names[class_id]}"
                for _, _, confidence, class_id, _
                in detections
            ]
            frame = annotator_bbox.annotate(
                scene=frame, 
                detections=detections, 
                labels=labels
            )
        
        if process_connection:
            # send signalr masseges here
            # print(f'{detections}')
            for detection in detections:
                x1, y1, x2, y2 = detection[0]
                confidence = detection[2]
                class_id = detection[3]
                label = f"{model.model.names[class_id]}"
                gesture_recognition(DetectionDto(coords=[float(x1), float(y1), float(x2), float(y2)], label=label, confidence=float(confidence)))

        if show_preview:
            cv2.imshow("Drone preview", frame)
            if (cv2.waitKey(30) == 27):
                break

def connection_error_handler(data):
    print(f"An exception was thrown closed{data.error}")    

def connection_opend(hub_connection):
    print(f"Connection opened")
    print(f"Connection state is: {hub_connection.transport.state.name}")   
    global connected
    connected = True

def gesture_recognition(detection: DetectionDto):
    global gestureCache
    global hub_connection
    gestureCache.append(detection.label)
    if len(gestureCache) > 3:
        if all(element == detection.label for element in gestureCache):
            hub_connection.send("SendDetection", [detection])
            LOGGER.info(f'Detected gesture: {detection.label}')
        gestureCache = []

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov8s.pt', help='model path or triton URL')
    parser.add_argument('--show-preview', action="store_true", default=False)
    parser.add_argument('--process-connection', action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print(f'ENTER THE SCRIPT')
    args = parse_opt()
    main(args.source, args.weights, args.show_preview, args.process_connection)

