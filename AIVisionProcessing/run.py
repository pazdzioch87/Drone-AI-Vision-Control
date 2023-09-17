# System / OS imports
from pathlib import Path
import argparse
import os
import sys

# SignalR imports
from signalrcore.hub_connection_builder import HubConnectionBuilder

# Displaying imports
from stream_loader import StreamLoader
from detection_dto import DetectionDto
from checks import check_imgsz
from LetterBox import LetterBox

# Ultralytics imports
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import (LOGGER)
from profiling import Profiling
import supervision as sv

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])
gestureCache = []
hub_connection = None
connected = False

def preprocess(imgsz, im):
    """Prepares input image before inference.
    Args:
        im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(pre_transform(imgsz, im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)
    img = im.to(torch.device('cpu'))
    img = img.float()  # uint8 to fp16/32
    if not_tensor:
        img /= 255  # 0 - 255 to 0.0 - 1.0
    return img

def pre_transform(imgsz, im):
    """Pre-transform input image before inference.
    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
    Return: A list of transformed imgs.
    """
    same_shapes = all(x.shape == im[0].shape for x in im)
    auto = same_shapes
    return [LetterBox(imgsz, auto=auto, stride=32)(image=x) for x in im]

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    Increments a file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and exist_ok is not set to True, the path will be incremented by appending a number and sep to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If mkdir is set to True, the path will be created as a
    directory if it does not already exist.

    Args:
        path (str, pathlib.Path): Path to increment.
        exist_ok (bool, optional): If True, the path will not be incremented and returned as-is. Defaults to False.
        sep (str, optional): Separator to use between the path and the incrementation number. Defaults to ''.
        mkdir (bool, optional): Create a directory if it does not exist. Defaults to False.

    Returns:
        (pathlib.Path): Incremented path.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def main(
        source=0, 
        weights="yolov8s.pt",
        show_preview=False, 
        process_connection=False):
    print(f'Arguments: source: {source}, weights: {weights}, show_preview: {show_preview}, process_connection: {process_connection}')
    LOGGER.info(f'ENTER THE DETECT FUNCTION')
    LOGGER.info(f'Arguments: source: {source}, weights: {weights}, show_preview: {show_preview}, process_connection: {process_connection}')
    imgsz = check_imgsz((1280, 720), min_dim=2)  # check image size

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    zone_polygon = (ZONE_POLYGON * np.array(imgsz)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(imgsz))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    print(f'ENTER THE DETECT FUNCTION')
    # pdb.set_trace() 
    if process_connection:
        # for tests not in container use this 
        # .with_url("http://controlapi/movementHub") \
        # .with_url("http://localhost:8001/movementHub") \
        # .with_url("http://controlapi:8001/movementHub") \
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

    model = YOLO(weights)
    source = str(source)
    st = f'{1}/{1}: {source}... '

    dataset = StreamLoader(source, imgsz=imgsz)
    profilers = (Profiling(), Profiling(), Profiling())
    for batch in dataset:
        path, im0s, vid_cap, s = batch

        frame = im0s[0].copy() 
        # Preprocess
        with profilers[0]:
            im = preprocess(imgsz, im0s)

        # Inference
        with profilers[1]:
            results = model(im0s[0], augment=False)
        result = results[0]
        result.speed = {
            'preprocess': profilers[0].dt * 1E3,
            'inference': profilers[1].dt * 1E3}
        
        # frame = postprocess(im0s[0], imgsz, results)   

        # if show_preview:
        #     # show(frame)
        #     cv2.imshow("Terefere", frame)
        #     cv2.waitKey(1)  # 1 millisecond

        # # plotted_image = get_plotted_image(0, results, (None, im, im0))
        # # if show-preview:
        # #     show(plotted_image)
        detections = sv.Detections.from_yolov8(results[0])

        if len(detections) > 0:
            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _
                in detections
            ]
            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections, 
                labels=labels
            )
    
            zone.trigger(detections=detections)
            frame = zone_annotator.annotate(scene=frame)      
        
        if process_connection:
            # send signalr masseges here
            print(f'{detections}')
            for detection in detections:
                x1, y1, x2, y2 = detection[0]
                confidence = detection[2]
                class_id = detection[3]
                label = f"{model.model.names[class_id]}"
                gesture_recognition(DetectionDto(coords=[float(x1), float(y1), float(x2), float(y2)], label=label, confidence=float(confidence)))

        if show_preview:
            cv2.imshow("yolov8", frame)
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

def postprocess(input_image, imgsz, outputs):
    """
    Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.
    Args:
        input_image (numpy.ndarray): The input image.
        output (numpy.ndarray): The output of the model.
    Returns:
        numpy.ndarray: The input image with detections drawn on it.
    """
    confidence_threshold = 0.5
    iou_thres = 0.7
    # Transpose and squeeze the output to match the expected shape
    output = outputs[0]
    data = output.boxes
    # Get the number of rows in the outputs array
    rows = data.shape[0]
    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []
    # Calculate the scaling factors for the bounding box coordinates
    
    # Get the height and width of the input image
    img_height, img_width = input_image.shape[:2]
    x_factor = img_width / imgsz[0]
    y_factor = img_height / imgsz[1]
    # Iterate over each row in the outputs array
    for i in range(rows):
        max_score = data[i].conf.item()
        # If the maximum score is above the confidence threshold
        if max_score >= confidence_threshold:
            # Get the class ID with the highest score
            class_id = data[i].cls.item()
            # Extract the bounding box coordinates from the current row
            x, y, w, h = data[i].xywh[0][0].item(), data[i].xywh[0][1].item(), data[i].xywh[0][2].item(), data[i].xywh[0][3].item()
            # Calculate the scaled coordinates of the bounding box
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])
    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_thres)
    # Iterate over the selected indices after non-maximum suppression
    for i in indices:
        # Get the box, score, and class ID corresponding to the index
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        # Draw the detection on the input image
        input_image = draw_detections(output, input_image, box, score, class_id)
    # Return the modified input image
    return input_image


def get_plotted_image(idx, results, batch):
    """Write inference results to a file or directory."""
    p, im, _ = batch
    log_string = ''
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    log_string += f'{idx}: '
    result = results[idx]
    plot_args = {
        'line_width': None,
        'boxes': True,
        'conf': True,
        'labels': True}
    plot_args['im_gpu'] = im[idx]
    plotted_img = result.plot(**plot_args)
    return plotted_img

def draw_detections(output, img, box, score, class_id):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.
    Args:
        img: The input image to draw detections on.
        box: Detected bounding box.
        score: Corresponding detection score.
        class_id: Class ID for the detected object.
    Returns:
        None
    """
    class_id = int(class_id)
    # Extract the coordinates of the bounding box
    x1, y1, w, h = box
    # Retrieve the color for the class ID
    classesDict = output.names
    color_palette = np.random.uniform(0, 255, size=(len(output.names), 3))
    color = color_palette[class_id]
    # Draw the bounding box on the image
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    # Create the label text with class name and score
    label = f'{classesDict[class_id]}: {score:.2f}'
    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # Calculate the position of the label text
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    # Draw a filled rectangle as the background for the label text
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                  cv2.FILLED)
    # Draw the label text on the image
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img

# def show(plotted_img):
#     print(f'proting image')
#     # cv2.imshow("Terefere", plotted_img)
#     # cv2.waitKey(1)  # 1 millisecond

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

