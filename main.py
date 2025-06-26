import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

def load_video(video_path):
    """
    Загружает видео из файла.

    :param video_path: путь к файлу видео.
    :return: объект видео.
    """
    return cv2.VideoCapture(video_path)

def load_model(weights_path, cfg_path):
    """
    Загружает веса модели.

    :param weights_path: путь к файлу весов модели.
    :param cfg_path: путь к файлу конфигурации модели.
    :return: объект модели.
    """
    return cv2.dnn.readNet(weights_path, cfg_path)

def load_classes(class_path):
    """
    Загружает список классов объектов.

    :param class_path: путь к файлу списка классов.
    :return: список классов.
    """
    with open(class_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def create_writer(fps, width, height, output_path):
    """
    Создает видео-писатель.

    :param fps: частота кадров.
    :param width: ширина кадра.
    :param height: высота кадра.
    :param output_path: путь к файлу вывода.
    :return: объект видео-писателя.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def detect_objects(frame, net, classes):
    """
    Детектирует объекты на кадре видео.

    :param frame: кадр видео.
    :param net: объект модели.
    :param classes: список классов.
    :return: список детектированных объектов.
    """
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    detected_objects = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classes[classID] == 'person' and confidence > 0.5:
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                detected_objects.append((x, y, int(width), int(height), classes[classID], confidence))
    return detected_objects

def draw_objects(frame, detected_objects):
    """
    Отрисовывает детектированные объекты на кадре видео.

    :param frame: кадр видео.
    :param detected_objects: список детектированных объектов.
    """
    for x, y, w, h, label, confidence in detected_objects:
        for obj_x, obj_y, obj_w, obj_h, obj_label, obj_confidence in detected_objects:
            if obj_label == label and obj_x != x and obj_y != y and obj_x + obj_w > x and obj_x < x + w and obj_y + obj_h > y and obj_y < y + h:
                detected_objects.remove((obj_x, obj_y, obj_w, obj_h, obj_label, obj_confidence))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {int(confidence*100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)

def main():
    """
    Основная функция программы.

    :return: None
    """
    video_path = 'crowd.mp4'
    weights_path = 'yolov3.weights'
    cfg_path = 'yolov3.cfg'
    class_path = 'coco.names'
    output_path = 'output.mp4'

    video = load_video(video_path)
    net = load_model(weights_path, cfg_path)
    classes = load_classes(class_path)
    writer = create_writer(video.get(cv2.CAP_PROP_FPS), int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), output_path)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        (H, W) = frame.shape[:2]
        detected_objects = detect_objects(frame, net, classes)
        draw_objects(frame, detected_objects)
        cv2.imshow('Video', frame)
        writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
