from facelib import AgeGenderEstimator, FaceDetector

from mtcnn import MTCNN
import numpy as np
import cv2
import os
from typing import List

def convert_tensor_to_int(tensor) -> List[int]:
    return list(map(lambda x : int(x.data), tensor))

def convert_bboxes(bboxes) -> List[List[int]]:
    return list(map(lambda x: convert_tensor_to_int(x), bboxes))
    

def main():
    image_path = "test3.jpg"

    age_gender_detector = AgeGenderEstimator()
    face_detector = FaceDetector()

    # read input image
    image = cv2.imread(image_path)


    faces, boxes, _, _ = face_detector.detect_align(image)
    boxes = convert_bboxes(boxes)
    genders, _ = age_gender_detector.detect(faces)

    for (startX, startY, endX, endY), gender in zip(boxes, genders):
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, gender, (startX, startY -10),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()