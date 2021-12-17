import cv2
import numpy as np
import os

def detect_rectangle(input_path, out_path):
    list_files = os.listdir(input_path)

    for img_name in list_files:
        if '.JPG' not in img_name:
            continue
        img_path = input_path + img_name
        print('image', img_path)

        image = cv2.imread(img_path)
        image = cv2.resize(image, (1024, 720))
        result = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 30)

        # Fill rectangular contours
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)

        # Morph open
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

        # Draw rectangles
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(image, [c], 0, (255, 255, 255), -1)
            cv2.drawContours(image, [c], 0, (255, 255, 255), 20)

            cv2.imwrite(os.path.join(out_path, "without_rect_" + img_name), image)

            cv2.waitKey()

