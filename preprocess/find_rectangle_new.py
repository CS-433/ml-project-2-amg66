import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import copy

def detect_rectangle(input_path, out_path):
    list_files = os.listdir(input_path)
    list_files.sort()
    pad_value = 2
    line_angle = 3
    line_length = 250

    for img_name in list_files:
        img_path = os.path.join(input_path, img_name)

        image = cv2.imread(img_path)
        if image.shape[1] > image.shape[0]:
            image = cv2.resize(image, (1024, 720))
        else:
            image = cv2.resize(image, (720, 1024))

        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y,u,v = cv2.split(yuv)
        y = cv2.equalizeHist(y)
        yuv = cv2.merge([y,u,v])
        image_new = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        gray = cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY)

        shape = gray.shape
        gray = gray[pad_value:shape[0]-pad_value, pad_value:shape[1]-pad_value]
        gray = np.pad(gray, ((pad_value,pad_value), (pad_value,pad_value)), 'constant',constant_values = (255,255))

        edges = cv2.Canny(gray,50,150,apertureSize=3)    #apertureSize是sobel算子大小，只能为1,3,5，7

        # Fill rectangular contours
        cnts = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        index = 0
        blank = np.zeros(gray.shape, dtype=np.uint8)
        for c in cnts:
            area = cv2.contourArea(c)
            length = np.max(c,axis=0) - np.min(c,axis=0)
            ratio = np.max(length)/(np.min(length)+1e-7)
            if area > 1500 and area <40000 and ratio<2.5 and length[0,0]<300 and length[0,1]<300:
                cv2.drawContours(blank, [c], -1, (255), -1)
                index += 1
        blank_edges = cv2.Canny(blank,50,150,apertureSize=3)
        lines = cv2.HoughLinesP(blank_edges, 1, np.pi / 180, 50, minLineLength=10)
                
        if index < 1 or lines is None:
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=10)
            blank = np.zeros(gray.shape, dtype=np.uint8)
            for line in lines:
                x1,y1,x2,y2 = line[0]
                a = y2-y1
                b = x2-x1
                if np.max([np.abs(a),np.abs(b)])/(np.min([np.abs(a),np.abs(b)])+1e-7) > line_angle:
                    y_c = np.mean([y2,y1])
                    x_c = np.mean([x2,x1])
                    initial_length = np.sqrt(a**2 + b**2)
                    ypad_length = (a*line_length)/(initial_length*2)
                    xpad_length = (b*line_length)/(initial_length*2)
                    n_y2 = int(y_c+ypad_length)
                    n_y1 = int(y_c-ypad_length)
                    n_x2 = int(x_c+xpad_length)
                    n_x1 = int(x_c-xpad_length)
                            
                    cv2.line(blank,(n_x1,n_y1),(n_x2,n_y2),(255),2)
                    cv2.line(edges,(n_x1,n_y1),(n_x2,n_y2),(255),2)
            cnts = cv2.findContours(blank, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            index = 0
            blank = np.zeros(gray.shape, dtype=np.uint8)
            for c in cnts:
                area = cv2.contourArea(c)
                length = np.max(c,axis=0) - np.min(c,axis=0)
                ratio = np.max(length)/(np.min(length)+1e-7)
                if area > 1500 and area <40000 and ratio<2.5:
                    cv2.drawContours(blank, [c], -1, (255), -1)
                    index += 1
            if index<1:
                cnts = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                index = 0
                blank = np.zeros(gray.shape, dtype=np.uint8)
                for c in cnts:
                    area = cv2.contourArea(c)
                    length = np.max(c,axis=0) - np.min(c,axis=0)
                    ratio = np.max(length)/(np.min(length)+1e-7)
                    if area > 1500 and area <40000 and ratio<2.5:
                        cv2.drawContours(blank, [c], -1, (255), -1)
                        index += 1
            if index <1:
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=5)
                for line in lines:
                    x1,y1,x2,y2 = line[0]
                    a = y2-y1
                    b = x2-x1
                    if np.max([np.abs(a),np.abs(b)])/(np.min([np.abs(a),np.abs(b)])+1e-7) > line_angle:
                        y_c = np.mean([y2,y1])
                        x_c = np.mean([x2,x1])
                        initial_length = np.sqrt(a**2 + b**2)
                        ypad_length = (a*line_length)/(initial_length*2)
                        xpad_length = (b*line_length)/(initial_length*2)
                        n_y2 = int(y_c+ypad_length)
                        n_y1 = int(y_c-ypad_length)
                        n_x2 = int(x_c+xpad_length)
                        n_x1 = int(x_c-xpad_length)

                        cv2.line(edges,(n_x1,n_y1),(n_x2,n_y2),(255),2)
                cnts = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                index = 0
                blank = np.zeros(gray.shape, dtype=np.uint8)
                for c in cnts:
                    area = cv2.contourArea(c)
                    length = np.max(c,axis=0) - np.min(c,axis=0)
                    ratio = np.max(length)/(np.min(length)+1e-7)
                    if area > 1500 and area <40000 and ratio<2.5:
                        cv2.drawContours(blank, [c], -1, (255), -1)
                        index += 1
            if index>1:
                index = 0
                blank = np.zeros(gray.shape, dtype=np.uint8)
                for c in cnts:
                    area = cv2.contourArea(c)
                    length = np.max(c,axis=0) - np.min(c,axis=0)
                    ratio = np.max(length)/(np.min(length)+1e-7)
                    if area > 1500 and area <40000 and ratio<2.5 and (np.min(c[:,0,1])>200 or np.min(c[:,0,0])<100):
                        cv2.drawContours(blank, [c], -1, (255), -1)
                        index += 1
        elif index > 2:
            blank = blank[pad_value:shape[0]-pad_value, pad_value:shape[1]-pad_value]
            blank = np.pad(blank, ((pad_value,pad_value), (pad_value,pad_value)), 'constant',constant_values = (0,0))
            edges = cv2.Canny(blank,50,150,apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=5)
            blank = np.zeros(gray.shape, dtype=np.uint8)
            for line in lines:
                x1,y1,x2,y2 = line[0]
                a = y2-y1
                b = x2-x1
                if np.max([np.abs(a),np.abs(b)])/(np.min([np.abs(a),np.abs(b)])+1e-7) > line_angle:
                    y_c = np.mean([y2,y1])
                    x_c = np.mean([x2,x1])
                    initial_length = np.sqrt(a**2 + b**2)
                    ypad_length = (a*line_length)/(initial_length*2)
                    xpad_length = (b*line_length)/(initial_length*2)
                    n_y2 = int(y_c+ypad_length)
                    n_y1 = int(y_c-ypad_length)
                    n_x2 = int(x_c+xpad_length)
                    n_x1 = int(x_c-xpad_length)                    
                    cv2.line(blank,(n_x1,n_y1),(n_x2,n_y2),(255),2)
            cnts = cv2.findContours(blank, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            index = 0
            blank = np.zeros(gray.shape, dtype=np.uint8)
            for c in cnts:
                area = cv2.contourArea(c)
                length = np.max(c,axis=0) - np.min(c,axis=0)
                ratio = np.max(length)/(np.min(length)+1e-7)
                if area > 1500 and area <40000 and ratio<2.7:
                    cv2.drawContours(blank, [c], -1, (255), -1)
                    index += 1

        # Morph open
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        opening = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel, iterations=3)

        # Draw rectangles
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(image, [c], 0, (255, 255, 255), -1)
            cv2.drawContours(image, [c], 0, (255, 255, 255), 20)
            cv2.imwrite(os.path.join(out_path, img_name[:-4] + '_processed.JPG'), image)
            cv2.waitKey()

input_path = 'D:\\Courses\\Machine learning\\Project2\\data\\all_data'
detect_rectangle(input_path, input_path)