
from detect_rectangle import *
import os


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "/media/data/mu/ML2/data2/our_data/HIV/1/"
    input_folder = "RxT_diabetiques_Benin/"

    input_path = path + input_folder
    out_path = path + "deleted_label/"
    os.makedirs(out_path, exist_ok=True)

    detect_rectangle(path, out_path)

# imgname = "/media/data/mu/ML2/data2/our_data/HIV/images/210001_0.JPG"
# img = cv2.imread(imgname)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ## Split the H channel in HSV, and get the red range
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# h,s,v = cv2.split(hsv)
# h[h<150]=0
# h[h>180]=0

# ## normalize, do the open-morp-op
# normed = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3,3))
# opened = cv2.morphologyEx(normed, cv2.MORPH_OPEN, kernel)
# res = np.hstack((h, normed, opened))
# cv2.imwrite("tmp1.png", res)

# contours = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))[-2]

# bboxes = []
# rboxes = []
# cnts = []
# dst = img.copy()
# for cnt in contours:
#     ## Get the stright bounding rect
#     bbox = cv2.boundingRect(cnt)
#     x,y,w,h = bbox
#     if w<30 or h < 30 or w*h < 2000 or w > 500:
#         continue

#     ## Draw rect
#     cv2.rectangle(dst, (x,y), (x+w,y+h), (255,0,0), 1, 16)

#     ## Get the rotated rect
#     rbox = cv2.minAreaRect(cnt)
#     (cx,cy), (w,h), rot_angle = rbox
#     print("rot_angle:", rot_angle)  

#     ## backup 
#     bboxes.append(bbox)
#     rboxes.append(rbox)
#     cnts.append(cnt)