import easyocr
import cv2
import numpy as np


reader = easyocr.Reader(['ru'])
image_path = '<path image>'
image = cv2.imread(image_path)

res_img = reader.readtext(image_path)
for box, text, val in res_img:
    for i in range(4):
        for j in range(2):
            box[i][j] = int(box[i][j])
    cv2.drawContours(image, [np.array(box)], 0, (255, 0, 0), 2)


image_rot = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
res_img_rot = reader.readtext(image_rot)
for box, text, val in res_img_rot:
    for i in range(4):
        for j in range(2):
            box[i][j] = int(box[i][j])
    cv2.drawContours(image_rot, [np.array(box)], 0, (255, 0, 0), 2)

cv2.imshow('img', image)
cv2.imshow('img_rot', image_rot)
cv2.waitKey(0)
