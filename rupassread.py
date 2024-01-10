import imutils
from imutils.contours import sort_contours
import numpy as np
import pytesseract
import cv2


def get_data_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (H, W) = gray.shape
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="bottom-to-top")[0]
    boxes = []
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        boxes.append(box)
    return boxes


if __name__ == "__main__":
    shift_border = 0

    image = cv2.imread('<path image>')
    h, w, _ = image.shape
    boxes = get_data_boxes(image)
    all_boxes = image.copy()

    result = []
    cnt = 0
    for box in boxes:
        area = cv2.contourArea(box)
        x1 = np.max([np.min(
            [box[0][0], box[1][0], box[2][0], box[3][0]]
        ) - shift_border, 0])
        x2 = np.min([np.max(
            [box[0][0], box[1][0], box[2][0], box[3][0]]
        ) + shift_border, w])
        y1 = np.max([np.min(
            [box[0][1], box[1][1], box[2][1], box[3][1]]
        ) - shift_border, 0])
        y2 = np.min([np.max(
            [box[0][1], box[1][1], box[2][1], box[3][1]]
        ) + shift_border, h])

        if y2 - y1 > 0 and x2 - x1 > 0 and area > 300:
            cv2.drawContours(all_boxes, [box], 0, (255, 0, 0), 2)
            result.append(
                pytesseract.image_to_string(
                    image[y1:y2, x1:x2],
                    lang='rus'
                )
            )

    print(result)
    cv2.imshow('all_boxes', all_boxes)
    cv2.waitKey(0)
