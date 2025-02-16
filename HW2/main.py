import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
reader = easyocr.Reader(['en'], gpu=True)


def warp_plate(img, corners):
    # 設定車牌的標準尺寸
    plate_width, plate_height = 240, 80
    #車牌目標座標
    dst_rect = np.array([
        [0, 0],
        [plate_width-1, 0],
        [plate_width-1, plate_height-1],
        [0, plate_height-1],
    ], dtype='float32')

    # 進行透視變換
    M = cv2.getPerspectiveTransform(corners, dst_rect)
    warped = cv2.warpPerspective(img, M, (plate_width, plate_height))
    return warped

def process_frame_for_plate(frame):
    #轉成灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #減少噪聲
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    #使用canny算法
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
    #找輪廓
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]


    #找出車牌最可能位置
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        return frame, None

    '''
    #擷取出車牌位置
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(frame, frame, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

    #reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    '''


    # 使用 warp_plate 進行透視變換
    corners = location.reshape(4, 2).astype('float32')
    warped_plate = warp_plate(gray, corners)

    # 將下一部分的文字識別應用於 warped_plate 而不是 cropped_image
    result = reader.readtext(warped_plate)


    if not result:
        return frame, None

    text = result[0][-2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(frame, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1,
                      color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(frame, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
    return res, text

# 使用相機
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame, plate_text = process_frame_for_plate(frame)

    cv2.imshow('Live License Plate Detection', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
