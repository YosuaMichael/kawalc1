import cv2
import numpy as np

drawn_image = None

def mouse_callback(event, x, y, flags, params):
    global drawn_image
    if not drawn_image is None and event == cv2.EVENT_LBUTTONDOWN:
        print("(x,y): ({},{}) | val: {}".format(x,y,drawn_image[y,x]))


def cb_show(img):
    global drawn_image
    drawn_image = img
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('img', mouse_callback)
    cv2.imshow("img", drawn_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ori_show(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

show = ori_show
img = cv2.imread("ref/pilpres_rekap.png")

cb_show(img)

x0, y0 = (2030,560)
x1, y1 = (2335,750)

extracted = img[y0:y1,x0:x1]

ori_show(extracted)
cv2.imwrite("ref/pilpres_rekap_jokowi_voting.png", extracted)


x0, y0 = (2025,850)
x1, y1 = (2345,1045)

extracted = img[y0:y1,x0:x1]

ori_show(extracted)
cv2.imwrite("ref/pilpres_rekap_prabowo_voting.png", extracted)
