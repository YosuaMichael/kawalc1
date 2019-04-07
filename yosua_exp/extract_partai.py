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
code = "plano_gerindra"
img = cv2.imread("ref/dpr_{}.png".format(code))

#cb_show(img)

x0, y0 = 275, 450
x1, y1 = 675, 680

extracted = img[y0:y1,x0:x1]

ori_show(extracted)
cv2.imwrite("ref/partai_{}.png".format(code), extracted)


