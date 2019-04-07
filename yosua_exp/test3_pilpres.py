# Test code for extracting the voting box from pilpres form

import cv2
import numpy as np


ref = cv2.imread("ref/pilpres_rekap.png")
code = "2"
img = cv2.imread("sample/pilpres_rekap_{}.jpg".format(code))

#target_h = 2000
#if ref.shape[0] > target_h:
#    scale = target_h/ref.shape[0]
#    ref = cv2.resize(ref, (0,0), fx=scale, fy=scale)

#if img.shape[0] > target_h:
#    scale = target_h/img.shape[0]
#    img = cv2.resize(img, (0,0), fx=scale, fy=scale)

print(ref.shape)
print(img.shape)

brisk = cv2.BRISK_create()
ref_kp, ref_dsc = brisk.detectAndCompute(ref, None)
im_kp, im_dsc = brisk.detectAndCompute(img, None)


# Match descriptors.
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(im_dsc.astype(np.float32), ref_dsc.astype(np.float32),k=2)

#bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#raw_matches = bf.knnMatch(im_dsc, ref_dsc)

# Filter matches
good = []
for m in matches:
    if len(m) == 2 and m[0].distance < 0.75*m[1].distance:
        good.append(m[0])


# Get homography
print(len(good))
MIN_MATCH_COUNT = 10
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ im_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ ref_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,d = img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(ref,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

# Draw Matches
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(img,im_kp,ref,ref_kp,good,None,**draw_params)
target_h = 640
if img3.shape[0] > target_h:
    scale = target_h/img3.shape[0]
    img3 = cv2.resize(img3, (0,0), fx=scale, fy=scale)

cv2.imshow("img", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Draw transformed
img_transformed = cv2.warpPerspective(img, M, ref.shape[:2][::-1])
img_show = img_transformed
target_h = 640
if img_show.shape[0] > target_h:
    scale = target_h / img_show.shape[0]
    img_show = cv2.resize(img_show, (0,0), fx=scale, fy=scale)
cv2.imshow("img", img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Extract jokowi voting
print(img_transformed.shape)
x0, y0 = (2030,560)
x1, y1 = (2335,750)
extracted = img_transformed[y0:y1,x0:x1]
cv2.imshow("img", extracted)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("sample/partai_rekap_jokowi_voting.png", extracted)

# Extract jokowi voting
print(img_transformed.shape)
x0, y0 = (2025,850)
x1, y1 = (2345,1045)
extracted = img_transformed[y0:y1,x0:x1]
cv2.imshow("img", extracted)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("sample/partai_rekap_prabowo_voting.png", extracted)
