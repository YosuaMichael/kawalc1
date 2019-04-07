# Test code to match the reference partai and sample partai

import cv2
import numpy as np

import sys

ref_code = sys.argv[1]
code = sys.argv[2]
print(ref_code, code)

#ref_code = "gerindra"
ref = cv2.imread("ref/partai_plano_{}.png".format(ref_code))
#code = "pkb"
img = cv2.imread("sample/partai_plano_{}.png".format(code))

def rotate_image(mat, angle):
  # angle in degrees

  height, width = mat.shape[:2]
  image_center = (width/2, height/2)

  rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

  abs_cos = abs(rotation_mat[0,0])
  abs_sin = abs(rotation_mat[0,1])

  bound_w = int(height * abs_sin + width * abs_cos)
  bound_h = int(height * abs_cos + width * abs_sin)

  rotation_mat[0, 2] += bound_w/2 - image_center[0]
  rotation_mat[1, 2] += bound_h/2 - image_center[1]

  rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  return rotated_mat

ref = rotate_image(ref, 90)
img = rotate_image(img, 90)

print(ref.shape)
print(img.shape)


brisk = cv2.BRISK_create()
ref_kp, ref_dsc = brisk.detectAndCompute(ref, None)
im_kp, im_dsc = brisk.detectAndCompute(img, None)


# Match descriptors.
# Using FLANN
#FLANN_INDEX_KDTREE = 1
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks = 50)
#flann = cv2.FlannBasedMatcher(index_params, search_params)
#matches = flann.knnMatch(im_dsc.astype(np.float32), ref_dsc.astype(np.float32),k=2)



# Using Bruteforce matching
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(im_dsc.astype(np.float32), ref_dsc.astype(np.float32), k=2)

# Filter matches
good = []
for m in matches:
    if len(m) == 2 and m[0].distance < 0.8*m[1].distance:
        good.append(m[0])
#good = [m[0] for m in matches]


# Get homography
print(len(good))
MIN_MATCH_COUNT = 10
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ im_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ ref_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    #h,w,d = img.shape
    #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #dst = cv2.perspectiveTransform(pts,M)
    #img2 = cv2.polylines(ref,[np.int32(dst)],True,255,3, cv2.LINE_AA)
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

cv2.imwrite("result/partai_{}_x_{}.png".format(code, ref_code), img3)








