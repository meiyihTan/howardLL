import os
import numpy as np
import cv2
from skimage import exposure

# dark_img_path = './dataset/Sony/short/00001_00_0.033s.png'

# rgb = cv2.imread(dark_img_path, cv2.COLOR_BGR2RGB) * 150
# img = np.uint8(np.minimum(rgb, 255.0))
# cv2.imwrite('Sony_expose_example/dark_img_exposed150.png', img)


# dark_img_path = './dataset/Sony/short/00001_00_0.04s.png'

# rgb = cv2.imread(dark_img_path, cv2.COLOR_BGR2RGB) * 50
# img = np.uint8(np.minimum(rgb, 255.0))
# cv2.imwrite('Sony_expose_example/dark_img_exposed50.png', img)


dark_img_path = './dataset/SID/Sony/test/short/10054_00_0.1s.png'

rgb = cv2.imread(dark_img_path, cv2.COLOR_BGR2RGB) * 50
img = np.uint8(np.minimum(rgb, 255.0))
cv2.imwrite('dark_img_exposed.png', img)

# rgb = cv2.imread(dark_img_path, cv2.COLOR_BGR2RGB) * 50
# # rgb = np.maximum(rgb - 8, 0) * 250
# img = np.uint8(np.minimum(rgb, 255.0))
# cv2.imwrite('dark_img_test.png', img)

# img = np.float32(img)
# r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
# gray = np.uint8(0.299*r + 0.587*g + 0.114*b)
# cv2.imwrite('dark_img_test_gray.png', gray)

# hist = cv2.equalizeHist(gray)
# cv2.imwrite('dark_img_test_gray_hist.png', hist)
# blur_hist = cv2.GaussianBlur(hist, (3, 3), 0)
# cv2.imwrite('dark_img_test_gray_hist_blur.png', blur_hist)

# img = np.float32(img / 255.0)
# r, g, b = img[:,:,0]+1, img[:,:,1]+1, img[:,:,2]+1
# gray = (1.0 - (0.299*r + 0.587*g + 0.114*b)/2.0)
# gray = np.uint8(gray*255.0)
# cv2.imwrite('dark_img_gray_inverse.png', gray)

# blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)
# blur_edge = cv2.Canny(blur_gray, 70, 140)
# cv2.imwrite('dark_img_test_blur_edge.png', blur_edge)


# bright img
# bright_img_path = './dataset/Sony/long/10054_00_10s.png'

# rgb = cv2.imread(bright_img_path, cv2.COLOR_BGR2RGB)
# img = np.uint8(np.minimum(rgb, 255.0))

# cv2.imwrite('birght_img_test.png', img)

# img = np.float32(img)
# r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
# gray = np.uint8(0.299*r + 0.587*g + 0.114*b)
# cv2.imwrite('bright_img_test_gray.png', gray)

# img = np.float32(img / 255.0)
# r, g, b = img[:,:,0]+1, img[:,:,1]+1, img[:,:,2]+1
# gray = (1.0 - (0.299*r + 0.587*g + 0.114*b)/2.0)
# gray = np.uint8(gray*255.0)
# cv2.imwrite('bright_img_test_gray_inverse.png', gray)

# blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)
# blur_edge = cv2.Canny(blur_gray, 70, 140)
# cv2.imwrite('birght_img_test_blur_edge.png', blur_edge)

# # gray img
# gray_img_path = './dataset/Sony/gray/10185_00_250.png'

# gray = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)

# blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)
# blur_edge = cv2.Canny(blur_gray, 70, 140)
# cv2.imwrite('gray_img_test_blur_edge.png', blur_edge)
