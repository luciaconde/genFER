import cv2

image_size = 90
image = cv2.imread("data/classes/sad/frame_det_00_000342.bmp")
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
image = image[0:image_size/3, 0:image_size]
cv2.imshow("cropped", image)
cv2.waitKey(0)
