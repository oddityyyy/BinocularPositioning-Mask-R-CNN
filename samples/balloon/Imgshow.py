import cv2
img=cv2.imread(r"D:\maskrcnn\Mask_RCNN-master\apple\val\3-ColorImg-21.jpg")
cv2.namedWindow("Image")
cv2.imshow("Image", img)
cv2.waitKey (0)
cv2.destroyAllWindows()