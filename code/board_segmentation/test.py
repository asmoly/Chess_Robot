import cv2
from data import *

image = cv2.imread("data/train/326_jpg.rf.826c91ea625a7029d393dc2021399f18.jpg")

annotations = np.array([[50, 50], [200, 50], [200, 200], [50, 200]])

image = draw_annotations_on_image(image, annotations, normalized=True)
cv2.imshow("main", image)
cv2.waitKey(0)
