import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

def approximate_segmentation_to_four_points(segmentation, max_iterations=100):
    segmentation = segmentation.reshape((-1, 1, 2))
    
    epsilon = 1.0
    iterations = 0

    while iterations < max_iterations:
        approximated_polygon = cv2.approxPolyDP(segmentation, epsilon, True)

        if len(approximated_polygon) == 4:
            return approximated_polygon

        if len(approximated_polygon) > 4:
            epsilon += 1.0
        else:
            epsilon -= 0.1

        iterations += 1

    return approximated_polygon

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="jgBnSYCd5GddJtCFBbQI"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = CLIENT.infer(frame, model_id="chessboard-segmentation/1")
    
    try:
        points_dict = result["predictions"][0]["points"]
        polygon = [(item["x"], item["y"]) for item in points_dict]
        polygon = np.array([polygon], dtype=np.int32)
        #polygon = approximate_segmentation_to_four_points(polygon)
        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
    except:
        pass

    cv2.imshow("test", frame)
    cv2.waitKey(1)