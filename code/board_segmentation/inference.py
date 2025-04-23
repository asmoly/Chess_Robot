import cv2
from train import *

PATH_TO_MODEL = "logs/run_20250325_201151/boardsegmentnet_50.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

board_segment_transformer_model = load_model(PATH_TO_MODEL).to(device)

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

    height, width, _ = frame.shape

    frame = frame[:, :height, :]
    frame = cv2.resize(frame, (640, 640))

    with torch.no_grad():
        model_input = torch.tensor(frame).permute(2, 0, 1)/255.0
        model_input = model_input[None, :, :, :].to(device)

        output = board_segment_transformer_model(model_input)
        output = output.cpu().numpy()[0]*640
        output = output.reshape(4, 2)

        frame = draw_annotations_on_image(frame, output, normalized=True)

    # Display the processed frame
    cv2.imshow("main", frame)
    cv2.waitKey(1)