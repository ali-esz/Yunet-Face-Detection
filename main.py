import os
from datetime import datetime

import cv2
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "face_detection_yunet_2026may.onnx"
)

CONFIDENCE_THRESHOLD = 0.75
NMS_THRESHOLD = 0.30
TOP_K = 5000

CAMERA_INDEX = 0
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600

# Requested display/window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Height of the black bars
TOP_BAR_HEIGHT = 60
BOTTOM_BAR_HEIGHT = 80
VIDEO_HEIGHT = WINDOW_HEIGHT - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT

# Exact requested colors, converted from RGB hex to OpenCV BGR
BLACK = (0, 0, 0)
FACE_GREEN = (21, 207, 22)  # #16CF15
MODE_RED = (63, 18, 214)  # #D6123F
GUIDE_BLUE = (235, 34, 41)  # #2922EB


def put_text(img, text, org, font_scale, color, thickness=2):
    cv2.putText(
        img=img,
        text=text,
        org=org,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def draw_top_bar(canvas, face_count, mode):
    canvas[0:TOP_BAR_HEIGHT, :] = BLACK

    # Left: detected Face count
    put_text(canvas, "Face", (20, 40), 0.9, FACE_GREEN, 2)
    put_text(canvas, f": {face_count}", (105, 40), 0.9, FACE_GREEN, 2)

    # Right: current mode
    if mode == 0:
        mode_label = "Default Mode = Normal"
    elif mode == 1:
        mode_label = "Mode = Sobel"
    else:
        mode_label = "Mode = Canny"

    text_size, _ = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = WINDOW_WIDTH - text_size[0] - 20

    put_text(canvas, mode_label, (x, 40), 0.8, MODE_RED, 2)


def draw_bottom_bar(canvas):
    y0 = WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT
    canvas[y0:WINDOW_HEIGHT, :] = BLACK

    guide = (
        "Controls: Q = Quit   N = Normal   S = Sobel   "
        "C = Canny   W = Save Screenshot"
    )

    text_size, _ = cv2.getTextSize(guide, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    x = max(15, (WINDOW_WIDTH - text_size[0]) // 2)
    y = y0 + 48

    put_text(canvas, guide, (x, y), 0.62, GUIDE_BLUE, 2)


def main():

    if not os.path.exists(MODEL_PATH):
        print("ERROR: YuNet model was not found!")
        print()
        print("Expected location:")
        print(MODEL_PATH)
        return

    if not hasattr(cv2, "FaceDetectorYN"):
        print("ERROR: FaceDetectorYN is not available.")
        print("Please install a compatible OpenCV version.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read camera frame!")
        cap.release()
        return

    height, width = frame.shape[:2]
    print("Camera resolution:", width, "x", height)

    detector = cv2.FaceDetectorYN.create(
        model=MODEL_PATH,
        config="",
        input_size=(width, height),
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        top_k=TOP_K,
    )

    # 0 = Normal, 1 = Sobel, 2 = Canny
    mode = 0

    window_name = "YuNet - Face Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read camera frame!")
            break

        frame = cv2.flip(frame, 1)

        # Face detection is done on the normal camera frame.
        _, faces = detector.detect(frame)
        face_count = 0 if faces is None else len(faces)

        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                confidence = float(face[-1])

                cv2.rectangle(
                    img=frame,
                    pt1=(x, y),
                    pt2=(x + w, y + h),
                    color=(0, 255, 0),
                    thickness=2,
                )

                cv2.putText(
                    img=frame,
                    text=f"Face: {confidence:.2f}",
                    org=(x, max(y - 10, 20)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.6,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

                landmarks = face[4:14].reshape(5, 2).astype(int)

                colors = [
                    (255, 0, 0),
                    (0, 0, 255),
                    (0, 255, 0),
                    (255, 0, 255),
                    (0, 255, 255),
                ]

                for point, color in zip(landmarks, colors):
                    px, py = point
                    cv2.circle(
                        img=frame,
                        center=(px, py),
                        radius=3,
                        color=color,
                        thickness=-1,
                    )

        # Apply the selected image-processing mode.
        processed_frame = frame.copy()

        if mode == 1:
            sobel_x = cv2.Sobel(processed_frame, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(processed_frame, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            processed_frame = cv2.convertScaleAbs(sobel_magnitude)

        elif mode == 2:
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray ,3)
            median_intensity = np.median(blurred)

            lower = int(max(0, 0.7 * median_intensity))
            upper = int(min(255, 1.3 * median_intensity))

            canny_edges = cv2.Canny(
                blurred,
                threshold1=lower,
                threshold2=upper,
            )

            processed_frame = cv2.bitwise_and(
                processed_frame,
                processed_frame,
                mask=canny_edges,
            )

        # Fit the camera image into the 1000x800 layout.
        video = cv2.resize(
            processed_frame,
            (WINDOW_WIDTH, VIDEO_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        # Complete 1000x800 image: black top bar + video + black bottom bar.
        canvas = np.zeros(
            (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
            dtype=np.uint8,
        )

        canvas[TOP_BAR_HEIGHT : TOP_BAR_HEIGHT + VIDEO_HEIGHT, :] = video

        draw_top_bar(canvas, face_count, mode)
        draw_bottom_bar(canvas)

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("n"):
            mode = 0
            print("Mode = Normal")

        elif key == ord("s"):
            mode = 1
            print("Mode = Sobel")

        elif key == ord("c"):
            mode = 2
            print("Mode = Canny")

        elif key == ord("w"):
            # Save the complete displayed 1000x800 image in the project folder.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            screenshot_path = os.path.join(
                os.path.dirname(__file__),
                f"screenshot_{timestamp}.png",
            )

            if cv2.imwrite(screenshot_path, canvas):
                print(f"Screenshot saved: {screenshot_path}")
            else:
                print("ERROR: Could not save screenshot!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
