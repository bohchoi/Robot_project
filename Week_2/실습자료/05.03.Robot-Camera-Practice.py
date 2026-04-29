# %%
import json
import cv2
import numpy as np
import yaml
import os
from ultralytics import YOLO

# 칼만 필터 기초 클래스 (단순화 버전)
class SimpleKalman:
    def __init__(self):
        # TODO: 칼만 필터 객체를 생성하고 measurementMatrix, transitionMatrix, processNoiseCov 를 설정하세요.
        pass

    def predict(self, coord_x, coord_y):
        # TODO: 측정값으로 칼만 필터를 업데이트하고 보정된 (x, y) 좌표를 반환하세요.
        pass

# %%
# 1. YOLO 모델 로드 (초경량 Nano 모델)
model = YOLO('yolov8n.pt')

# %%
# TODO: camera_info.yaml 에서 캘리브레이션 파라미터를 로드하고
#       camera_matrix, dist_coeffs 를 numpy 배열로 만드세요.
YAML_PATH = "camera_info.yaml"
camera_matrix = None
dist_coeffs = None

# %%

# 미션 1 — 탐지된 객체까지의 거리 추정
REAL_HEIGHT  = 1.7           # 사람의 평균 키 (미터)
def estimate_distance(frame, x1, y1, x2, y2):
    # TODO: 핀홀 카메라 모델로 거리를 계산하고 frame 에 표시하세요.
    #       distance = (REAL_HEIGHT * focal_length) / pixel_height
    pass

# 미션 2 — 특정 클래스 탐지 시 경고 출력
TARGET_LABEL = "cell phone"  # 경고를 발생시킬 클래스 이름
def check_warning(frame, label):
    # TODO: label 이 TARGET_LABEL 이면 frame 에 경고 메시지를 표시하세요.
    pass

# 미션 3 — 탐지 결과를 JSON 파일로 저장
detection_data = []
def save_detection(label, x1, y1, x2, y2, cx, cy):
    # TODO: 탐지 결과를 detection_data 에 추가하세요.
    pass

def save_detection_log(path="detection_log.json"):
    # TODO: detection_data 를 JSON 파일로 저장하세요.
    pass

# %%
# 웹캠 연결 (0번: 내장 웹캠)
cap = cv2.VideoCapture(0)

# 칼만 필터 추적기 생성
tracker = SimpleKalman()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 왜곡 보정 적용 (Undistort)
    undistorted_frame = cv2.undistort(
        frame, camera_matrix, dist_coeffs)

    # YOLOv8 객체 탐지
    results = model(undistorted_frame, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 칼만 필터 추적
            px, py = tracker.predict(cx, cy)

            cls = int(box.cls[0])
            label = model.names[cls]

            # 탐지 박스 (파란색)
            cv2.rectangle(undistorted_frame,
                          (x1, y1), (x2, y2), (255, 0, 0), 2)
            # 칼만 필터 예측 점 (빨간색)
            cv2.circle(undistorted_frame, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(undistorted_frame, f"{label} (Tracked)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

            estimate_distance(undistorted_frame, x1, y1, x2, y2)
            check_warning(undistorted_frame, label)
            save_detection(label, x1, y1, x2, y2, cx, cy)

    cv2.imshow("Robot Vision Pipeline", undistorted_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
save_detection_log()
