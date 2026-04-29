import cv2       # OpenCV 라이브러리 임포트
import numpy as np  # 행렬 연산을 위한 NumPy
from ultralytics import YOLO


def edge_detection_pipeline():
    # 1. 웹캠 연결 (0번은 내장 웹캠)
    cap = cv2.VideoCapture(0)
    model = YOLO('yolov8n.pt')

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 트랙바 창 생성
    cv2.namedWindow('Controls')
    cv2.createTrackbar('Low Threshold',  'Controls', 100, 255, lambda _: None)
    cv2.createTrackbar('High Threshold', 'Controls', 200, 255, lambda _: None)

    # 파란색 HSV 범위
    lower_blue = np.array([100, 150,  50])
    upper_blue = np.array([140, 255, 255])

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, stream=True)

        for r in results:
            annotated_frame = r.plot()  # OpenCV 호환 배열 반환
            cv2.imshow("YOLOv8 Real-time Inference", annotated_frame)

        # 7. 결과 표시
        cv2.imshow('Original Video', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키로 종료
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

# 실행
edge_detection_pipeline()