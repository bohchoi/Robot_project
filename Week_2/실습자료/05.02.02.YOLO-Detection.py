import cv2
from ultralytics import YOLO


def segmentation_pipeline():
    cap = cv2.VideoCapture(0)
    model = YOLO('yolov8n-seg.pt')  # 세그멘테이션 모델

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        result = results[0]

        annotated = result.plot()  # 마스크 + 박스 오버레이

        # 마스크 정보 출력
        if result.masks is not None:
            classes = result.boxes.cls
            scores = result.boxes.conf
            for i in range(len(classes)):
                label = f"{model.names[int(classes[i])]}: {scores[i]:.2f}"
                print(label)

        cv2.imshow("YOLOv8 Segmentation", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


segmentation_pipeline()
