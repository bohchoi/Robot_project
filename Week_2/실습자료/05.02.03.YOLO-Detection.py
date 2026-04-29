import cv2
from ultralytics import YOLO


def pose_pipeline():
    cap = cv2.VideoCapture(0)
    model = YOLO('yolov8n-pose.pt')  # 포즈 추정 모델

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        result = results[0]

        annotated = result.plot()  # 스켈레톤 + 박스 오버레이

        # 키포인트 정보 출력 (17개 관절)
        if result.keypoints is not None:
            keypoints = result.keypoints.xy   # (N, 17, 2) - x, y 좌표
            scores = result.keypoints.conf    # (N, 17) - 각 관절 신뢰도

            for person_idx in range(len(keypoints)):
                print(f"Person {person_idx + 1}:")
                for kp_idx, (kp, conf) in enumerate(zip(keypoints[person_idx], scores[person_idx])):
                    kp_name = KEYPOINT_NAMES[kp_idx]
                    print(f"  {kp_name}: ({kp[0]:.1f}, {kp[1]:.1f}), conf={conf:.2f}")

        cv2.imshow("YOLOv8 Pose", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# COCO 17개 관절 이름
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


pose_pipeline()
