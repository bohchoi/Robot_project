import cv2       # OpenCV 라이브러리 임포트
import numpy as np  # 행렬 연산을 위한 NumPy

def edge_detection_pipeline():
    # 1. 웹캠 연결 (0번은 내장 웹캠)
    cap = cv2.VideoCapture(0)

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

        # 2. 색공간 변환: BGR -> Gray / BGR -> HSV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 3. 파란색 마스크 생성 및 적용 (배경 노이즈 제거)
        mask   = cv2.inRange(hsv, lower_blue, upper_blue)
        masked = cv2.bitwise_and(gray, gray, mask=mask)

        # 4. 가우시안 블러: 노이즈 제거
        blurred = cv2.GaussianBlur(masked, (5, 5), 0)

        # 5. 트랙바에서 임계값 읽기
        low  = cv2.getTrackbarPos('Low Threshold',  'Controls')
        high = cv2.getTrackbarPos('High Threshold', 'Controls')

        # 6. Canny 에지 검출 (마스킹된 영역에만 적용)
        edges = cv2.Canny(blurred, low, high)

        # 7. 결과 표시
        cv2.imshow('Original Video', frame)
        cv2.imshow('Blue Mask', mask)
        cv2.imshow('Edge Detection Pipeline', edges)

        # 키 입력 처리 (25ms 대기)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('s'):  # 's' 키: 현재 프레임 저장
            cv2.imwrite('result.png', edges)
            print("에지 이미지 저장 완료: result.png")
        if key == ord('q'):  # 'q' 키: 종료
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

# 실행
edge_detection_pipeline()