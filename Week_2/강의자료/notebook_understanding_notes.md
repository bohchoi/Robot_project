# OpenCV Calibration, YOLO, Kalman Filter 이해 노트

노트북을 "완벽히 이해"하려면 외울 게 아니라 **왜 이 수식과 코드가 필요한지**를 연결해야 한다. 큰 흐름은 두 노트북이다.

```text
05.01.OpenCV-Calibration.ipynb
→ 카메라 수학 모델 + 캘리브레이션

05.02.YOLO-Kalman.ipynb
→ YOLO 객체 탐지 + 칼만 필터
```

## 1. 카메라 모델 핵심

카메라는 3D 세계를 2D 이미지로 줄인다.

현실의 한 점:

```text
(X, Y, Z)
```

이미지의 한 점:

```text
(u, v)
```

여기서 `Z`는 카메라로부터의 깊이, 즉 거리다.

핀홀 카메라 모델의 핵심 수식은 이것이다.

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

뜻은 다음과 같다.

```text
X / Z = 물체가 카메라 정면 기준으로 얼마나 오른쪽에 있는가
Y / Z = 물체가 카메라 정면 기준으로 얼마나 아래/위에 있는가
fx, fy = 카메라 확대 비율
cx, cy = 이미지 중심점
```

코드에서는 이렇게 나온다.

```python
u = fx * (X / Z) + cx
v = fy * (Y / Z) + cy
```

예를 들어:

```python
X, Y, Z = 0.3, 0.1, 2.0
```

이 뜻은:

```text
카메라 기준 오른쪽 0.3m
위/아래 방향 0.1m
앞쪽 2.0m
```

이다.

`Z`가 커지면:

```text
X / Z 값이 작아짐
→ 이미지에서 중심에 가까워짐
→ 멀리 있는 물체는 작고 중심 쪽에 모여 보임
```

## 2. 카메라 행렬 K

위 수식을 행렬로 묶은 것이 `K`다.

```text
K = [ fx   0   cx ]
    [  0  fy   cy ]
    [  0   0    1 ]
```

코드:

```python
K = np.array([
    [fx,  0,  cx],
    [ 0, fy,  cy],
    [ 0,  0,   1]
], dtype=np.float64)
```

초심자식 해석:

```text
fx, fy: 카메라가 얼마나 확대해서 보는가
cx, cy: 렌즈 중심이 이미지 어디에 있는가
```

보통 이미지가 `640 x 480`이면 이상적인 중심은:

```python
cx = 640 / 2  # 320
cy = 480 / 2  # 240
```

하지만 실제 카메라는 렌즈나 센서가 완벽히 중앙에 있지 않아서 `cx`, `cy`가 조금 다를 수 있다.

## 3. 픽셀에서 다시 실제 방향 구하기

앞의 식을 거꾸로 풀면:

```text
x_norm = (u - cx) / fx
y_norm = (v - cy) / fy
```

코드:

```python
x_norm = (u - cx) / fx
y_norm = (v - cy) / fy
```

이건 픽셀 좌표를 카메라 중심 기준 방향으로 바꾸는 것이다.

예를 들어 어떤 물체가 이미지 오른쪽에 있으면 `u > cx`다.

```text
u - cx > 0
→ x_norm > 0
→ 카메라 기준 오른쪽에 있음
```

하지만 여기까지만 하면 방향만 안다. 실제 거리까지 알려면 `Z`가 필요하다.

```python
X_back = x_norm * Z
Y_back = y_norm * Z
```

즉:

```text
픽셀 좌표 + 깊이 Z
→ 실제 3D 위치 복원 가능
```

중요한 점:

```text
카메라 한 대만으로는 정확한 깊이 Z를 직접 알 수 없다.
```

그래서 거리 추정에서는 물체의 실제 크기 같은 추가 정보가 필요하다.

## 4. 거리 추정 수식

노트북에 나오는 거리 추정 수식:

```text
Distance = 실제 높이 × 초점거리 / 이미지에서 보이는 픽셀 높이
```

코드:

```python
distance = (REAL_HEIGHT * focal_length) / pixel_height
```

예를 들어 사람 키를 `1.7m`라고 가정한다.

```python
REAL_HEIGHT = 1.7
focal_length = camera_matrix[1][1]  # fy
pixel_height = y2 - y1
distance = (REAL_HEIGHT * focal_length) / pixel_height
```

왜 이렇게 되냐면, 가까운 물체는 이미지에서 크게 보이고, 먼 물체는 작게 보이기 때문이다.

```text
pixel_height 큼  → distance 작음  → 가까움
pixel_height 작음 → distance 큼  → 멂
```

단, 이건 가정이 많다.

```text
사람이 서 있어야 함
전체 몸이 보여야 함
REAL_HEIGHT가 실제와 비슷해야 함
카메라 캘리브레이션이 잘 되어 있어야 함
```

## 5. 왜 캘리브레이션을 하나

지금까지의 수식은 "이상적인 카메라" 기준이다.

하지만 실제 카메라는 렌즈 때문에 휘어진다.

```text
직선이 곡선처럼 보임
가장자리 물체 위치가 틀어짐
픽셀 좌표로 거리/방향 계산 시 오차 발생
```

그래서 체커보드를 찍고 카메라의 실제 특성을 구한다.

캘리브레이션에서 구하는 것:

```text
mtx  = camera matrix K
dist = distortion coefficients
```

코드:

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
```

각각 의미:

```text
ret     = 재투영 오차, 작을수록 좋음
mtx     = 카메라 행렬 K
dist    = 왜곡 계수
rvecs   = 각 사진에서 체커보드가 회전한 정도
tvecs   = 각 사진에서 체커보드가 이동한 정도
```

## 6. objpoints와 imgpoints

이 부분이 처음에 가장 헷갈린다.

```python
objpoints = []
imgpoints = []
```

`objpoints`는 실제 세계의 점이다.

```text
체커보드의 실제 3D 좌표
예: (0,0,0), (1,0,0), (2,0,0) ...
```

체커보드는 평평하니까 `Z=0`이다.

코드:

```python
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE
```

이 코드는 체커보드 내부 교차점들의 실제 위치를 만든다.

예를 들어 `CHECKERBOARD = (9, 6)`이면 점은 총:

```text
9 × 6 = 54개
```

이다.

`imgpoints`는 이미지에서 검출된 픽셀 좌표다.

```python
ret, corners = cv2.findChessboardCorners(gray, pattern, None)
```

이게 이미지 속 체커보드 코너를 찾는다.

```python
corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
```

이건 찾은 코너를 더 정밀하게 다듬는다.

즉:

```text
objpoints: 실제 체커보드 점 위치
imgpoints: 이미지에서 보인 체커보드 점 위치
```

OpenCV는 이 둘을 비교해서:

```text
"이 실제 점들이 이미지에서 이렇게 보였구나.
그렇다면 이 카메라의 fx, fy, cx, cy, 왜곡계수는 이 정도겠네."
```

라고 계산한다.

## 7. 재투영 오차

재투영 오차는 캘리브레이션 품질 점수다.

절차는 이렇다.

```text
1. 캘리브레이션으로 K, dist를 구함
2. 실제 3D 체커보드 점을 다시 이미지에 투영해봄
3. 실제 검출된 코너 위치와 비교함
4. 차이가 작으면 좋은 캘리브레이션
```

코드:

```python
proj, _ = cv2.projectPoints(op, rvecs[i], tvecs[i], mtx, dist)
err = cv2.norm(ip, proj, cv2.NORM_L2) / len(proj)
```

해석:

```text
proj = 모델이 예측한 코너 위치
ip   = 실제 이미지에서 검출한 코너 위치
err  = 둘 사이의 평균 차이
```

기준:

```text
0.5 px 미만: 매우 좋음
1.0 px 이하: 보통 사용 가능
1.0 px 초과: 사진을 다시 찍는 게 좋음
```

## 8. 왜곡 보정

캘리브레이션 결과를 저장한다.

```python
data = {
    'camera_matrix': mtx.tolist(),
    'dist_coeff': dist.tolist()
}
```

그리고 `camera_info.yaml`에 저장한다.

나중에 다시 불러온다.

```python
with open(filepath, 'r') as f:
    data = yaml.safe_load(f)

mtx  = np.array(data['camera_matrix'], np.float64)
dist = np.array(data['dist_coeff'], np.float64)
```

왜곡 보정은 이 코드다.

```python
dst = cv2.undistort(img, mtx, dist, None, mtx)
```

뜻:

```text
img  = 왜곡된 원본 이미지
mtx  = 카메라 행렬
dist = 왜곡 계수
dst  = 보정된 이미지
```

## 9. YOLO 결과 이해

YOLO는 이미지에서 객체를 찾고 박스를 준다.

```python
results = model('bus.jpg')
```

결과에서 박스 정보를 꺼낸다.

```python
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
```

뜻:

```text
x1, y1 = 박스 왼쪽 위
x2, y2 = 박스 오른쪽 아래
conf   = 신뢰도
cls_id = 클래스 번호
label  = 클래스 이름
```

중심점 계산:

```python
cx = (x1 + x2) // 2
cy = (y1 + y2) // 2
```

로봇에서는 이 중심점이 중요하다.

```text
cx < 화면 중앙 → 물체가 왼쪽
cx > 화면 중앙 → 물체가 오른쪽
cy가 크다     → 물체가 아래쪽
```

예를 들어 추적 로봇이라면:

```text
물체 중심이 화면 왼쪽 → 로봇 왼쪽 회전
물체 중심이 화면 오른쪽 → 로봇 오른쪽 회전
물체가 너무 가까움 → 정지
```

## 10. 칼만 필터 핵심

YOLO 박스는 흔들린다. 그래서 칼만 필터로 부드럽게 만든다.

칼만 필터의 핵심은:

```text
예측값 + 측정값을 적절히 섞는다
```

상태 벡터:

```text
x = [position, velocity]
```

2D 추적에서는 보통:

```text
state = [x, y, dx, dy]
```

즉:

```text
x, y   = 현재 위치
dx, dy = 속도
```

OpenCV 코드:

```python
self.kf = cv2.KalmanFilter(4, 2)
```

뜻:

```text
상태 변수 4개: x, y, dx, dy
측정 변수 2개: x, y
```

측정 행렬:

```python
self.kf.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], np.float32)
```

뜻:

```text
센서는 x, y만 측정한다.
dx, dy는 직접 측정하지 않는다.
```

전이 행렬:

```python
self.kf.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)
```

이건 이런 뜻이다.

```text
다음 x = 현재 x + dx
다음 y = 현재 y + dy
다음 dx = 현재 dx
다음 dy = 현재 dy
```

즉 등속 운동 가정이다.

## 11. 칼만 필터 predict/correct

```python
self.kf.predict()
corrected = self.kf.correct(measured)
```

뜻:

```text
predict  : 이전 속도를 보고 다음 위치 예상
correct  : YOLO가 측정한 실제 중심점으로 보정
```

전체 함수:

```python
def predict(self, coord_x, coord_y):
    measured = np.array([
        [np.float32(coord_x)],
        [np.float32(coord_y)]
    ])

    self.kf.predict()
    corrected = self.kf.correct(measured)

    return int(corrected[0]), int(corrected[1])
```

YOLO 중심점이 조금 흔들려도 칼만 필터 결과는 더 부드럽게 움직인다.

## 12. 최종 파이프라인

최종 실습의 전체 구조는 이것이다.

```python
cap = cv2.VideoCapture(0)
tracker = SimpleKalman()

while cap.isOpened():
    success, frame = cap.read()

    undistorted_frame = cv2.undistort(
        frame, camera_matrix, dist_coeffs
    )

    results = model(undistorted_frame, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            px, py = tracker.predict(cx, cy)

            cv2.rectangle(...)
            cv2.circle(...)
```

한 줄씩 해석하면:

```text
1. 웹캠에서 프레임 읽기
2. 카메라 왜곡 보정
3. YOLO로 객체 탐지
4. 박스 중심점 계산
5. 칼만 필터로 중심점 안정화
6. 화면에 박스와 추적점 표시
```

## 핵심 그림

```text
체커보드
→ 카메라 보정값 구함
→ camera_info.yaml 저장

웹캠 영상
→ camera_info.yaml로 왜곡 보정
→ YOLO로 물체 찾기
→ 중심점 계산
→ 칼만 필터로 흔들림 제거
→ 로봇 판단에 사용
```

처음에는 이 순서만 정확히 이해하면 된다.

그리고 수식은 딱 세 개만 먼저 잡으면 된다.

```text
1. 3D → 픽셀
u = fx * X / Z + cx
v = fy * Y / Z + cy

2. 픽셀 → 방향
x = (u - cx) / fx
y = (v - cy) / fy

3. 크기로 거리 추정
Distance = 실제 높이 × 초점거리 / 픽셀 높이
```

이 세 개가 카메라 파트의 뼈대고, YOLO와 칼만 필터는 그 위에 올라가는 응용이다.

---

# 05.02 YOLOv8 추론 & Kalman Filter 강의자료 정리

이 강의자료의 핵심은 **로봇이 카메라 영상에서 물체를 찾고, 그 위치를 안정적으로 추적하게 만드는 방법**이다.

전체 흐름은 다음과 같다.

```text
카메라 영상
→ YOLOv8 객체 탐지
→ 바운딩 박스와 클래스 추출
→ 중심점 좌표 계산
→ 칼만 필터로 흔들림 제거
→ 로봇 제어 또는 경고 로직에 사용
```

## 13. 객체 탐지란 무엇인가

이미지 인식에는 크게 두 가지가 있다.

```text
이미지 분류 Classification
→ 사진 안에 무엇이 있는지만 말함
→ 예: "사람"

객체 탐지 Object Detection
→ 무엇이 어디에 있는지까지 말함
→ 예: "사람이 이미지 왼쪽 아래에 있음"
```

로봇에는 분류보다 탐지가 훨씬 중요하다.

```text
"앞에 사람이 있다"
```

만으로는 부족하다. 로봇은 다음 정보가 필요하다.

```text
사람이 왼쪽에 있는가?
오른쪽에 있는가?
얼마나 가까운가?
피해야 하는가?
따라가야 하는가?
```

그래서 YOLO 결과에서 `label`만 보는 게 아니라 `box.xyxy`, `box.xywh`, `conf` 같은 값을 같이 본다.

## 14. 1-Stage와 2-Stage 탐지 모델

객체 탐지 모델은 크게 두 계열로 나뉜다.

```text
1-Stage
→ 위치와 종류를 한 번에 예측
→ 빠름
→ YOLO, SSD, RetinaNet

2-Stage
→ 먼저 물체 후보 영역을 찾고, 그다음 분류
→ 느리지만 정밀함
→ Faster R-CNN, Mask R-CNN
```

로봇에서는 보통 1-Stage가 유리하다.

이유:

```text
로봇은 실시간으로 움직임
판단이 늦으면 충돌하거나 반응이 어색해짐
라즈베리파이, 젯슨 같은 엣지 장치는 연산 성능이 제한됨
```

그래서 YOLO처럼 빠른 모델을 많이 쓴다.

## 15. YOLO의 핵심 아이디어

YOLO는 `You Only Look Once`의 약자다.

뜻:

```text
이미지를 여러 번 훑지 않고,
한 번의 신경망 계산으로
물체 위치와 종류를 동시에 예측한다.
```

기존 방식은 이미지 안에서 물체가 있을 만한 후보 영역을 많이 만들고 하나씩 검사했다.

YOLO는 이미지 전체를 한 번 보고:

```text
어디에 박스를 그릴지
그 박스 안 물체가 무엇인지
얼마나 확신하는지
```

를 한 번에 낸다.

그래서 로봇, 자율주행, 실시간 카메라 시스템에 잘 맞는다.

## 16. YOLOv8 모델 크기 선택

YOLOv8에는 여러 크기의 모델이 있다.

```text
yolov8n.pt → nano, 가장 빠르고 가벼움
yolov8s.pt → small, 속도와 정확도 균형
yolov8m.pt → medium, 더 정확하지만 무거움
yolov8l.pt / yolov8x.pt → 크고 정확하지만 느림
```

초심자 실습이나 웹캠 실시간 탐지에서는 보통 이것을 쓴다.

```python
model = YOLO('yolov8n.pt')
```

`n` 모델을 쓰는 이유:

```text
가장 가벼움
CPU에서도 상대적으로 잘 돌아감
실시간 웹캠 실습에 적합
```

정확도는 큰 모델보다 낮을 수 있지만, 로봇에서는 **속도도 정확도만큼 중요**하다.

## 17. YOLO 설치와 기본 사용

강의자료의 설치 명령:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics numpy==1.26.4 opencv-python==4.10.0.84
```

기본 코드:

```python
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
results = model.predict(source='robot_view.jpg', save=True)
```

웹캠으로 바로 실행할 수도 있다.

```python
model.predict(source=0, show=True)
```

여기서 `source`에는 여러 종류가 들어갈 수 있다.

```text
source='image.jpg'   → 이미지 파일
source='video.mp4'   → 동영상 파일
source=0             → 기본 웹캠
source=1             → 두 번째 카메라
```

## 18. YOLO 결과 객체 이해

YOLO를 실행하면 `results`가 나온다.

```python
results = model('robot_view.jpg')
```

보통 이렇게 순회한다.

```python
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
```

각 값의 의미:

```text
box.xyxy
→ [x1, y1, x2, y2]
→ 왼쪽 위 점과 오른쪽 아래 점

box.xywh
→ [center_x, center_y, width, height]
→ 중심점, 너비, 높이

box.conf
→ 신뢰도
→ 0.0 ~ 1.0 사이

box.cls
→ 클래스 번호
→ COCO 기준 0은 person, 2는 car 등

model.names[cls_id]
→ 클래스 번호를 사람이 읽는 이름으로 변환
```

초심자에게 가장 중요한 값은 세 가지다.

```text
label: 무엇인가?
conf: 얼마나 확실한가?
bbox: 어디에 있는가?
```

## 19. 바운딩 박스 좌표

YOLO의 `xyxy` 형식:

```text
(x1, y1) = 박스 왼쪽 위
(x2, y2) = 박스 오른쪽 아래
```

그림으로 보면:

```text
(x1, y1) ┌─────────────┐
         │             │
         │   object    │
         │             │
         └─────────────┘ (x2, y2)
```

중심점은 이렇게 계산한다.

```python
cx = (x1 + x2) // 2
cy = (y1 + y2) // 2
```

박스의 크기는 이렇게 계산한다.

```python
width = x2 - x1
height = y2 - y1
```

로봇 제어에서는 이런 식으로 쓴다.

```text
cx가 화면 중앙보다 왼쪽 → 로봇을 왼쪽으로 회전
cx가 화면 중앙보다 오른쪽 → 로봇을 오른쪽으로 회전
height가 큼 → 가까움
height가 작음 → 멂
```

## 20. 신뢰도 Confidence

`conf`는 YOLO가 자기 예측을 얼마나 믿는지 나타낸다.

```python
conf = box.conf[0]
```

예:

```text
conf = 0.92 → 꽤 확실함
conf = 0.31 → 애매함
```

실전에서는 낮은 신뢰도의 결과를 버린다.

```python
results = model.predict(frame, conf=0.5)
```

뜻:

```text
신뢰도 0.5 미만은 무시
```

너무 낮게 잡으면:

```text
엉뚱한 물체도 탐지됨
```

너무 높게 잡으면:

```text
진짜 물체도 놓칠 수 있음
```

보통 처음에는 `0.5`부터 시작해서 상황에 맞게 조절한다.

## 21. 특정 클래스만 탐지하기

COCO 데이터셋 기준으로 YOLO는 80개 클래스를 기본으로 안다.

예:

```text
0 = person
1 = bicycle
2 = car
...
```

사람과 자동차만 탐지하고 싶으면:

```python
results = model.predict(frame, classes=[0, 2])
```

사람만 탐지:

```python
results = model.predict(frame, classes=[0])
```

이렇게 하면 필요 없는 물체를 검사하지 않으므로 결과가 더 깔끔해지고 속도에도 도움이 될 수 있다.

## 22. NMS란 무엇인가

NMS는 `Non-Maximum Suppression`이다.

문제 상황:

```text
한 물체에 박스가 여러 개 생김
```

NMS는 겹치는 박스들 중에서 가장 신뢰도가 높은 박스만 남긴다.

```text
같은 사람을 가리키는 박스 3개
→ conf가 가장 높은 박스 1개만 남김
```

이때 박스가 얼마나 겹치는지 보는 기준이 `IoU`다.

```text
IoU = 두 박스가 겹친 면적 / 두 박스를 합친 면적
```

초심자 단계에서는 이렇게 이해하면 충분하다.

```text
NMS는 중복 박스를 정리하는 후처리다.
```

## 23. 실시간 웹캠 추론 코드 구조

강의자료의 기본 구조:

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, stream=True)

    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("YOLOv8 Real-time Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

중요한 줄:

```python
cap = cv2.VideoCapture(0)
```

기본 웹캠을 연다.

```python
success, frame = cap.read()
```

웹캠에서 프레임 한 장을 읽는다.

```python
results = model.predict(frame, stream=True)
```

현재 프레임에서 객체를 탐지한다.

```python
annotated_frame = r.plot()
```

YOLO가 박스, 라벨, 신뢰도를 그린 이미지를 만들어준다.

```python
cv2.imshow(...)
```

화면에 보여준다.

## 24. `stream=True`를 쓰는 이유

실시간 영상에서는 프레임이 계속 들어온다.

처리가 느리면 문제가 생긴다.

```text
카메라는 초당 30장 입력
YOLO는 초당 5장 처리
→ 처리 대기열이 쌓임
→ 화면이 과거 장면을 늦게 보여줌
→ 로봇 반응이 늦어짐
```

`stream=True`는 결과를 제너레이터 방식으로 처리해서 메모리를 아끼고 실시간 처리에 유리하다.

```python
results = model.predict(frame, stream=True)
```

초심자식으로 말하면:

```text
한꺼번에 다 쌓아두지 말고,
나오는 대로 바로 처리하자.
```

## 25. FPS와 최적화

로봇에서는 FPS가 중요하다.

```text
FPS = Frames Per Second
→ 1초에 몇 장의 영상을 처리하는가
```

대략 기준:

```text
1~5 FPS    → 너무 느림
15 FPS 이상 → 어느 정도 실시간
30 FPS 이상 → 부드러운 실시간
```

속도를 높이는 방법:

```text
작은 모델 사용: yolov8n
이미지 크기 줄이기: imgsz=320
필요한 클래스만 탐지: classes=[0]
GPU 사용
ONNX, OpenVINO, TensorRT로 변환
```

이미지 크기 줄이기:

```python
results = model.predict(frame, imgsz=320)
```

주의:

```text
imgsz를 줄이면 속도는 빨라지지만
작은 물체 탐지 성능은 떨어질 수 있다.
```

## 26. ROS2와 연결할 때 필요한 데이터

로봇 시스템에서는 YOLO 결과를 다른 노드로 넘겨야 한다.

추천 데이터 구조:

```text
Header
→ 시간 정보 timestamp
→ 프레임 ID frame_id

Detections
→ class_id
→ confidence
→ bbox = [x_center, y_center, width, height]
```

ROS2 카메라 이미지는 보통 `sensor_msgs/Image`로 들어오고, OpenCV에서 쓰려면 NumPy 배열로 바꿔야 한다.

이때 자주 쓰는 것이:

```text
cv_bridge
```

흐름:

```text
ROS2 Image 메시지
→ cv_bridge
→ OpenCV frame
→ YOLO 처리
→ Detection 메시지 publish
```

## 27. 칼만 필터가 필요한 이유

YOLO 결과는 매 프레임 조금씩 흔들린다.

예:

```text
실제 물체는 가만히 있음
YOLO 중심점:
321, 318, 323, 320, 324 ...
```

이 값을 그대로 로봇 제어에 넣으면 로봇이 계속 미세하게 흔들릴 수 있다.

칼만 필터는:

```text
센서 측정값 + 운동 예측값
→ 둘을 적절히 섞어서 더 안정적인 추정값 생성
```

에 쓰인다.

대표 활용:

```text
YOLO 중심점 안정화
GPS + IMU 위치 추정
속도 추정
물체가 잠깐 가려졌을 때 위치 예측
```

## 28. 칼만 필터의 두 단계

칼만 필터는 두 단계를 계속 반복한다.

```text
Predict
→ 이전 상태와 운동 모델로 다음 위치 예측

Update / Correct
→ 새 센서 측정값을 보고 예측을 보정
```

간단한 비유:

```text
Predict:
"이 물체가 오른쪽으로 움직이고 있었으니 다음에도 조금 오른쪽에 있겠지."

Update:
"YOLO가 실제로 여기 있다고 하네. 그럼 내 예측을 조금 수정하자."
```

## 29. 상태 State

칼만 필터가 추적하는 대상이 `상태`다.

1차원에서는:

```text
x = [p, v]
```

뜻:

```text
p = 위치
v = 속도
```

2차원 YOLO 중심점 추적에서는:

```text
x = [cx, cy, vx, vy]
```

뜻:

```text
cx, cy = 현재 중심점 위치
vx, vy = 중심점이 움직이는 속도
```

위치만 있으면 다음 위치를 예측하기 어렵다. 속도가 있으면 다음 위치를 예상할 수 있다.

## 30. 공분산 P

`P`는 내 추정이 얼마나 불확실한지 나타낸다.

```text
P가 큼
→ 내 추정이 불확실함

P가 작음
→ 내 추정을 꽤 믿음
```

초기에는 보통 잘 모르기 때문에 크게 둔다.

```python
P_0 = [[25, 0],
       [0, 4]]
```

뜻:

```text
위치는 꽤 불확실함
속도도 어느 정도 불확실함
```

칼만 필터는 시간이 지나며 측정값을 보면서 점점 안정된다.

## 31. Q와 R

칼만 필터에서 매우 중요한 두 행렬이다.

```text
Q = 프로세스 노이즈
→ 내 운동 모델이 얼마나 틀릴 수 있는가

R = 센서 노이즈
→ 센서 측정값이 얼마나 부정확한가
```

초심자식 해석:

```text
Q가 큼
→ "내 예측 모델을 별로 못 믿겠다."
→ 측정값 쪽으로 더 빨리 따라감

R이 큼
→ "센서를 별로 못 믿겠다."
→ 예측값을 더 믿음
```

YOLO 중심점 추적에서는:

```text
YOLO 박스가 많이 흔들림 → R을 크게
물체 움직임이 불규칙함 → Q를 크게
```

## 32. Predict 수식

Predict 단계:

```text
x_k^- = A x_{k-1} + B u_k
P_k^- = A P_{k-1} A^T + Q
```

뜻:

```text
x_k^- = 이번 스텝의 예측 상태
P_k^- = 이번 스텝의 예측 불확실성
A = 상태 전이 행렬
B u_k = 제어 입력의 영향
Q = 모델이 틀릴 가능성
```

등속 운동이면:

```text
다음 위치 = 현재 위치 + 속도 × 시간
다음 속도 = 현재 속도
```

행렬로는:

```text
A = [1  dt]
    [0   1]
```

YOLO 2D 추적에서 `dt=1 프레임`이라고 단순화하면:

```text
다음 x = 현재 x + vx
다음 y = 현재 y + vy
다음 vx = 현재 vx
다음 vy = 현재 vy
```

OpenCV 코드의 전이 행렬:

```python
self.kf.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)
```

## 33. Update 수식

Update 단계는 센서 측정값으로 예측값을 보정한다.

센서 행렬 `C`:

```text
C = 센서가 상태 중 무엇을 볼 수 있는지 나타냄
```

1차원에서 위치만 측정하면:

```text
C = [1 0]
```

뜻:

```text
상태 [위치, 속도] 중 위치만 측정 가능
속도는 직접 측정하지 못함
```

YOLO 2D 중심점에서는:

```python
measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], np.float32)
```

뜻:

```text
YOLO는 cx, cy만 측정한다.
vx, vy는 직접 측정하지 않는다.
```

Innovation:

```text
실제 측정값 - 예측한 측정값
```

즉:

```text
"내가 예상한 위치와 YOLO가 말한 위치가 얼마나 다른가?"
```

## 34. 칼만 게인 K

칼만 게인은 예측과 센서 중 무엇을 더 믿을지 정하는 가중치다.

```text
K가 1에 가까움
→ 센서 측정값을 많이 믿음

K가 0에 가까움
→ 예측값을 많이 믿음
```

수식:

```text
K = P^- C^T (C P^- C^T + R)^-1
```

초심자에게 중요한 해석:

```text
예측이 불확실하면 P^-가 큼
→ 센서를 더 믿음

센서가 부정확하면 R이 큼
→ 예측을 더 믿음
```

사람이 직접 매번 `K`를 정하는 것이 아니라, `P`와 `R`을 보고 자동 계산된다.

## 35. Update 결과

보정 수식:

```text
x_k = x_k^- + K(y_k - Cx_k^-)
P_k = (I - KC)P_k^-
```

뜻:

```text
예측 상태에 보정량을 더해서 최종 상태를 만든다.
불확실성 P는 Update 후 줄어든다.
```

왜 줄어드나?

```text
새로운 센서 정보를 받았기 때문에
이전보다 더 잘 알게 되었기 때문
```

다만 완전히 0이 되지는 않는다.

```text
현실의 센서는 완벽하지 않기 때문
```

## 36. OpenCV KalmanFilter 코드와 수식 연결

OpenCV 코드:

```python
self.kf = cv2.KalmanFilter(4, 2)
```

뜻:

```text
상태 4개: x, y, dx, dy
측정 2개: x, y
```

측정 행렬:

```python
self.kf.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], np.float32)
```

수식의 `C`에 해당한다.

전이 행렬:

```python
self.kf.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)
```

수식의 `A`에 해당한다.

프로세스 노이즈:

```python
self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
```

수식의 `Q`에 해당한다.

측정값 넣기:

```python
measured = np.array([
    [np.float32(coord_x)],
    [np.float32(coord_y)]
])
```

이게 YOLO가 준 중심점이다.

예측과 보정:

```python
self.kf.predict()
corrected = self.kf.correct(measured)
```

뜻:

```text
predict(): A를 이용해 다음 위치 예측
correct(): YOLO 측정값으로 예측 보정
```

## 37. 성능 문제와 해결 방법

강의자료에서 강조하는 문제는 세 가지다.

### 지연 시간 Latency

문제:

```text
처리가 입력보다 느려서 화면이 밀림
```

해결:

```text
stream=True 사용
작은 모델 사용
최신 프레임 위주로 처리
```

### 탐지 결과 흔들림 Flickering

문제:

```text
박스가 프레임마다 조금씩 떨림
```

해결:

```text
칼만 필터 적용
```

### 엣지 장치 성능 저하

문제:

```text
라즈베리파이 등에서 1~2 FPS만 나옴
```

해결:

```text
yolov8n 사용
imgsz 줄이기
ONNX / OpenVINO / TensorRT 변환
FP16 / INT8 양자화
```

## 38. 실습 체크리스트

이 강의자료를 제대로 이해했는지 확인하려면 다음을 할 수 있어야 한다.

```text
[ ] ultralytics 설치 후 import 가능
[ ] yolov8n.pt 모델 로드 가능
[ ] 이미지 또는 웹캠에서 박스가 표시됨
[ ] box.xyxy, box.conf, box.cls를 꺼낼 수 있음
[ ] 박스 중심점 cx, cy를 계산할 수 있음
[ ] conf 또는 classes로 탐지 결과를 필터링할 수 있음
[ ] FPS가 어느 정도 나오는지 측정할 수 있음
[ ] YOLO 중심점에 칼만 필터를 적용할 수 있음
[ ] 필터 적용 전보다 추적점이 부드러워졌는지 확인할 수 있음
```

## 39. 05.02 전체 핵심 요약

이 자료의 핵심 문장:

```text
YOLO는 "무엇이 어디에 있는지" 빠르게 찾고,
칼만 필터는 "그 위치를 더 믿을 만하고 부드럽게" 만든다.
```

YOLO가 주는 것:

```text
label
confidence
bounding box
center point
```

칼만 필터가 하는 것:

```text
이전 움직임으로 다음 위치 예측
YOLO 측정값으로 보정
노이즈를 줄이고 추적을 안정화
```

로봇 비전에서 둘을 연결하면:

```text
YOLO 탐지 결과
→ 중심점 계산
→ 칼만 필터 추적
→ 로봇이 따라가기, 멈추기, 피하기, 경고하기
```

최종적으로 기억할 수식과 코드 연결:

```text
A = 상태가 다음 순간 어떻게 변하는가
C = 센서가 상태 중 무엇을 측정하는가
Q = 내 예측 모델을 얼마나 못 믿는가
R = 센서를 얼마나 못 믿는가
K = 예측과 센서를 섞는 비율
```

OpenCV 코드에서는:

```python
transitionMatrix    # A
measurementMatrix   # C
processNoiseCov     # Q
predict()           # 예측
correct(measured)   # 보정
```
