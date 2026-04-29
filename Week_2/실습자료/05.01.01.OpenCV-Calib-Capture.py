# %% [markdown]
# # 카메라 캘리브레이션 이미지 수집
#
# 체커보드를 이용해 캘리브레이션용 사진을 촬영하고 저장하는 실습입니다.
#
# **사용 방법**
# - `SPACE` 또는 `c` : 체커보드가 감지되면 현재 프레임 저장
# - `q` 또는 `ESC` : 촬영 종료
#
# **촬영 권장 사항**
# - 최소 10~20장 촬영 (다양한 각도·거리·위치)
# - 체커보드를 ±15°~45° 다양하게 기울여 촬영
# - 이미지 전체 영역(특히 가장자리)을 골고루 커버

# %%
import cv2
import numpy as np
import os
from datetime import datetime

# %% [markdown]
# ## 1. 설정

# %%
# 체커보드 내부 교차점 수 (칸 수 - 1)
# 예: 10×7 칸 → 내부 교차점 9×6개
CHECKERBOARD = (9, 6)

# 저장 폴더
SAVE_DIR = "calib_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 카메라 인덱스 (기본 0번)
CAMERA_INDEX = 0

# True: 카메라 영상에 배럴 왜곡을 씌워 저장 (캘리브레이션 효과를 눈에 띄게 확인)
# cx, cy는 실제 프레임 크기에 맞게 아래 루프에서 자동 보정됨
USE_DISTORTION = True
DIST_COEF = np.array([[-0.40, 0.15, 0.001, -0.001, 0.05]], np.float64)

print(f"체커보드 내부 교차점: {CHECKERBOARD[0]}×{CHECKERBOARD[1]}")
print(f"저장 폴더: {os.path.abspath(SAVE_DIR)}")
print(f"합성 왜곡: {'ON  k1=' + str(DIST_COEF[0,0]) if USE_DISTORTION else 'OFF'}")

# %% [markdown]
# ## 2. 실시간 촬영 및 이미지 수집

# %%
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"카메라(인덱스 {CAMERA_INDEX})를 열 수 없습니다.")

# USE_DISTORTION=True 일 때 왜곡 맵을 미리 계산 (루프 안에서 매 프레임 반복 계산 방지)
dist_map_x = dist_map_y = None
if USE_DISTORTION:
    _fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # cx, cy를 실제 프레임 중앙으로, fx가 작을수록 정규화 좌표 범위가 넓어져 왜곡이 강해짐
    _fx = _fy = _fw * 1.5
    _cx, _cy = _fw / 2, _fh / 2
    _k1, _k2, _p1, _p2, _k3 = DIST_COEF.ravel()[:5]
    _ug, _vg = np.meshgrid(np.arange(_fw, dtype=np.float64),
                            np.arange(_fh, dtype=np.float64))
    _xd = (_ug - _cx) / _fx
    _yd = (_vg - _cy) / _fy
    _x, _y = _xd.copy(), _yd.copy()
    for _ in range(15):
        _r2  = _x*_x + _y*_y
        _rad = 1 + _k1*_r2 + _k2*_r2**2 + _k3*_r2**3
        _x   = (_xd - 2*_p1*_x*_y - _p2*(_r2 + 2*_x*_x)) / _rad
        _y   = (_yd - _p1*(_r2 + 2*_y*_y) - 2*_p2*_x*_y) / _rad
    dist_map_x = (_fx*_x + _cx).astype(np.float32)
    dist_map_y = (_fy*_y + _cy).astype(np.float32)
    print(f"왜곡 맵 계산 완료 ({_fw}×{_fh})")

# cornerSubPix() 반복 종료 조건:
#   - MAX_ITER: 최대 30회 반복
#   - EPS     : 코너 이동량이 0.001px 미만이면 수렴으로 판단
# 두 조건 중 하나라도 만족하면 멈춤
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

saved_count = 0
print("촬영 시작 — SPACE/c: 저장,  q/ESC: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 왜곡 씌우기 (USE_DISTORTION=True 일 때)
    if USE_DISTORTION:
        distorted = cv2.remap(frame, dist_map_x, dist_map_y,
                              cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        distorted = frame

    gray = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    display = distorted.copy()

    if found:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners_refined, found)
        status_text = f"DETECTED — SPACE/c: save ({saved_count}장 저장됨)"
        color = (0, 255, 0)
    else:
        status_text = f"체커보드 미감지 ({saved_count}장 저장됨)"
        color = (0, 0, 255)

    cv2.putText(display, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 왜곡 ON 일 때 원본과 나란히 표시
    if USE_DISTORTION:
        label_orig = frame.copy()
        cv2.putText(label_orig, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(display, "Distorted", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        show = np.hstack([label_orig, display])
    else:
        show = display

    cv2.imshow("Calibration Image Capture", show)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):   # q 또는 ESC
        break
    elif key in (ord(' '), ord('c')):
        if found:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(SAVE_DIR, f"calib_{timestamp}.jpg")
            cv2.imwrite(filename, distorted)   # 왜곡된 프레임 저장
            saved_count += 1
            print(f"  [{saved_count:02d}] 저장: {filename}")
        else:
            print("  체커보드가 감지되지 않아 저장을 건너뜁니다.")

cap.release()
cv2.destroyAllWindows()
print(f"\n촬영 완료 — 총 {saved_count}장 저장 → {os.path.abspath(SAVE_DIR)}")

# %% [markdown]
# ## 3. 수집된 이미지 확인

# %%
import glob

images = sorted(glob.glob(os.path.join(SAVE_DIR, "*.jpg")) +
                glob.glob(os.path.join(SAVE_DIR, "*.png")))
print(f"저장된 이미지: {len(images)}장")
for path in images:
    print(f"  {os.path.basename(path)}")
