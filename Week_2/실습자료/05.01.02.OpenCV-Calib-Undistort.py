# %% [markdown]
# # 실시간 왜곡 보정 (Live Undistort)
#
# `05.01.01.OpenCV-Calib-Capture.py` 와 동일한 합성 왜곡을 씌운 뒤
# `camera_info.yaml` 의 캘리브레이션 파라미터로 실시간 보정합니다.
#
# **사전 조건**
# - `camera_info.yaml` 이 존재해야 합니다 (`05.01.OpenCV-Calibration.py` 실행 후 생성)
#
# **사용 방법**
# - `q` 또는 `ESC` : 종료

# %%
import cv2
import numpy as np
import yaml
import os

# %% [markdown]
# ## 1. 설정

# %%
CAMERA_INDEX   = 0
YAML_PATH      = "camera_info.yaml"

# 05.01.01.OpenCV-Calib-Capture.py 와 동일한 합성 왜곡 설정
USE_DISTORTION = True
DIST_COEF      = np.array([[-0.40, 0.15, 0.001, -0.001, 0.05]], np.float64)

# %% [markdown]
# ## 2. 캘리브레이션 파라미터 로드

# %%
if not os.path.exists(YAML_PATH):
    raise FileNotFoundError(f"{YAML_PATH} 파일이 없습니다. 먼저 캘리브레이션을 실행하세요.")

with open(YAML_PATH, 'r') as f:
    data = yaml.safe_load(f)

mtx  = np.array(data['camera_matrix'], np.float64)
dist = np.array(data['dist_coeff'], np.float64)
print(f"캘리브레이션 파라미터 로드 완료: {YAML_PATH}")
print(f"fx={mtx[0,0]:.1f}  fy={mtx[1,1]:.1f}  cx={mtx[0,2]:.1f}  cy={mtx[1,2]:.1f}")

# %% [markdown]
# ## 3. 실시간 왜곡 → 보정

# %%
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"카메라(인덱스 {CAMERA_INDEX})를 열 수 없습니다.")

# 첫 프레임으로 맵 사전 계산
ret, frame = cap.read()
if not ret:
    raise RuntimeError("첫 프레임을 읽을 수 없습니다.")

h, w = frame.shape[:2]

# 합성 왜곡 맵 계산 (05.01.01과 동일한 방식, USE_DISTORTION=True 일 때만)
synth_map_x = synth_map_y = None
if USE_DISTORTION:
    _fx = _fy = w * 1.5
    _cx, _cy = w / 2, h / 2
    _k1, _k2, _p1, _p2, _k3 = DIST_COEF.ravel()[:5]
    _ug, _vg = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    _xd = (_ug - _cx) / _fx
    _yd = (_vg - _cy) / _fy
    _x, _y = _xd.copy(), _yd.copy()
    for _ in range(15):
        _r2  = _x*_x + _y*_y
        _rad = 1 + _k1*_r2 + _k2*_r2**2 + _k3*_r2**3
        _x   = (_xd - 2*_p1*_x*_y - _p2*(_r2 + 2*_x*_x)) / _rad
        _y   = (_yd - _p1*(_r2 + 2*_y*_y) - 2*_p2*_x*_y) / _rad
    synth_map_x = (_fx*_x + _cx).astype(np.float32)
    synth_map_y = (_fy*_y + _cy).astype(np.float32)
    print(f"합성 왜곡 맵 계산 완료 (k1={DIST_COEF[0,0]})")

# undistort 맵 계산 (매 프레임 반복 계산 방지)
undist_map1, undist_map2 = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), cv2.CV_16SC2)

print(f"맵 계산 완료 ({w}×{h})")
print("실행 중 — q/ESC: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 합성 왜곡 적용 (USE_DISTORTION=False 이면 원본 그대로 사용)
    if USE_DISTORTION:
        distorted = cv2.remap(frame, synth_map_x, synth_map_y,
                              cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        distorted = frame

    # 캘리브레이션 파라미터로 보정
    undistorted = cv2.remap(distorted, undist_map1, undist_map2, cv2.INTER_LINEAR)

    cv2.putText(distorted,   "Distorted",   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(undistorted, "Undistorted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Live Undistort", np.hstack([distorted, undistorted]))

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
print("종료")
