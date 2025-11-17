# Windows RTX 4060 학습 환경 설정 가이드

**목적**: Mac에서 개발 → Windows RTX 4060에서 학습

---

## 🎯 빠른 시작

### 1. CUDA 확인
```cmd
nvidia-smi
```
출력 예시:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.xx       Driver Version: 545.xx       CUDA Version: 12.x               |
```

**미설치 시**: https://developer.nvidia.com/cuda-downloads

---

### 2. 환경 설정 (자동)
```cmd
setup_windows.bat
```

또는 **수동 설정**:

```cmd
# Python 가상환경
python -m venv venv
venv\Scripts\activate

# PyTorch GPU 버전 (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Ultralytics
pip install ultralytics
```

---

### 3. GPU 확인
```cmd
venv\Scripts\activate
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**출력**: `CUDA: True` ✅

---

### 4. 학습 실행
```cmd
venv\Scripts\activate
python src\training\train_windows.py
```

**예상 시간**: 100 epochs → 약 1-2시간 (RTX 4060)

---

## 📊 학습 결과

**저장 위치**:
```
runs/construction-ppe/windows_yolo11n_e100/
├── weights/
│   ├── best.pt      ← 최고 성능 (이 파일을 사용)
│   └── last.pt      ← 마지막 epoch
├── results.png
└── ...
```

---

## 🔄 Mac과 동기화

### GitHub 사용
```bash
# Windows에서 학습 후
git add runs/construction-ppe/windows_yolo11n_e100/weights/best.pt
git commit -m "Add trained model weights"
git push

# Mac에서
git pull
```

### 수동 복사
`best.pt`만 Mac의 `models/` 폴더로 복사

---

## ⚙️ RTX 4060 최적화 설정

| 파라미터 | 값 | 이유 |
|---------|---|------|
| `batch` | 32 | 8GB VRAM에 최적 |
| `workers` | 8 | CPU 활용 |
| `cache` | True | 메모리 캐시 → 속도↑ |
| `imgsz` | 640 | 표준 해상도 |

**메모리 부족 시**: `batch=16`으로 줄이기

---

## 🐛 문제 해결

### CUDA 인식 안됨
```cmd
# PyTorch CUDA 버전 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 메모리 부족
```python
# train_windows.py 수정
batch=16  # 32 → 16
```

### 느린 학습
- `cache=False` 제거
- `workers=4`로 줄이기
- 백그라운드 프로그램 종료

---

## 📝 CUDA 버전 확인 방법

```cmd
nvidia-smi
nvcc --version
```

**PyTorch 호환성**:
- CUDA 11.8 → `cu118`
- CUDA 12.1 → `cu121`
- CUDA 12.4 → `cu124`

---

**작성일**: 2025-11-17
