# Models Directory

학습된 YOLO 모델 저장소

## 📦 저장 규칙

학습 완료 후 best 모델을 이곳에 복사:

```bash
# Windows
copy runs\helmet-detection\{experiment_name}\weights\best.pt models\yolo11n_shwd_best.pt

# Mac/Linux
cp runs/helmet-detection/{experiment_name}/weights/best.pt models/yolo11n_shwd_best.pt
```

## 📋 모델 목록

### YOLOv11n - SHWD Dataset

| 모델명 | 학습일 | Epochs | mAP50 | mAP50-95 | 파일크기 | 비고 |
|--------|--------|--------|-------|----------|---------|------|
| `yolo11n_shwd_best.pt` | TBD | 100 | TBD | TBD | ~6MB | 최종 모델 |
| `yolo11n_shwd_e50.pt` | TBD | 50 | TBD | TBD | ~6MB | 중간 체크포인트 |

## 🚀 모델 사용법

### 추론 (Inference)
```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('models/yolo11n_shwd_best.pt')

# 추론
results = model('path/to/image.jpg')
```

### 검증 (Validation)
```bash
yolo val model=models/yolo11n_shwd_best.pt data=datasets/helmet-detection/data.yaml
```

### 내보내기 (Export)
```bash
# ONNX
yolo export model=models/yolo11n_shwd_best.pt format=onnx

# TensorRT (Jetson용)
yolo export model=models/yolo11n_shwd_best.pt format=engine device=0
```

## 📊 성능 기록

### 학습 환경
- **MacBook**: 1 epoch 테스트용
- **Windows RTX 4060**: 본격 학습 (100 epochs)
- **Dataset**: SHWD (5,457 train / 607 val / 1,517 test)

### 최종 모델 성능
- Train mAP: TBD
- Val mAP: TBD
- Test mAP: TBD
- Inference Time (RTX 4060): TBD ms
- Inference Time (Jetson Orin Nano): TBD ms

---

**⚠️ 주의**:
- 모델 파일은 Git에 포함되므로 용량 주의 (YOLOv11n ~6MB)
- 큰 모델(m, l, x)은 Git LFS 사용 권장
