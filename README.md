# Jetson Orin Nano YOLO 헬멧 착용 검증 시스템
> 데이터셋 다운로드: https://drive.google.com/file/d/1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX/view

**목표**: 작업 현장에서 작업자의 헬멧 착용 여부를 실시간으로 검증하는 Edge AI 시스템 개발

**개발 환경**: MacBook Air M3 (ARM) → Jetson Orin Nano (ARM) 이식 가능 파이프라인

---

## 📋 프로젝트 개요

이 프로젝트는 **YOLO 모델 최적화 기술을 학습**하고, **헬멧 착용 검증 시스템**을 Jetson Orin Nano에 배포하는 것을 목표로 합니다.

### 핵심 질문
1. ❓ YOLO 모델을 Edge 디바이스에서 실시간으로 돌리려면 어떻게 최적화해야 하는가?
2. ❓ 양자화, ONNX, TensorRT는 무엇이고, 어떻게 적용하는가?
3. ❓ 헬멧 착용 검증을 위한 모델은 어떻게 준비하는가?

### 접근 방식
- **학습 방식**: 이론과 실습을 번갈아 진행하는 병렬 학습
- **모델 전략**: 사전 학습된 PPE 모델 + Fine-tuning
- **진행 방식**: 단계별 안내를 받으며 수동 진행 (학습 효과 극대화)

---

## 🎯 목표

### Phase 1: MacBook 학습 환경 (현재)
- [x] 프로젝트 Plan 문서 작성
- [ ] YOLO 최적화 원리 이해 (양자화, ONNX, TensorRT)
- [ ] MacBook에서 각 최적화 기법 실습 및 성능 측정
- [ ] 헬멧 검증용 사전 학습 모델 조사 및 선정

### Phase 2: Jetson Orin Nano 배포 (하드웨어 도착 후)
- [ ] TensorRT 변환 및 INT8 양자화 적용
- [ ] 실시간 추론 파이프라인 구축 (FPS > 30)
- [ ] 24시간 안정성 테스트
- [ ] 최종 배포 및 문서화

---

## 📂 프로젝트 구조

```
orin-yolo/
├── README.md                          # 프로젝트 개요 (이 파일)
├── docs/
│   └── pdca/
│       └── yolo-helmet-detection/
│           ├── plan.md                # Plan: 가설, 목표, 설계
│           ├── do.md                  # Do: 실험, 시행착오
│           ├── check.md               # Check: 평가, 분석
│           └── act.md                 # Act: 개선, 다음 액션
├── src/                               # 소스 코드
│   ├── optimization/                  # 최적화 관련 코드
│   ├── inference/                     # 추론 파이프라인
│   └── utils/                         # 유틸리티
├── notebooks/                         # Jupyter 노트북 (학습 및 실험)
├── models/                            # 학습된 모델 파일
├── data/                              # 데이터셋
│   ├── raw/                           # 원본 데이터
│   ├── processed/                     # 전처리된 데이터
│   └── validation/                    # 검증 데이터
├── tests/                             # 테스트 코드
├── scripts/                           # 셋업 및 유틸리티 스크립트
└── requirements.txt                   # Python 패키지 목록
```

---

## 🛠️ 기술 스택

### MacBook Phase
- **Framework**: Ultralytics YOLOv8
- **최적화**: PyTorch Quantization, ONNX Runtime
- **GPU Backend**: Metal Performance Shaders

### Jetson Phase
- **최적화**: TensorRT INT8 Quantization
- **GPU Backend**: CUDA 11.4, cuDNN
- **배포**: JetPack SDK 5.x

---

## 🚀 Quick Start

### 1. 환경 설정 (MacBook)
```bash
# Python 가상환경 생성 (pyenv + virtualenv)
pyenv install 3.10.12
pyenv virtualenv 3.10.12 orin-yolo
pyenv local orin-yolo

# 패키지 설치
pip install -U pip
pip install -r requirements.txt
```

### 2. YOLO 기본 동작 확인
```bash
# 샘플 이미지로 추론 테스트
python scripts/test_yolo_baseline.py
```

### 3. 학습 문서 확인
```bash
# PDCA Plan 문서 읽기
cat docs/pdca/yolo-helmet-detection/plan.md
```

---

## 📚 학습 자료

### 공식 문서
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)

### 데이터셋
- [Roboflow Universe - Hard Hat Detection](https://universe.roboflow.com/)
- [Kaggle - Construction Site Safety](https://www.kaggle.com/datasets)

---

## 📊 성능 목표

| 지표 | 베이스라인 | 목표 (최적화 후) |
|------|-----------|-----------------|
| **FPS** (Jetson) | 15-20 | 40-60 |
| **메모리** | 2-3GB | < 1.5GB |
| **정확도 (mAP)** | 0.85 | > 0.80 |
| **추론 시간** | 50-70ms | < 25ms |

---

## 🔄 PDCA Workflow

이 프로젝트는 **PDCA 사이클**을 따릅니다:

1. **Plan** (계획): 가설 수립, 목표 설정, 설계
   - 📄 `docs/pdca/yolo-helmet-detection/plan.md`

2. **Do** (실행): 실험, 시행착오, 학습
   - 📄 `docs/pdca/yolo-helmet-detection/do.md`

3. **Check** (평가): 결과 분석, 성과 측정
   - 📄 `docs/pdca/yolo-helmet-detection/check.md`

4. **Act** (개선): 성공 패턴 정리, 실패 방지책
   - 📄 `docs/pdca/yolo-helmet-detection/act.md`

---

## 📝 진행 상황

- [x] 2025-11-17: 프로젝트 Plan 문서 작성
- [ ] Week 1-2: MacBook 환경 구축 및 최적화 원리 학습
- [ ] Week 3-4: 헬멧 검증 모델 개발
- [ ] Week 5+: Jetson Orin Nano 배포

---

## 👤 개발자

- **Name**: lollolha97
- **Email**: lollolha97@gmail.com
- **Hardware**: MacBook Air M3 (2024), 16GB RAM, 512GB SSD

---

## 📄 라이선스

개인 학습 프로젝트. 코드는 MIT License, 모델은 각 소스의 라이선스를 따릅니다.

---

**마지막 업데이트**: 2025-11-17
