# Do: YOLO 헬멧 검증 시스템 실험 로그

**시작일**: 2025-11-17
**상태**: In Progress
**담당**: lollolha97

---

## 📝 실험 로그 (Implementation Log)

### 날짜별 작업 기록

#### 2025-11-17: 헬멧 착용 검증용 사전 학습 모델 조사
**작업 시간**: 시작

**수행 작업**:
- Ultralytics Construction-PPE 데이터셋 공식 문서 조사 (WebFetch)
- Roboflow Universe 헬멧 검증 모델 검색 (Tavily Search)
- Hugging Face 사전 학습 모델 조사 (Tavily Search)
- GitHub 오픈소스 프로젝트 탐색 (Tavily Search)
- Context7로 Ultralytics 공식 패턴 확인

**조사 도구**:
- ✅ WebFetch: https://docs.ultralytics.com/ko/datasets/detect/construction-ppe/
- ✅ Tavily: Advanced search (helmet detection, PPE, YOLOv8)
- ✅ Context7: /ultralytics/ultralytics 라이브러리 문서

**발견한 리소스**:
1. **Ultralytics 공식 Construction-PPE 데이터셋**
2. **Roboflow Universe 여러 사전 학습 모델**
3. **Hugging Face 모델 허브**
4. **GitHub 오픈소스 프로젝트 (사전 학습 가중치 포함)**

**학습 내용**:
- PPE 검증은 헬멧뿐 아니라 조끼, 장갑, 고글 등 포괄적 안전장비 검증
- "누락된 장비" 클래스(no_helmet 등)를 포함하는 데이터셋이 실시간 안전 위반 감지에 유리
- 사전 학습 모델이 다양하게 존재하므로 처음부터 학습할 필요 없음
- Roboflow Universe에서 바로 사용 가능한 API 제공

---

## 🧪 최적화 실험 기록

### 실험 1: 양자화 (Quantization)

#### Static Quantization
**날짜**: TBD
**목표**: FP32 → INT8 변환으로 추론 속도 2배 향상

**설정**:
```yaml
모델: YOLOv8n
입력 해상도: 640x640
Calibration 데이터: 100장
```

**결과**:
| 지표 | FP32 | INT8 | 변화율 |
|------|------|------|--------|
| FPS | - | - | - |
| 메모리 (MB) | - | - | - |
| mAP | - | - | - |
| 추론 시간 (ms) | - | - | - |

**관찰 사항**:
-

**다음 실험**:
-

---

#### Dynamic Quantization
**날짜**: TBD
**목표**: Static과 성능 비교

**결과**:
(TBD)

---

### 실험 2: ONNX 변환

**날짜**: TBD
**목표**: PyTorch → ONNX 변환 및 ONNX Runtime 최적화

**변환 설정**:
```python
# 변환 코드 스니펫
```

**결과**:
| 지표 | PyTorch | ONNX Runtime | 변화율 |
|------|---------|--------------|--------|
| FPS | - | - | - |
| 메모리 (MB) | - | - | - |
| mAP | - | - | - |

**관찰 사항**:
-

---

### 실험 3: TensorRT (Jetson Phase)

**날짜**: TBD
**목표**: ONNX → TensorRT 엔진 변환 및 INT8 Calibration

**변환 설정**:
```bash
# trtexec 명령어
```

**결과**:
(Jetson 도착 후 실험)

---

## 🎯 헬멧 모델 개발 기록

### 모델 조사 결과 (2025-11-17)

#### 후보 1: Ultralytics Construction-PPE 데이터셋 ⭐ (공식 권장)
**출처**: Ultralytics 공식 데이터셋
**링크**: https://docs.ultralytics.com/datasets/detect/construction-ppe/
**다운로드**: https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip

**데이터셋 정보**:
- **크기**: 1,416개 이미지 (train 1,132 + val 143 + test 141)
- **용량**: 178.4 MB
- **라이선스**: AGPL-3.0

**클래스** (11개):
- **착용 장비**: `helmet`, `gloves`, `vest`, `boots`, `goggles`, `worker`
- **누락 장비**: `no_helmet`, `no_goggle`, `no_gloves`, `no_boots`, `none`

**학습 방법**:
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # 사전 학습 모델
model.train(data="construction-ppe.yaml", epochs=100, imgsz=640)
```

**평가**:
- ✅ **장점**:
  - Ultralytics 공식 지원 → 문서화 완벽, 유지보수 보장
  - "누락된 장비" 클래스 포함 → 실시간 안전 위반 감지 가능
  - 다양한 조명, 자세, 환경에서 수집된 실제 건설 현장 데이터
  - yolo11n.pt 사전 학습 모델 사용 가능
  - 규정 준수/미준수 사례 모두 포함

- ⚠️ **단점**:
  - 데이터셋 크기가 중간 정도 (1,416개)
  - 특정 작업 환경에 대한 Fine-tuning 필요할 수 있음

- 🎯 **선정 여부**: **최우선 후보 (1순위)**
- 📊 **품질 평가**: 공식 데이터셋, 고품질 큐레이션

---

#### 후보 2: Roboflow Universe - PPE Detection Models
**출처**: Roboflow Universe (커뮤니티 모델)
**링크**: https://universe.roboflow.com/search?q=class%3Ahelmet

**발견한 모델들**:

##### 2-1. Helmet Detection YOLOv8
- **이미지**: 500개
- **제공**: 사전 학습 모델 + API
- **클래스**: helmet 관련
- **상태**: 바로 사용 가능

##### 2-2. Construction PPE Detection (by Huiyao Hu)
- **이미지**: 2,092개
- **모델**: YOLOv8s
- **클래스**: `helmet`, `human`, `vest`, `boots`, `gloves`
- **상태**: 학습된 가중치 제공

##### 2-3. PPE Detection (by HX)
- **이미지**: 2,197개
- **모델**: YOLOv8, YOLOv8m, YOLOv8l, YOLOv11
- **클래스**: `helmet`, `human`, `vest`, `boots`, `gloves`
- **상태**: 여러 모델 크기 제공

##### 2-4. Construction Site Safety (GitHub: snehilsanyal)
- **링크**: https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection
- **이미지**: 2,801개 (train 2,605 + val 114 + test 82)
- **클래스** (10개): `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `machinery`, `vehicle`
- **상태**: YOLOv8 사전 학습 가중치 제공

**평가**:
- ✅ **장점**:
  - 즉시 사용 가능한 API 제공 (Roboflow)
  - 다양한 모델 크기 선택 가능 (n, s, m, l)
  - 커뮤니티에서 검증된 모델들
  - 일부는 Ultralytics보다 데이터셋 크기가 큼

- ⚠️ **단점**:
  - 커뮤니티 모델 → 품질 일관성 보장 어려움
  - 라이선스 개별 확인 필요
  - 유지보수 불확실

- 🎯 **선정 여부**: **2순위 (Ultralytics 대안)**
- 📊 **품질 평가**: 커뮤니티 검증, 다양한 선택지

---

#### 후보 3: Hugging Face - keremberke/yolov8m-protective-equipment-detection
**출처**: Hugging Face Model Hub
**링크**: https://huggingface.co/keremberke/yolov8m-protective-equipment-detection
**모델**: YOLOv8m (Medium 크기)

**클래스** (10개):
`glove`, `goggles`, `helmet`, `mask`, `no_glove`, `no_goggles`, `no_helmet`, `no_mask`, `no_shoes`, `shoes`

**사용 방법**:
```python
from ultralyticsplus import YOLO
model = YOLO('keremberke/yolov8m-protective-equipment-detection')
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45   # NMS IoU threshold
# 추론 실행
```

**평가**:
- ✅ **장점**:
  - Hugging Face에서 바로 다운로드 가능
  - YOLOv8m → Nano보다 정확도 높음
  - "누락된 장비" 클래스 포함
  - ultralyticsplus 라이브러리로 쉬운 사용

- ⚠️ **단점**:
  - Medium 모델 → Jetson Orin Nano에서 무거울 수 있음
  - 데이터셋 정보 불명확

- 🎯 **선정 여부**: **3순위 (정확도 우선 시나리오)**
- 📊 **품질 평가**: Hugging Face 검증, 즉시 사용 가능

---

#### 후보 4: GitHub - Helmet Detection Projects
**출처**: GitHub 오픈소스 프로젝트

##### 4-1. Vansh2693/Helmet_Detection_OpenCV
- **링크**: https://github.com/Vansh2693/Helmet_Detection_OpenCV
- **제공**: 사전 학습 가중치 (helmet.pt)
- **데이터**: Roboflow 데이터셋 사용
- **특징**: 영상 추론 예제 코드 포함

##### 4-2. M3GHAN/YOLOv8-Object-Detection
- **링크**: https://github.com/M3GHAN/YOLOv8-Object-Detection
- **특징**: PascalVOC → YOLO 형식 변환 포함
- **제공**: 학습 및 추론 전체 파이프라인

**평가**:
- ✅ **장점**:
  - 실제 작동하는 전체 코드 제공
  - 학습 경험 공유 → 참고 가능
  - 영상 추론 예제

- ⚠️ **단점**:
  - 개인 프로젝트 → 품질 보장 없음
  - 유지보수 불확실

- 🎯 **선정 여부**: **참고용 (코드 학습)**
- 📊 **품질 평가**: 학습 자료로 활용

---

### 최종 선정 전략 (권장)

#### 1단계: Ultralytics Construction-PPE 베이스라인
```python
# 공식 데이터셋으로 시작
model = YOLO("yolo11n.pt")
model.train(data="construction-ppe.yaml", epochs=100, imgsz=640)
```
**이유**:
- 공식 지원 → 문서화 완벽
- 안정적인 품질
- 작은 모델(yolo11n) → Jetson에 적합

#### 2단계: 성능 평가 및 비교
- Ultralytics 모델 성능 측정
- Roboflow 모델 중 1-2개 테스트
- mAP, FPS, 메모리 비교

#### 3단계: Fine-tuning (필요 시)
- 작업 환경 데이터 100장 수집
- 베이스라인 모델에 Fine-tuning
- 성능 개선 확인

#### 4단계: 최적화 적용
- 선정된 모델에 양자화, TensorRT 적용
- Jetson 배포

---

### Fine-tuning 기록

**날짜**: TBD
**베이스 모델**:
**데이터셋**:

**학습 설정**:
```yaml
Epochs:
Batch Size:
Learning Rate:
Augmentation:
```

**학습 로그**:
```
Epoch 1/N: loss=X.XX, mAP=X.XX
Epoch 2/N: loss=X.XX, mAP=X.XX
...
```

**최종 성능**:
- mAP:
- Precision:
- Recall:

---

## 💡 시행착오 및 학습 (Trial & Error)

### 배운 것 (What Worked)
1.

### 실패한 것 (What Failed)
1.

### 다음에 시도할 것 (What's Next)
1.

---

## 🔗 참고 자료

### 도움이 된 문서
- [Ultralytics Construction-PPE Dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/)
- [Roboflow Universe - Helmet Detection Models](https://universe.roboflow.com/search?q=class%3Ahelmet)
- [Hugging Face - YOLOv8 Protective Equipment Detection](https://huggingface.co/keremberke/yolov8m-protective-equipment-detection)
- [Construction Safety: YOLOv8 for PPE Detection - Medium](https://python.plainenglish.io/enhancing-workplace-safety-a-guide-to-custom-training-yolov8-for-safety-helmet-detection-a928bf9c6f6e)
- [Ultralytics YOLO Training Documentation](https://docs.ultralytics.com/)

### 참고한 코드
- [snehilsanyal/Construction-Site-Safety-PPE-Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)
- [Vansh2693/Helmet_Detection_OpenCV](https://github.com/Vansh2693/Helmet_Detection_OpenCV)
- [M3GHAN/YOLOv8-Object-Detection](https://github.com/M3GHAN/YOLOv8-Object-Detection)

### 학술 자료
- [Personal protective equipment detection using YOLOv8](https://www.tandfonline.com/doi/full/10.1080/23311916.2024.2333209)
- [An improved YOLOv8 safety helmet wearing detection network](https://www.nature.com/articles/s41598-024-68446-z)
- [Detection Method for Safety Helmet Wearing on Construction Sites](https://www.mdpi.com/2075-5309/15/3/354)

---

**최종 업데이트**: 2025-11-17
