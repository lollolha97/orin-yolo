"""
Construction-PPE 데이터셋 기본 동작 확인 스크립트

목적:
1. Ultralytics 정상 설치 확인
2. Construction-PPE 데이터셋 자동 다운로드 확인
3. 간단한 학습 실행으로 동작 검증 (3 epochs만)
4. 베이스라인 성능 측정
"""

from ultralytics import YOLO
import time
import torch

def check_environment():
    """환경 설정 확인"""
    print("=" * 60)
    print("🔍 환경 확인 중...")
    print("=" * 60)

    # Python 버전
    import sys
    print(f"Python 버전: {sys.version}")

    # PyTorch 버전
    print(f"PyTorch 버전: {torch.__version__}")

    # MPS (Metal Performance Shaders) 사용 가능 여부 (MacBook M3)
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal) 가속 사용 가능!")
        device = "mps"
    else:
        print("⚠️ MPS 사용 불가, CPU로 실행")
        device = "cpu"

    print(f"사용할 디바이스: {device}")
    print()

    return device

def test_basic_inference():
    """기본 추론 테스트 (사전 학습 모델)"""
    print("=" * 60)
    print("🚀 Step 1: 사전 학습 모델 다운로드 및 추론 테스트")
    print("=" * 60)

    # YOLOv11n (가장 작은 모델) 다운로드
    print("YOLOv11n 모델 다운로드 중...")
    model = YOLO("yolo11n.pt")

    print(f"✅ 모델 다운로드 완료: {model.ckpt_path}")
    print(f"모델 정보: {model.info()}")
    print()

    return model

def test_construction_ppe_training(model, device, epochs=3):
    """Construction-PPE 데이터셋으로 학습 테스트"""
    print("=" * 60)
    print("🏗️ Step 2: Construction-PPE 데이터셋 학습 테스트")
    print("=" * 60)
    print(f"Epochs: {epochs} (빠른 테스트용)")
    print(f"디바이스: {device}")
    print()

    # 학습 시작 시간 기록
    start_time = time.time()

    print("학습 시작...")
    print("📥 데이터셋 자동 다운로드 시작 (최초 1회만, 178.4 MB)")
    print()

    try:
        # Construction-PPE 데이터셋으로 학습
        # imgsz=640: 입력 이미지 크기
        # batch=16: 배치 크기 (MacBook M3에 맞게 조정)
        # epochs=3: 빠른 테스트를 위해 3 epoch만
        results = model.train(
            data="construction-ppe.yaml",
            epochs=epochs,
            imgsz=640,
            batch=16,
            device=device,
            project="runs/construction-ppe",
            name="test_run",
            verbose=True
        )

        # 학습 시간 계산
        elapsed_time = time.time() - start_time

        print()
        print("=" * 60)
        print("✅ 학습 완료!")
        print("=" * 60)
        print(f"소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
        print(f"저장 위치: {results.save_dir}")
        print()

        # 학습 결과 요약
        print("📊 학습 결과 요약:")
        print(f"- 최종 mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"- 최종 mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print()

        return results

    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        raise

def test_validation(model):
    """검증 데이터셋으로 성능 평가"""
    print("=" * 60)
    print("📊 Step 3: 검증 데이터셋 평가")
    print("=" * 60)

    try:
        metrics = model.val(data="construction-ppe.yaml")

        print()
        print("검증 결과:")
        print(f"- mAP50: {metrics.box.map50:.4f}")
        print(f"- mAP50-95: {metrics.box.map:.4f}")
        print(f"- Precision: {metrics.box.mp:.4f}")
        print(f"- Recall: {metrics.box.mr:.4f}")
        print()

        return metrics

    except Exception as e:
        print(f"⚠️ 검증 중 오류: {e}")
        return None

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎯 " + "=" * 56)
    print("   Construction-PPE 데이터셋 기본 동작 확인 테스트")
    print("=" * 60)
    print()

    # 1. 환경 확인
    device = check_environment()

    # 2. 사전 학습 모델 테스트
    model = test_basic_inference()

    # 3. Construction-PPE 학습 테스트 (3 epochs)
    print("⚠️  주의: 이 테스트는 3 epochs만 실행합니다.")
    print("   완전한 학습을 위해서는 100+ epochs 필요")
    print()

    user_input = input("학습을 시작하시겠습니까? (y/n): ")

    if user_input.lower() == 'y':
        results = test_construction_ppe_training(model, device, epochs=3)

        # 4. 검증 (선택)
        print()
        val_input = input("검증 데이터셋으로 평가하시겠습니까? (y/n): ")
        if val_input.lower() == 'y':
            test_validation(model)

        print()
        print("=" * 60)
        print("🎉 모든 테스트 완료!")
        print("=" * 60)
        print()
        print("다음 단계:")
        print("1. runs/construction-ppe/test_run/ 에서 학습 결과 확인")
        print("2. TensorBoard로 학습 곡선 시각화:")
        print("   tensorboard --logdir runs/construction-ppe/test_run")
        print("3. 본격적인 학습을 위해서는 epochs=100으로 재실행")
        print()
    else:
        print("테스트를 중단했습니다.")

if __name__ == "__main__":
    main()
