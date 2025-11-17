"""
Windows RTX 4060 테스트 학습 스크립트 (1 epoch)

빠른 테스트용 - 데이터셋과 학습 파이프라인 검증
"""

from ultralytics import YOLO
import torch
from pathlib import Path

def check_cuda():
    """CUDA 사용 가능 여부 확인"""
    print("=" * 60)
    print("🔍 GPU 환경 확인")
    print("=" * 60)

    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✅ CUDA 버전: {torch.version.cuda}")
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        device = 0  # GPU 사용
    else:
        print("❌ CUDA 미설치 또는 인식 불가")
        print("CPU로 학습 진행 (매우 느림)")
        device = "cpu"

    print()
    return device

def test_train_1epoch():
    """1 epoch 테스트 학습"""

    device = check_cuda()

    # 프로젝트 루트 경로 계산
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/training/ → orin-yolo/
    DATA_YAML = PROJECT_ROOT / "datasets" / "helmet-detection" / "data.yaml"

    print("=" * 60)
    print("🧪 테스트 학습 (1 epoch)")
    print("=" * 60)
    print("목적: 데이터셋 로딩 및 학습 파이프라인 검증")
    print("모델: yolo11n.pt")
    print("Epochs: 1")
    print("Batch: 16")
    print("이미지 크기: 640")
    print(f"디바이스: {device}")
    print(f"데이터: {DATA_YAML}")
    print()

    # 모델 로드
    model = YOLO("yolo11n.pt")

    # 테스트 학습 실행
    results = model.train(
        data=str(DATA_YAML),
        epochs=1,
        imgsz=640,
        batch=16,
        device=device,
        project="runs/helmet-detection",
        name="windows_test_1epoch",

        # 빠른 테스트 설정
        patience=0,         # Early stopping 비활성화
        save=True,          # 모델 저장
        save_period=-1,     # 중간 저장 안함 (마지막만)

        # 데이터 증강 최소화 (빠른 테스트)
        augment=False,

        # 학습 속도 향상
        workers=4,
        cache=False,        # 메모리 캐시 비활성화 (첫 테스트)

        verbose=True
    )

    print()
    print("=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)
    print(f"결과 디렉토리: {results.save_dir}")
    print(f"모델 저장: {results.save_dir}/weights/last.pt")
    print()
    print("다음 단계:")
    print("1. 학습 로그 확인 - 데이터 로딩 정상?")
    print("2. mAP 확인 - 1 epoch 결과는 낮아도 OK")
    print("3. 문제 없으면 본격 학습 시작:")
    print("   python src/training/train_windows.py")
    print()

    return results

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 YOLO 헬멧 검증 - 1 Epoch 테스트")
    print("=" * 60)
    print("Windows RTX 4060 환경")
    print("데이터셋: SHWD (5,457 train / 607 val)")
    print()

    input("Enter를 눌러 테스트 시작...")

    test_train_1epoch()
