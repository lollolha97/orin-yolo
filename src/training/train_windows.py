"""
Windows RTX 4060 GPU 학습 스크립트

Mac에서 작성 → Windows에서 실행
"""

from ultralytics import YOLO
import torch

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

def train_construction_ppe(
    epochs=100,
    batch=32,
    imgsz=640,
    model_size="n"  # n, s, m, l, x
):
    """Construction-PPE 데이터셋 학습"""

    device = check_cuda()

    print("=" * 60)
    print("🏋️ Construction-PPE 학습 시작")
    print("=" * 60)
    print(f"모델: yolo11{model_size}.pt")
    print(f"Epochs: {epochs}")
    print(f"Batch: {batch}")
    print(f"이미지 크기: {imgsz}")
    print(f"디바이스: {device}")
    print()

    # 모델 로드
    model = YOLO(f"yolo11{model_size}.pt")

    # 학습 실행
    results = model.train(
        data="construction-ppe.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/construction-ppe",
        name=f"windows_yolo11{model_size}_e{epochs}",

        # 최적화 설정
        patience=50,        # Early stopping
        save_period=10,     # 10 epoch마다 저장

        # 데이터 증강
        augment=True,

        # 학습 속도 향상
        workers=8,          # RTX 4060에 맞게 조정
        cache=True,         # 메모리에 캐시 (속도 향상)

        verbose=True
    )

    print()
    print("=" * 60)
    print("✅ 학습 완료!")
    print("=" * 60)
    print(f"최고 성능 모델: {results.save_dir}/weights/best.pt")
    print(f"마지막 모델: {results.save_dir}/weights/last.pt")
    print()

    return results

if __name__ == "__main__":
    # RTX 4060에 최적화된 설정
    train_construction_ppe(
        epochs=100,
        batch=32,      # RTX 4060 8GB → batch 32 권장
        imgsz=640,
        model_size="n" # Nano 모델 (Jetson 타겟이므로)
    )
