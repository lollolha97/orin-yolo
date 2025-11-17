"""
학습 완료 후 best 모델을 models/ 폴더에 저장하는 스크립트
"""

import shutil
from pathlib import Path
import argparse

def save_best_model(experiment_name, model_name=None):
    """
    runs/helmet-detection/{experiment_name}/weights/best.pt를
    models/ 폴더에 복사

    Args:
        experiment_name: 실험 이름 (예: "windows_yolo11n_shwd_e100")
        model_name: 저장할 모델 이름 (기본값: yolo11n_shwd_best.pt)
    """
    # 경로 설정
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    source = PROJECT_ROOT / "runs" / "helmet-detection" / experiment_name / "weights" / "best.pt"

    if model_name is None:
        model_name = "yolo11n_shwd_best.pt"

    dest = PROJECT_ROOT / "models" / model_name

    # 파일 존재 확인
    if not source.exists():
        print(f"❌ 오류: {source} 파일을 찾을 수 없습니다.")
        print(f"\n사용 가능한 실험:")
        experiments_dir = PROJECT_ROOT / "runs" / "helmet-detection"
        if experiments_dir.exists():
            for exp in experiments_dir.iterdir():
                if exp.is_dir():
                    print(f"  - {exp.name}")
        return False

    # models/ 디렉토리 확인
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 복사
    print(f"📦 모델 저장 중...")
    print(f"   Source: {source}")
    print(f"   Dest:   {dest}")

    shutil.copy2(source, dest)

    # 파일 크기 확인
    size_mb = dest.stat().st_size / (1024 * 1024)

    print(f"\n✅ 저장 완료!")
    print(f"   파일: {dest.name}")
    print(f"   크기: {size_mb:.2f} MB")
    print(f"\n사용법:")
    print(f"   from ultralytics import YOLO")
    print(f'   model = YOLO("models/{model_name}")')

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 모델을 models/ 폴더에 저장")
    parser.add_argument("experiment_name", help="실험 이름 (예: windows_yolo11n_shwd_e100)")
    parser.add_argument("--name", help="저장할 모델 이름 (기본: yolo11n_shwd_best.pt)")

    args = parser.parse_args()

    save_best_model(args.experiment_name, args.name)
