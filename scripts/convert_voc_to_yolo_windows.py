"""
Pascal VOC → YOLO 형식 변환 스크립트
Safety-Helmet-Wearing-Dataset 전용
"""

import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

# 클래스 매핑
CLASS_MAPPING = {
    'hat': 0,      # helmet (헬멧 착용)
    'person': 1    # no_helmet (헬멧 미착용)
}

def convert_voc_box_to_yolo(size, box):
    """
    VOC bounding box → YOLO 형식 변환

    Args:
        size: (width, height) 이미지 크기
        box: (xmin, ymin, xmax, ymax) VOC bbox

    Returns:
        (x_center, y_center, width, height) normalized
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]

    # Normalize
    x_center *= dw
    y_center *= dh
    width *= dw
    height *= dh

    return (x_center, y_center, width, height)

def convert_annotation(xml_path, output_txt_path):
    """
    하나의 XML 파일을 YOLO txt 파일로 변환

    Args:
        xml_path: Pascal VOC XML 파일 경로
        output_txt_path: 출력 YOLO txt 파일 경로
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 이미지 크기
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # YOLO 라벨 리스트
    yolo_labels = []

    # 모든 객체 처리
    for obj in root.iter('object'):
        cls_name = obj.find('name').text

        # 클래스 매핑 확인
        if cls_name not in CLASS_MAPPING:
            print(f"⚠️ Unknown class: {cls_name} in {xml_path}")
            continue

        cls_id = CLASS_MAPPING[cls_name]

        # Bounding box
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        ymin = float(xmlbox.find('ymin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymax = float(xmlbox.find('ymax').text)

        # VOC → YOLO 변환
        yolo_box = convert_voc_box_to_yolo((w, h), (xmin, ymin, xmax, ymax))

        # YOLO 라벨 형식: class x_center y_center width height
        yolo_label = f"{cls_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n"
        yolo_labels.append(yolo_label)

    # 파일 저장
    if yolo_labels:
        with open(output_txt_path, 'w') as f:
            f.writelines(yolo_labels)
        return True
    else:
        return False

def convert_dataset(voc_dir, output_dir):
    """
    전체 데이터셋 변환

    Args:
        voc_dir: Pascal VOC 데이터셋 디렉토리 (SHWD/)
        output_dir: YOLO 형식 출력 디렉토리
    """
    voc_dir = Path(voc_dir)
    output_dir = Path(output_dir)

    # 출력 디렉토리 생성
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 각 split 처리
    splits = ['train', 'val', 'test']
    stats = {'train': 0, 'val': 0, 'test': 0}

    for split in splits:
        print(f"\n📂 Processing {split} set...")

        # ImageSets/Main/{split}.txt 읽기
        split_file = voc_dir / 'ImageSets' / 'Main' / f'{split}.txt'

        if not split_file.exists():
            print(f"⚠️ {split_file} not found, skipping {split}")
            continue

        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]

        print(f"  Found {len(image_ids)} images")

        # 각 이미지 처리
        for img_id in image_ids:
            # XML 경로
            xml_path = voc_dir / 'Annotations' / f'{img_id}.xml'

            if not xml_path.exists():
                print(f"⚠️ XML not found: {xml_path}")
                continue

            # 이미지 경로
            img_path = voc_dir / 'JPEGImages' / f'{img_id}.jpg'

            if not img_path.exists():
                print(f"⚠️ Image not found: {img_path}")
                continue

            # YOLO txt 경로
            txt_path = output_dir / 'labels' / split / f'{img_id}.txt'

            # 변환
            success = convert_annotation(xml_path, txt_path)

            if success:
                # 이미지 복사
                dst_img_path = output_dir / 'images' / split / f'{img_id}.jpg'
                shutil.copy2(img_path, dst_img_path)
                stats[split] += 1

        print(f"  ✅ Converted {stats[split]} images")

    # data.yaml 생성
    create_data_yaml(output_dir, stats)

    print(f"\n{'='*60}")
    print(f"✅ 변환 완료!")
    print(f"{'='*60}")
    print(f"Train: {stats['train']} images")
    print(f"Val: {stats['val']} images")
    print(f"Test: {stats['test']} images")
    print(f"Total: {sum(stats.values())} images")
    print(f"\n출력 디렉토리: {output_dir}")

def create_data_yaml(output_dir, stats):
    """
    data.yaml 파일 생성

    Args:
        output_dir: 출력 디렉토리
        stats: 각 split별 이미지 개수
    """
    yaml_content = f"""# Safety-Helmet-Wearing-Dataset (YOLO format)
# Converted from Pascal VOC format

# Dataset root
path: {output_dir.absolute()}

# Splits
train: images/train  # {stats['train']} images
val: images/val      # {stats['val']} images
test: images/test    # {stats['test']} images

# Classes
names:
  0: helmet      # 헬멧 착용 (원본: hat)
  1: no_helmet   # 헬멧 미착용 (원본: person)

# Original dataset info
# Source: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset
# License: MIT
# Total images: {sum(stats.values())}
# Detection type: Person-level
"""

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n📝 data.yaml 생성: {yaml_path}")

if __name__ == "__main__":
    # 경로 설정
    BASE_DIR = Path.home() / "Developments" / "mirae-city" / "orin-yolo"
    VOC_DIR = BASE_DIR / "datasets" / "SHWD" / "VOC2028"
    OUTPUT_DIR = BASE_DIR / "datasets" / "helmet-detection"

    print("="*60)
    print("Pascal VOC → YOLO 변환 스크립트")
    print("Safety-Helmet-Wearing-Dataset")
    print("="*60)
    print(f"\n입력 디렉토리: {VOC_DIR}")
    print(f"출력 디렉토리: {OUTPUT_DIR}")

    # VOC 디렉토리 존재 확인
    if not VOC_DIR.exists():
        print(f"\n❌ 오류: VOC 디렉토리를 찾을 수 없습니다: {VOC_DIR}")
        print(f"\n다운로드 안내:")
        print(f"1. https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset")
        print(f"2. Google Drive/Baidu Drive에서 데이터셋 다운로드")
        print(f"3. {VOC_DIR} 에 압축 해제")
        exit(1)

    # 변환 시작
    user_input = input("\n변환을 시작하시겠습니까? (y/n): ")

    if user_input.lower() == 'y':
        convert_dataset(VOC_DIR, OUTPUT_DIR)

        print(f"\n다음 단계:")
        print(f"1. data.yaml 경로 확인: {OUTPUT_DIR / 'data.yaml'}")
        print(f"2. 학습 스크립트에서 data.yaml 사용")
        print(f"3. 학습 시작!")
    else:
        print("변환을 취소했습니다.")
