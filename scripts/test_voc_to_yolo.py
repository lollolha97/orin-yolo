"""
Pascal VOC → YOLO 변환 테스트 스크립트
2-3개 이미지만 변환하여 검증
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2

# 클래스 매핑
CLASS_MAPPING = {
    'hat': 0,      # helmet
    'person': 1    # no_helmet
}

def convert_voc_box_to_yolo(size, box):
    """VOC bbox → YOLO 변환"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]

    x_center *= dw
    y_center *= dh
    width *= dw
    height *= dh

    return (x_center, y_center, width, height)

def draw_yolo_boxes(img, yolo_labels, class_names):
    """
    YOLO 라벨을 이미지에 그리기

    Args:
        img: OpenCV 이미지
        yolo_labels: YOLO 형식 라벨 리스트
        class_names: 클래스 이름 딕셔너리

    Returns:
        bbox가 그려진 이미지
    """
    h, w = img.shape[:2]
    img_copy = img.copy()

    for label in yolo_labels:
        parts = label.split()
        cls_id = int(parts[0])
        x_c = float(parts[1])
        y_c = float(parts[2])
        box_w = float(parts[3])
        box_h = float(parts[4])

        # 정규화된 좌표 → 픽셀 좌표
        x_center = int(x_c * w)
        y_center = int(y_c * h)
        width = int(box_w * w)
        height = int(box_h * h)

        # 좌상단, 우하단 좌표
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # 클래스별 색상
        color = (0, 255, 0) if cls_id == 0 else (0, 165, 255)  # helmet: 초록, no_helmet: 주황
        class_name = class_names.get(cls_id, f"class_{cls_id}")

        # 박스 그리기
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # 라벨 텍스트
        label_text = f"{class_name}"
        cv2.putText(img_copy, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img_copy

def test_conversion(voc_dir, num_samples=3):
    """
    샘플 변환 테스트 + 시각화

    Args:
        voc_dir: VOC 데이터셋 디렉토리
        num_samples: 테스트할 샘플 개수
    """
    voc_dir = Path(voc_dir)

    print("="*60)
    print("Pascal VOC → YOLO 변환 테스트")
    print("="*60)
    print(f"VOC 디렉토리: {voc_dir}")
    print(f"테스트 샘플: {num_samples}개\n")

    # train.txt에서 샘플 ID 가져오기
    train_file = voc_dir / 'ImageSets' / 'Main' / 'train.txt'

    if not train_file.exists():
        print(f"❌ 오류: {train_file} 없음")
        return

    with open(train_file, 'r') as f:
        sample_ids = [line.strip() for line in f.readlines()[:num_samples]]

    print(f"테스트 샘플 ID: {sample_ids}\n")

    # 각 샘플 변환 테스트
    for idx, sample_id in enumerate(sample_ids, 1):
        print(f"{'='*60}")
        print(f"[{idx}/{num_samples}] 샘플 ID: {sample_id}")
        print(f"{'='*60}")

        # 경로 설정
        xml_path = voc_dir / 'Annotations' / f'{sample_id}.xml'
        img_path = voc_dir / 'JPEGImages' / f'{sample_id}.jpg'

        if not xml_path.exists():
            print(f"⚠️ XML 없음: {xml_path}")
            continue

        if not img_path.exists():
            print(f"⚠️ 이미지 없음: {img_path}")
            continue

        # 이미지 정보
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ 이미지 로드 실패")
            continue

        h, w = img.shape[:2]
        print(f"📷 이미지 크기: {w}x{h}")

        # XML 파싱
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 원본 XML 정보
        print(f"\n📄 원본 XML:")
        objects = list(root.iter('object'))
        print(f"   객체 수: {len(objects)}")

        # 변환
        yolo_labels = []

        for obj in objects:
            cls_name = obj.find('name').text

            if cls_name not in CLASS_MAPPING:
                print(f"   ⚠️ 알 수 없는 클래스: {cls_name}")
                continue

            cls_id = CLASS_MAPPING[cls_name]

            # Bounding box
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)

            print(f"   - {cls_name} (class {cls_id}): [{int(xmin)}, {int(ymin)}, {int(xmax)}, {int(ymax)}]")

            # YOLO 변환
            yolo_box = convert_voc_box_to_yolo((w, h), (xmin, ymin, xmax, ymax))
            yolo_label = f"{cls_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}"
            yolo_labels.append(yolo_label)

        # YOLO 형식 출력
        print(f"\n✅ YOLO 변환 결과:")
        print(f"   파일명: {sample_id}.txt")
        print(f"   내용:")
        for label in yolo_labels:
            parts = label.split()
            cls_id = int(parts[0])
            cls_name = 'helmet' if cls_id == 0 else 'no_helmet'
            print(f"      {parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]}  # {cls_name}")

        # 검증
        print(f"\n🔍 검증:")
        for label in yolo_labels:
            parts = [float(x) for x in label.split()]
            cls_id, x_c, y_c, box_w, box_h = parts

            # 범위 체크 (0~1)
            if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 <= box_w <= 1 and 0 <= box_h <= 1):
                print(f"   ❌ 좌표 범위 오류: {label}")
            else:
                print(f"   ✅ 좌표 범위 정상")

        # 시각화
        print(f"\n🎨 시각화:")
        class_names = {0: 'helmet', 1: 'no_helmet'}
        img_with_boxes = draw_yolo_boxes(img, yolo_labels, class_names)

        # 저장
        output_dir = Path.home() / "Developments" / "orin-yolo" / "test_output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{sample_id}_yolo.jpg"
        cv2.imwrite(str(output_path), img_with_boxes)
        print(f"   📁 저장: {output_path}")

        print()

    print("="*60)
    print("✅ 테스트 완료!")
    print("="*60)
    print("\n📁 시각화 결과:")
    print(f"   {Path.home() / 'Developments' / 'orin-yolo' / 'test_output'}/")
    print(f"   - 000000_yolo.jpg")
    print(f"   - 000001_yolo.jpg")
    print(f"   - 000002_yolo.jpg")
    print("\n🎨 색상:")
    print("   - 초록(Green): helmet")
    print("   - 주황(Orange): no_helmet")
    print("\n다음 단계:")
    print("1. 시각화 이미지 확인 → 라벨링 정확한지 확인")
    print("2. 결과가 정상이면 전체 변환 실행:")
    print("   python scripts/convert_voc_to_yolo.py")
    print("3. 문제가 있으면 보고해주세요!")

if __name__ == "__main__":
    BASE_DIR = Path.home() / "Developments" / "orin-yolo"
    VOC_DIR = BASE_DIR / "datasets" / "SHWD" / "VOC2028"

    if not VOC_DIR.exists():
        print(f"❌ VOC 디렉토리 없음: {VOC_DIR}")
        exit(1)

    test_conversion(VOC_DIR, num_samples=3)
