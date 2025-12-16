# 3dgs_sam3

SAM3 텍스트 프롬프트 기반 프레임 단위 세그멘테이션 & 간단 IoU 트래킹 + 시각화

## 구성
- `multi_prompt_iou_tracking.py`  
  - 프레임마다 `sam3_inference`로 프롬프트별 인스턴스 마스크를 추출하고,  
  - 연속 프레임 간 IoU 매칭으로 트랙 ID를 이어서  
  - `prompt_xx_<name>/frame_YYYY_track_ZZZ.npy` 형태로 저장.
- `visualize_tracks_iou.py`  
  - 위에서 저장된 트랙별 마스크를 원본 이미지에 오버레이 PNG로 생성.

## 요구 사항
- conda env: `gaussian_splatting_sam3` (이미 사용 중인 환경)
- SAM3 패키지가 `/home/minjeong/sam3` 에 설치되어 있어야 함 (BPE 포함).

## 예시 경로
- 이미지 폴더: `/home/minjeong/gaussian-splatting/data/colmap/gerrard-hall/images_4`
- 출력 (트랙 마스크): `/home/minjeong/gaussian-splatting/data/colmap/gerrard-hall/images_4/sam3_label_masks_tracks`
- 출력 (오버레이): `/home/minjeong/gaussian-splatting/data/colmap/gerrard-hall/images_4/sam3_label_masks_tracks/vis_overlays`

## 사용법

### 1) 프레임 단위 세그멘테이션 + IoU 트래킹
```bash
cd /home/minjeong/gaussian-splatting/3dgs_sam3
conda run -n gaussian_splatting_sam3 python multi_prompt_iou_tracking.py \
  --images_dir /home/minjeong/gaussian-splatting/data/colmap/gerrard-hall/images_4 \
  --output_dir /home/minjeong/gaussian-splatting/data/colmap/gerrard-hall/images_4/sam3_label_masks_tracks \
  --text_prompts "house" "tree" \
  --device cuda \
  --iou_threshold 0.3
```
- 결과: `prompt_00_house/frame_0000_track_000.npy` … 식으로 트랙별 마스크 저장
- 라벨 매핑: `label_map_all_prompts.json`

### 2) 트랙 시각화 (모든 트랙 오버레이)
```bash
conda run -n gaussian_splatting_sam3 python visualize_tracks_iou.py \
  --images_dir /home/minjeong/gaussian-splatting/data/colmap/gerrard-hall/images_4 \
  --masks_dir /home/minjeong/gaussian-splatting/data/colmap/gerrard-hall/images_4/sam3_label_masks_tracks \
  --output_dir /home/minjeong/gaussian-splatting/data/colmap/gerrard-hall/images_4/sam3_label_masks_tracks/vis_overlays \
  --max_frames 5   # 원하는 만큼
```

### 3) 특정 트랙만 단독 오버레이 (필요 시)
아래처럼 Python 원라이너로 원하는 트랙만 추출 가능:
```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

images_dir = Path("/home/minjeong/gaussian-splatting/data/colmap/gerrard-hall/images_4")
masks_dir  = Path("/home/minjeong/gaussian-splatting/data/colmap/gerrard-hall/images_4/sam3_label_masks_tracks/prompt_01_tree")
out_dir    = Path("/home/minjeong/gaussian-splatting/data/colmap/gerrard-hall/images_4/sam3_label_masks_tracks/vis_single_track")
track_id   = "003"
max_frames = 6

out_dir.mkdir(parents=True, exist_ok=True)
for frame_idx in range(max_frames):
    img_path = images_dir / f"IMG_{2331+frame_idx}.JPG"
    mask_path = masks_dir / f"frame_{frame_idx:04d}_track_{track_id}.npy"
    if not (img_path.exists() and mask_path.exists()):
        continue
    img = Image.open(img_path).convert("RGB")
    mask = np.load(mask_path).astype(bool)
    if mask.shape[:2] != img.size[::-1]:
        mask = np.array(Image.fromarray(mask.astype(np.uint8)*255).resize(img.size, Image.NEAREST), dtype=bool)
    overlay = np.array(img, dtype=np.float32)
    color = np.array([255, 0, 0], dtype=np.float32)
    overlay[mask] = overlay[mask]*0.4 + color*0.6
    out = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    ys, xs = np.nonzero(mask)
    if len(xs) > 0:
        x, y = int(xs.mean()), int(ys.mean())
        draw = ImageDraw.Draw(out)
        draw.text((x, y), f"tree/T{track_id}", fill=(255, 255, 255))
    out.save(out_dir / f"frame_{frame_idx:04d}_track_{track_id}.png")
PY
```

## 노트
- 비디오 트래커를 쓰지 않고, 프레임별 디텍션 + IoU 매칭으로 단순 트랙을 만든 구조입니다.
- IoU 임계값은 `--iou_threshold`로 조절 가능합니다 (기본 0.3).


