#!/usr/bin/env python3
"""
IoU 트래킹 결과(프레임/프롬프트/트랙별 .npy 마스크)를 원본 이미지 위에 오버레이해서
프레임별 PNG를 저장하는 스크립트.

입력:
  - --images_dir: 원본 프레임 이미지 디렉토리 (예: images_4)
  - --masks_dir: 트래킹 결과 디렉토리 (예: images_4/sam3_label_masks_tracks)
    내부 구조 예시:
      masks_dir/
        prompt_00_house/
          frame_0000_track_000.npy
          frame_0001_track_000.npy
          frame_0001_track_001.npy
          ...
        prompt_01_tree/
          frame_0000_track_000.npy
          ...
        label_map_all_prompts.json
  - --output_dir: 오버레이 PNG를 저장할 곳 (기본: masks_dir/vis_overlays)

출력:
  - output_dir/frame_0000_overlay.png, frame_0001_overlay.png, ...
"""

import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def color_for_track(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed=track_id)
    return tuple(rng.integers(30, 225, size=3).tolist())


def load_image_paths(images_dir: Path) -> List[Path]:
    """images_dir 안의 이미지 파일들을 프레임 순서대로 정렬해서 반환."""
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    image_files = sorted(images_dir.glob("*.JPG")) + sorted(images_dir.glob("*.jpg")) + \
                  sorted(images_dir.glob("*.PNG")) + sorted(images_dir.glob("*.png"))
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    return image_files


def load_label_map(masks_dir: Path) -> Dict[int, str]:
    label_map_path = masks_dir / "label_map_all_prompts.json"
    if label_map_path.exists():
        with open(label_map_path, "r") as f:
            data = json.load(f)
        # 키가 문자열일 수 있으므로 int로 변환
        return {int(k): v for k, v in data.items()}
    return {}


def gather_prompt_dirs(masks_dir: Path) -> List[Path]:
    return sorted([p for p in masks_dir.iterdir() if p.is_dir() and p.name.startswith("prompt_")])


def gather_masks_for_frame(prompt_dir: Path, frame_idx: int) -> List[Tuple[int, np.ndarray]]:
    pattern = prompt_dir / f"frame_{frame_idx:04d}_track_*.npy"
    paths = sorted(glob.glob(str(pattern)))
    masks = []
    for p in paths:
        track_str = Path(p).stem.split("_")[-1]  # track_XXX
        try:
            track_id = int(track_str)
        except Exception:
            continue
        mask = np.load(p).astype(bool)
        masks.append((track_id, mask))
    return masks


def overlay_frame(
    image_path: Path,
    prompt_dirs: List[Path],
    frame_idx: int,
    label_map: Dict[int, str],
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    base = np.array(img, dtype=np.float32)
    overlay = base.copy()
    draw = ImageDraw.Draw(img)

    for prompt_dir in prompt_dirs:
        # prompt dir 이름에서 label_id 추출: prompt_00_house
        name_parts = prompt_dir.name.split("_")
        if len(name_parts) < 2:
            continue
        try:
            label_id = int(name_parts[1])
        except Exception:
            continue
        prompt_name = label_map.get(label_id, prompt_dir.name)

        masks = gather_masks_for_frame(prompt_dir, frame_idx)
        for track_id, mask in masks:
            color = np.array(color_for_track(track_id), dtype=np.float32)
            if mask.shape[:2] != overlay.shape[:2]:
                # 리사이즈 (nearest)
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_img = mask_img.resize((overlay.shape[1], overlay.shape[0]), resample=Image.NEAREST)
                mask = np.array(mask_img, dtype=bool)
            overlay[mask] = overlay[mask] * 0.4 + color * 0.6

            # 중심에 텍스트: label/track
            ys, xs = np.nonzero(mask)
            if len(xs) > 0:
                x = int(xs.mean())
                y = int(ys.mean())
                text = f"{prompt_name}/T{track_id}"
                draw.text((x, y), text, fill=(255, 255, 255))

    result = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 IoU 트래킹 결과를 프레임별 오버레이 PNG로 시각화"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="원본 이미지 디렉토리 (예: .../images_4)",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        required=True,
        help="트래킹 마스크 디렉토리 (예: .../sam3_label_masks_tracks)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="오버레이 PNG 저장 경로 (기본: masks_dir/vis_overlays)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="시각화할 최대 프레임 수 (기본: 전체)",
    )

    args = parser.parse_args()

    images_dir = Path(args.images_dir).resolve()
    masks_dir = Path(args.masks_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else masks_dir / "vis_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = load_image_paths(images_dir)
    if args.max_frames is not None:
        image_files = image_files[: args.max_frames]

    label_map = load_label_map(masks_dir)
    prompt_dirs = gather_prompt_dirs(masks_dir)
    if not prompt_dirs:
        print("프롬프트 디렉토리가 없습니다. (prompt_XX_...)")
        return

    print(f"이미지 {len(image_files)}장, 프롬프트 디렉토리 {len(prompt_dirs)}개")
    for frame_idx, img_path in enumerate(image_files):
        print(f"프레임 {frame_idx:04d} 시각화 중... ({img_path.name})")
        overlay_img = overlay_frame(img_path, prompt_dirs, frame_idx, label_map)
        out_path = output_dir / f"frame_{frame_idx:04d}_overlay.png"
        overlay_img.save(out_path)
    print(f"완료. 출력: {output_dir}")


if __name__ == "__main__":
    main()


