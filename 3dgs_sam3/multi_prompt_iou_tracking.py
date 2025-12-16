#!/usr/bin/env python3
"""
SAM3 텍스트 프롬프트 기반 프레임 단위 세그멘테이션 + IoU 매칭 트래킹

- 모든 프레임에 대해 sam3_inference (이미지 API)로 프롬프트별 인스턴스 마스크를 얻는다.
- 프레임 간 IoU 매칭으로 트랙 ID를 이어붙여 "간단한" 트래킹을 수행한다.
- 각 프레임/프롬프트/트랙별로 마스크를 개별 .npy 파일로 저장한다.

저장 예시:
  frame_0000_prompt_house_track_000.npy
  frame_0000_prompt_house_track_001.npy
  frame_0001_prompt_house_track_000.npy  # 프레임0의 track_000과 매칭된 경우
  frame_0001_prompt_house_track_002.npy  # 새로 생성된 트랙 (매칭 실패한 새 인스턴스)

라벨/프롬프트 매핑:
  text_prompts[0] -> prompt "house"
  text_prompts[1] -> prompt "tree"
  ...

트랙 ID는 프롬프트별로 독립된 카운터를 가진다:
  house_track_000, house_track_001, ...
  tree_track_000, tree_track_001, ...
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import pycocotools.mask as maskUtils


def load_image_paths(images_dir: Path) -> List[Path]:
    """images_dir 안의 이미지 파일들을 프레임 순서대로 정렬해서 반환."""
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    image_files = sorted(images_dir.glob("*.JPG")) + sorted(images_dir.glob("*.jpg")) + \
                  sorted(images_dir.glob("*.PNG")) + sorted(images_dir.glob("*.png"))
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    return image_files


def build_sam3_processor(device: str):
    """SAM3 이미지 모델 + Processor 생성."""
    sam3_dir = Path("/home/.../sam3").resolve()
    if str(sam3_dir) not in sys.path:
        sys.path.insert(0, str(sam3_dir))

    from sam3.model_builder import build_sam3_image_model  # type: ignore[import]
    from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[import]

    bpe_path = sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    print("\n[1/3] SAM3 이미지 모델 로드 및 Processor 생성...")
    try:
        if bpe_path.exists():
            print(f"  > BPE 파일 경로: {bpe_path}")
            model = build_sam3_image_model(bpe_path=str(bpe_path))
        else:
            print("  > BPE 파일이 없어 기본 설정으로 모델을 빌드합니다.")
            model = build_sam3_image_model()

        model = model.to(device)
        processor = Sam3Processor(model, device=device, confidence_threshold=0.3)
        print("  > 이미지 모델 + Processor 준비 완료")
        return processor
    except Exception as e:
        print(f"  ✗ SAM3 이미지 모델 로드 실패: {e}")
        raise


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """두 바이너리 마스크의 IoU 계산."""
    assert mask_a.shape == mask_b.shape
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return inter / union


def match_tracks(
    prev_masks: List[np.ndarray],
    prev_track_ids: List[int],
    curr_masks: List[np.ndarray],
    iou_threshold: float = 0.3,
) -> Tuple[Dict[int, int], List[int]]:
    """
    단순 IoU 기반 매칭:
      - prev_masks 와 curr_masks 사이에 IoU를 계산해서 greedy 매칭
      - iou_threshold 이상인 것만 매칭
    Returns:
      match_map: {curr_idx -> matched_prev_track_id}
      unmatched_curr: 매칭 실패한 curr 인덱스 리스트
    """
    match_map: Dict[int, int] = {}
    unmatched_curr = list(range(len(curr_masks)))

    used_prev = set()
    # greedy: curr 마스크를 순회하며 prev와 IoU 최대인 것 선택 (단, threshold 이상)
    for c_idx, cm in enumerate(curr_masks):
        best_iou = 0.0
        best_prev_idx = None
        for p_idx, pm in enumerate(prev_masks):
            if p_idx in used_prev:
                continue
            iou = compute_mask_iou(pm, cm)
            if iou > best_iou:
                best_iou = iou
                best_prev_idx = p_idx
        if best_prev_idx is not None and best_iou >= iou_threshold:
            match_map[c_idx] = prev_track_ids[best_prev_idx]
            used_prev.add(best_prev_idx)
            unmatched_curr.remove(c_idx)
    return match_map, unmatched_curr


def run_iou_tracking_per_prompt(
    image_files: List[Path],
    prompt: str,
    label_id: int,
    processor,
    device: str,
    output_dir: Path,
    iou_threshold: float = 0.3,
):
    """
    특정 프롬프트에 대해 모든 프레임을 순회하면서:
      - sam3_inference로 인스턴스 마스크들 얻기
      - IoU로 이전 프레임 트랙과 매칭
      - 프레임/트랙별로 npy 저장
    """
    from sam3.agent.client_sam3 import sam3_inference, remove_overlapping_masks  # type: ignore[import]

    # 트랙 ID 카운터 (프롬프트별로 독립)
    next_track_id = 0
    # 이전 프레임 정보
    prev_masks: List[np.ndarray] = []
    prev_track_ids: List[int] = []

    prompt_dir = output_dir / f"prompt_{label_id:02d}_{prompt.replace(' ', '_')}"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    for frame_id, img_path in enumerate(image_files):
        print(f"  [prompt '{prompt}'] 프레임 {frame_id:04d}: {img_path.name}")

        # SAM3 inference
        outputs = sam3_inference(processor, str(img_path), prompt)
        outputs = remove_overlapping_masks(outputs)

        pred_masks_rle = outputs.get("pred_masks", [])
        pred_scores = outputs.get("pred_scores", [])
        h = outputs.get("orig_img_h")
        w = outputs.get("orig_img_w")

        curr_masks: List[np.ndarray] = []
        if pred_masks_rle and h is not None and w is not None:
            print(f"    > {len(pred_masks_rle)}개 마스크, 점수 상위 3개: "
                  f"{[f'{s:.3f}' for s in pred_scores[:3]] if pred_scores else 'N/A'}")
            for rle_mask, score in zip(pred_masks_rle, pred_scores):
                rle_obj = {"size": [h, w], "counts": rle_mask}
                mask = maskUtils.decode(rle_obj).astype(bool)
                if mask.sum() == 0:
                    continue
                curr_masks.append(mask)
        else:
            print("    > 마스크 없음")

        # 매칭
        match_map = {}
        unmatched_curr: List[int] = []

        if prev_masks and curr_masks:
            match_map, unmatched_curr = match_tracks(
                prev_masks, prev_track_ids, curr_masks, iou_threshold=iou_threshold
            )
            # 매칭된 것 저장
            for c_idx, track_id in match_map.items():
                save_path = prompt_dir / f"frame_{frame_id:04d}_track_{track_id:03d}.npy"
                np.save(save_path, curr_masks[c_idx])
            # 매칭 안 된 현재 마스크는 새 트랙 생성
            for c_idx in unmatched_curr:
                track_id = next_track_id
                next_track_id += 1
                save_path = prompt_dir / f"frame_{frame_id:04d}_track_{track_id:03d}.npy"
                np.save(save_path, curr_masks[c_idx])
                match_map[c_idx] = track_id  # 새 트랙 ID 기록
        else:
            # 이전이 없거나 현재가 없으면 새 트랙 생성 (현재 마스크 전부)
            for c_idx, cm in enumerate(curr_masks):
                track_id = next_track_id
                next_track_id += 1
                save_path = prompt_dir / f"frame_{frame_id:04d}_track_{track_id:03d}.npy"
                np.save(save_path, cm)
                match_map[c_idx] = track_id
            unmatched_curr = []

        # 현재를 다음 프레임의 이전 상태로 (curr 순서대로 track_id를 기록)
        prev_masks = curr_masks
        prev_track_ids = [match_map.get(i, -1) for i in range(len(curr_masks))]

    # 완료 후 label_map.json 저장
    import json

    label_map_path = output_dir / "label_map.json"
    label_map = {"prompt": prompt, "label_id": label_id}
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"\n  [prompt '{prompt}'] 라벨 매핑 정보 저장: {label_map_path}")


def run_iou_tracking(
    images_dir: Path,
    output_dir: Path,
    text_prompts: List[str],
    device: str = "cuda",
    iou_threshold: float = 0.3,
):
    # 디바이스 설정
    if device == "cuda" and torch.cuda.is_available():
        print("  > 디바이스: cuda")
    else:
        device = "cpu"
        print("  > CUDA 사용 불가 또는 요청되지 않음, cpu 사용")
    if device == "cuda":
        print(f"  > GPU: {torch.cuda.get_device_name(0)}")

    # 이미지 목록
    image_files = load_image_paths(images_dir)
    num_frames = len(image_files)
    print(f"\n총 {num_frames}개 프레임 발견")

    # SAM3 Processor (한 번만 로드, 프롬프트 별로 재사용)
    processor = build_sam3_processor(device)

    # 출력 디렉토리 준비
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[2/3] 프롬프트별 IoU 트래킹 실행...")
    for label_id, prompt in enumerate(text_prompts):
        print(f"\n=== 프롬프트 '{prompt}' (label {label_id}) 트래킹 시작 ===")
        run_iou_tracking_per_prompt(
            image_files=image_files,
            prompt=prompt,
            label_id=label_id,
            processor=processor,
            device=device,
            output_dir=output_dir,
            iou_threshold=iou_threshold,
        )

    # 전체 라벨 매핑 저장
    import json

    label_map = {label_id: prompt for label_id, prompt in enumerate(text_prompts)}
    with open(output_dir / "label_map_all_prompts.json", "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"\n[3/3] 전체 라벨 매핑 정보 저장: {output_dir / 'label_map_all_prompts.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 텍스트 프롬프트 기반 프레임 단위 세그멘테이션 + IoU 매칭 트래킹"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/home/.../gaussian-splatting/data/colmap/gerrard-hall/images_4",
        help="프레임 이미지들이 들어있는 디렉토리 (예: .../images_4)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="트랙별 마스크를 저장할 디렉토리 (기본: <images_dir>/sam3_label_masks_tracks)",
    )
    parser.add_argument(
        "--text_prompts",
        type=str,
        nargs="+",
        required=True,
        help='텍스트 프롬프트 리스트 (예: --text_prompts "house" "tree" "car")',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="사용할 디바이스 (기본: cuda)",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.3,
        help="프레임 간 트래킹 매칭 IoU 임계값 (기본: 0.3)",
    )

    args = parser.parse_args()

    images_dir = Path(args.images_dir).resolve()
    if args.output_dir is None:
        output_dir = images_dir / "sam3_label_masks_tracks"
    else:
        output_dir = Path(args.output_dir).resolve()

    print("=" * 60)
    print("SAM3 프레임 단위 멀티 프롬프트 IoU 트래킹")
    print("=" * 60)
    print(f"Images dir : {images_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Device     : {args.device}")
    print(f"Prompts    : {args.text_prompts}")
    print(f"IoU thr    : {args.iou_threshold}")
    print("=" * 60)

    run_iou_tracking(
        images_dir=images_dir,
        output_dir=output_dir,
        text_prompts=args.text_prompts,
        device=args.device,
        iou_threshold=args.iou_threshold,
    )

    print("\n완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()


