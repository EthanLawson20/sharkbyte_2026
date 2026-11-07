"""Filter already-extracted images without rerunning video extraction."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_hand_model(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".task.download")
    print(f"Downloading hand-detection model to {model_path}")
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, temporary_path)
        if temporary_path.stat().st_size == 0:
            raise RuntimeError("Downloaded hand-detection model is empty")
        temporary_path.replace(model_path)
    except OSError as error:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(
            f"Could not download the hand-detection model from {HAND_MODEL_URL}"
        ) from error
    return model_path


def frame_rejection_reasons(
    frame: Any,
    cv2: Any,
    hand_detector: Any,
    mp: Any,
    min_sharpness: float,
) -> list[str]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    reasons = []
    if sharpness < min_sharpness:
        reasons.append("blurry")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if hand_detector.detect(image).hand_landmarks:
        reasons.append("hand")
    return reasons


def paths_overlap(first: Path, second: Path) -> bool:
    try:
        second.relative_to(first)
        return True
    except ValueError:
        try:
            first.relative_to(second)
            return True
        except ValueError:
            return False


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_input_dir = script_dir / "data" / "frames"

    parser = argparse.ArgumentParser(
        description=(
            "Filter extracted images, copying blurry or hand-occluded frames "
            "to a separate review folder."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help=f"Folder containing extracted images (default: {default_input_dir})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Folder for accepted images (default: <input-dir>_filtered)",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=2.0,
        help=(
            "Reject images whose Laplacian-variance sharpness is below this "
            "threshold (default: 2.0)"
        ),
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        help=(
            "Path to a MediaPipe hand_landmarker.task file "
            "(downloaded automatically when omitted)"
        ),
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else input_dir.with_name(f"{input_dir.name}_filtered")
    )
    rejected_dir = output_dir.with_name(f"{output_dir.name}_rejected")

    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    if args.min_sharpness < 0:
        parser.error("--min-sharpness must be non-negative")
    if paths_overlap(input_dir, output_dir) or paths_overlap(input_dir, rejected_dir):
        parser.error("Output folders must be outside the input directory")
    if output_dir.exists() or rejected_dir.exists():
        parser.error(
            "Output folders already exist; choose a fresh --output-dir "
            "to keep results from separate runs from mixing"
        )

    images = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        parser.error(f"No supported image files found in: {input_dir}")

    try:
        import cv2
    except ImportError:
        print(
            "OpenCV is required. Install it with: python -m pip install opencv-python",
            file=sys.stderr,
        )
        return 1

    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError:
        print(
            "Hand filtering requires MediaPipe Tasks. Install it with: "
            "python -m pip install mediapipe",
            file=sys.stderr,
        )
        return 1

    model_path = (
        args.hand_model.resolve()
        if args.hand_model
        else Path.home()
        / ".cache"
        / "sharkbyte_2026"
        / "hand_landmarker.task"
    )

    try:
        model_path = ensure_hand_model(model_path)
        hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
            )
        )
        kept_count = 0
        rejected_count = 0
        corrupt_count = 0
        try:
            for image_path in images:
                relative_path = image_path.relative_to(input_dir)
                frame = cv2.imread(str(image_path))
                if frame is None:
                    reasons = ["unreadable"]
                    corrupt_count += 1
                else:
                    reasons = frame_rejection_reasons(
                        frame,
                        cv2,
                        hand_detector,
                        mp,
                        args.min_sharpness,
                    )

                destination_root = rejected_dir if reasons else output_dir
                destination = destination_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, destination)
                if reasons:
                    rejected_count += 1
                else:
                    kept_count += 1
        finally:
            hand_detector.close()
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Processed {len(images)} images: kept {kept_count}, "
        f"sent {rejected_count} to review"
    )
    if corrupt_count:
        print(f"Unreadable images copied to review: {corrupt_count}")
    print(f"Accepted images: {output_dir}")
    print(f"Rejected images for review: {rejected_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
