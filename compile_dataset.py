"""Extract every frame from the videos in the dataset's data directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4"}


def extract_frames(video_path: Path, output_dir: Path, cv2: Any) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_number = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            image_path = output_dir / f"frame_{frame_number:06d}.jpg"
            if not cv2.imwrite(
                str(image_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise RuntimeError(f"Could not write image: {image_path}")
            frame_number += 1
    finally:
        capture.release()

    if frame_number == 0:
        raise RuntimeError(f"No frames could be read from video: {video_path}")
    return frame_number


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir / "sharkbyte_2026" / "data"
    if not default_data_dir.is_dir():
        default_data_dir = script_dir / "data"

    parser = argparse.ArgumentParser(
        description="Extract every video frame as a separate JPEG image."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_data_dir,
        help=f"Folder containing videos (default: {default_data_dir})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Folder for extracted images (default: <input-dir>/frames)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir else input_dir / "frames"
    )
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    videos = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        parser.error(f"No supported video files found in: {input_dir}")

    try:
        import cv2
    except ImportError:
        print(
            "OpenCV is required. Install it with: python -m pip install opencv-python",
            file=sys.stderr,
        )
        return 1

    try:
        for video_path in videos:
            count = extract_frames(video_path, output_dir / video_path.stem, cv2)
            print(f"{video_path.name}: extracted {count} frames")
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())