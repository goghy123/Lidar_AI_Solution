#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import cv2
import imageio
from pathlib import Path


# =========================
# 默认参数：终端不指定时使用这里
# =========================

DEFAULT_INPUT_DIR = "outputs/scene-0"
DEFAULT_OUTPUT_DIR = "outputs/video"

# "gif" 或 "video"
DEFAULT_OUTPUT_TYPE = "video"

DEFAULT_OUTPUT_NAME = "scene-0"
DEFAULT_FPS = 7
DEFAULT_IMAGE_EXT = ".jpg"

DEFAULT_RESIZE_TO_FIRST_FRAME = True
DEFAULT_VIDEO_CODEC = "mp4v"


def extract_frame_number(filename: str):
    """
    从 frame_000000.jpg 中提取编号。
    """
    match = re.search(r"frame_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def collect_image_files(input_dir: str, image_ext: str):
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

    image_files = []

    for file in input_path.iterdir():
        if file.is_file() and file.suffix.lower() == image_ext.lower():
            frame_num = extract_frame_number(file.name)
            if frame_num is not None:
                image_files.append((frame_num, file))

    image_files.sort(key=lambda x: x[0])

    return [file for _, file in image_files]


def make_video(
    image_files,
    output_path: str,
    fps: int,
    resize_to_first_frame: bool,
    video_codec: str,
):
    first_frame = cv2.imread(str(image_files[0]))

    if first_frame is None:
        raise RuntimeError(f"无法读取第一张图片: {image_files[0]}")

    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*video_codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")

    written_count = 0

    for img_path in image_files:
        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"警告：跳过无法读取的图片: {img_path}")
            continue

        if resize_to_first_frame and frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)
        written_count += 1

    writer.release()

    print(f"视频已生成: {output_path}")
    print(f"实际写入帧数: {written_count}")


def make_gif(
    image_files,
    output_path: str,
    fps: int,
    resize_to_first_frame: bool,
):
    frames = []

    first_frame = cv2.imread(str(image_files[0]))

    if first_frame is None:
        raise RuntimeError(f"无法读取第一张图片: {image_files[0]}")

    height, width = first_frame.shape[:2]

    for img_path in image_files:
        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"警告：跳过无法读取的图片: {img_path}")
            continue

        if resize_to_first_frame and frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    if not frames:
        raise RuntimeError("没有成功读取任何图片，无法生成 GIF")

    duration = 1.0 / fps
    imageio.mimsave(output_path, frames, duration=duration)

    print(f"GIF 已生成: {output_path}")
    print(f"实际写入帧数: {len(frames)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 frame_000000.jpg 这类连续图片拼接成 GIF 或视频"
    )

    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"输入图片文件夹，默认: {DEFAULT_INPUT_DIR}",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出文件夹，默认: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--output-type",
        choices=["gif", "video"],
        default=DEFAULT_OUTPUT_TYPE,
        help=f"输出类型：gif 或 video，默认: {DEFAULT_OUTPUT_TYPE}",
    )

    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"输出文件名，不需要后缀，默认: {DEFAULT_OUTPUT_NAME}",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"帧率，默认: {DEFAULT_FPS}",
    )

    parser.add_argument(
        "--image-ext",
        default=DEFAULT_IMAGE_EXT,
        help=f"图片后缀，例如 .jpg、.png，默认: {DEFAULT_IMAGE_EXT}",
    )

    parser.add_argument(
        "--video-codec",
        default=DEFAULT_VIDEO_CODEC,
        help=f"视频编码器，默认: {DEFAULT_VIDEO_CODEC}",
    )

    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="不自动缩放图片到第一张图片的尺寸",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_type = args.output_type
    output_name = args.output_name
    fps = args.fps
    image_ext = args.image_ext
    video_codec = args.video_codec
    resize_to_first_frame = not args.no_resize

    os.makedirs(output_dir, exist_ok=True)

    image_files = collect_image_files(input_dir, image_ext)

    if not image_files:
        raise RuntimeError(
            f"没有找到符合命名规则的图片，例如: frame_000000{image_ext}"
        )

    print(f"输入文件夹: {input_dir}")
    print(f"输出文件夹: {output_dir}")
    print(f"输出类型: {output_type}")
    print(f"帧率: {fps}")
    print(f"图片后缀: {image_ext}")
    print(f"是否自动缩放: {resize_to_first_frame}")
    print(f"共找到 {len(image_files)} 张图片")
    print(f"第一张: {image_files[0].name}")
    print(f"最后一张: {image_files[-1].name}")

    if output_type == "video":
        output_path = os.path.join(output_dir, f"{output_name}.mp4")
        make_video(
            image_files=image_files,
            output_path=output_path,
            fps=fps,
            resize_to_first_frame=resize_to_first_frame,
            video_codec=video_codec,
        )

    elif output_type == "gif":
        output_path = os.path.join(output_dir, f"{output_name}.gif")
        make_gif(
            image_files=image_files,
            output_path=output_path,
            fps=fps,
            resize_to_first_frame=resize_to_first_frame,
        )


if __name__ == "__main__":
    main()
