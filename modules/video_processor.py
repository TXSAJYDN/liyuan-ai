"""
视频处理模块：基于 FFmpeg 和 OpenCV 实现视频切片与关键帧抽取
"""
import os
import subprocess
import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from configs.settings import (
    PROCESSED_DIR, KEYFRAME_DIR,
    VIDEO_SEGMENT_DURATION, KEYFRAME_INTERVAL,
    SCENE_CHANGE_THRESHOLD, KEYFRAME_SIZE,
    OPERA_DATA_DIR
)

logger = logging.getLogger(__name__)


class VideoProcessor:
    """视频处理器：切片、抽帧、管理"""

    def __init__(self):
        self.processed_dir = PROCESSED_DIR
        self.keyframe_dir = KEYFRAME_DIR

    def get_video_info(self, video_path: str) -> dict:
        """获取视频基本信息"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        }
        cap.release()
        return info

    def segment_video(self, video_path: str, output_dir: Optional[str] = None,
                      segment_duration: int = VIDEO_SEGMENT_DURATION) -> List[str]:
        """使用 FFmpeg 将视频切分为固定时长的片段"""
        video_path = Path(video_path)
        video_name = video_path.stem
        if output_dir is None:
            output_dir = self.processed_dir / video_name
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = str(output_dir / f"{video_name}_seg_%04d.mp4")
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-c", "copy", "-map", "0",
            "-segment_time", str(segment_duration),
            "-f", "segment", "-reset_timestamps", "1",
            output_pattern, "-y"
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg 切片失败: {e.stderr}")
            raise RuntimeError(f"视频切片失败: {e.stderr}")
        segments = sorted(output_dir.glob(f"{video_name}_seg_*.mp4"))
        logger.info(f"视频 {video_name} 切分为 {len(segments)} 个片段")
        return [str(s) for s in segments]

    def extract_keyframes(self, video_path: str, output_dir: Optional[str] = None,
                          interval: float = KEYFRAME_INTERVAL,
                          use_scene_detection: bool = False) -> List[dict]:
        """从视频中抽取关键帧"""
        video_path = Path(video_path)
        video_name = video_path.stem
        if output_dir is None:
            output_dir = self.keyframe_dir / video_name
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        frame_interval = int(fps * interval)

        keyframes = []
        frame_idx = 0
        prev_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            should_extract = False
            if use_scene_detection and prev_frame is not None:
                diff = cv2.absdiff(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                )
                mean_diff = np.mean(diff)
                if mean_diff > SCENE_CHANGE_THRESHOLD:
                    should_extract = True
            if frame_idx % frame_interval == 0:
                should_extract = True
            if should_extract:
                timestamp = frame_idx / fps
                resized = cv2.resize(frame, KEYFRAME_SIZE)
                frame_filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
                keyframes.append({
                    "path": str(frame_path),
                    "timestamp": round(timestamp, 2),
                    "frame_idx": frame_idx,
                })
            prev_frame = frame.copy() if use_scene_detection else None
            frame_idx += 1
        cap.release()
        logger.info(f"从 {video_name} 抽取了 {len(keyframes)} 个关键帧")
        return keyframes

    def process_video(self, video_path: str,
                      segment_duration: int = VIDEO_SEGMENT_DURATION,
                      keyframe_interval: float = KEYFRAME_INTERVAL,
                      use_scene_detection: bool = False) -> dict:
        """完整的视频处理流水线：切片 + 抽帧"""
        logger.info(f"开始处理视频: {video_path}")
        video_info = self.get_video_info(video_path)
        segments = self.segment_video(video_path, segment_duration=segment_duration)
        all_keyframes = []
        for seg_path in segments:
            kfs = self.extract_keyframes(
                seg_path, interval=keyframe_interval,
                use_scene_detection=use_scene_detection
            )
            all_keyframes.extend(kfs)
        result = {
            "video_path": str(video_path),
            "video_info": video_info,
            "segments": segments,
            "keyframes": all_keyframes,
        }
        logger.info(f"视频处理完成: {len(segments)} 片段, {len(all_keyframes)} 关键帧")
        return result

    @staticmethod
    def list_opera_videos(genre: Optional[str] = None) -> List[dict]:
        """列出原始戏曲数据目录中的视频（只读访问）"""
        videos = []
        if genre:
            search_dirs = [OPERA_DATA_DIR / genre]
        else:
            search_dirs = [d for d in OPERA_DATA_DIR.iterdir() if d.is_dir()]
        for d in search_dirs:
            if not d.exists():
                continue
            genre_name = d.name
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".mp4", ".avi", ".mkv", ".flv", ".mov"):
                    txt_file = f.with_suffix(".txt")
                    videos.append({
                        "path": str(f),
                        "genre": genre_name,
                        "name": f.stem,
                        "txt_path": str(txt_file) if txt_file.exists() else None,
                    })
        return videos


video_processor = VideoProcessor()
