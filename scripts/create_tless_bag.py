"""
Create a ROS2 rosbag from a single RGB image for offline FoundationPose testing.

Topics written
--------------
  image_rect        sensor_msgs/msg/Image       640×480  rgb8
  camera_info_rect  sensor_msgs/msg/CameraInfo
  depth             sensor_msgs/msg/Image       640×480  32FC1 (metres)

Usage
-----
  python scripts/create_tless_bag.py --image data/ima_b21d9f4.jpg

Options
-------
  --image PATH       Input image (any format OpenCV can read)
  --out-dir PATH     Output bag directory  [default: data/tless_test_bag]
  --depth FLOAT      Constant synthetic depth in metres  [default: 0.5]
  --frames INT       Number of frames to write  [default: 60 = 6 s at 10 Hz]
  --fps FLOAT        Frame rate  [default: 10.0]
  --width INT        Output width   [default: 640]
  --height INT       Output height  [default: 480]

  Camera intrinsics (default: T-LESS Primesense Carmine 1.09 at 640×480)
  --fx FLOAT  --fy FLOAT  --cx FLOAT  --cy FLOAT

  --crop             Center-crop to output aspect ratio before resizing
                     (recommended for portrait phone photos)

IMPORTANT LIMITATIONS
---------------------
- A constant synthetic depth is used. The pipeline will run end-to-end but
  pose estimates will only be geometrically meaningful with real depth data
  (e.g. from a RealSense D4xx) and matching camera intrinsics.
- Default intrinsics are from the T-LESS Primesense sensor, not a phone camera.
  Pass --fx/--fy/--cx/--cy if you know your camera's actual calibration.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import rclpy
import rosbag2_py
from builtin_interfaces.msg import Time
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header


def make_header(stamp_ns: int, frame_id: str = 'camera') -> Header:
    h = Header()
    h.stamp = Time(sec=int(stamp_ns // 1_000_000_000),
                   nanosec=int(stamp_ns % 1_000_000_000))
    h.frame_id = frame_id
    return h


def make_rgb_msg(rgb: np.ndarray, stamp_ns: int) -> Image:
    msg = Image()
    msg.header = make_header(stamp_ns)
    msg.height, msg.width = rgb.shape[:2]
    msg.encoding = 'rgb8'
    msg.is_bigendian = False
    msg.step = msg.width * 3
    msg.data = rgb.tobytes()
    return msg


def make_depth_msg(height: int, width: int, depth_m: float, stamp_ns: int) -> Image:
    arr = np.full((height, width), fill_value=depth_m, dtype=np.float32)
    msg = Image()
    msg.header = make_header(stamp_ns)
    msg.height = height
    msg.width = width
    msg.encoding = '32FC1'
    msg.is_bigendian = False
    msg.step = width * 4
    msg.data = arr.tobytes()
    return msg


def make_camera_info_msg(height: int, width: int, stamp_ns: int,
                         fx: float, fy: float,
                         cx: float, cy: float) -> CameraInfo:
    msg = CameraInfo()
    msg.header = make_header(stamp_ns)
    msg.height = height
    msg.width = width
    msg.distortion_model = 'plumb_bob'
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [fx,  0.0, cx,
             0.0, fy,  cy,
             0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]
    msg.p = [fx,  0.0, cx,  0.0,
             0.0, fy,  cy,  0.0,
             0.0, 0.0, 1.0, 0.0]
    return msg


def center_crop_to_aspect(bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Crop the center of bgr to match the target_w:target_h aspect ratio."""
    h, w = bgr.shape[:2]
    target_ar = target_w / target_h
    current_ar = w / h

    if current_ar > target_ar:
        # Image is wider than target — crop horizontally
        new_w = int(h * target_ar)
        x0 = (w - new_w) // 2
        return bgr[:, x0:x0 + new_w]
    else:
        # Image is taller than target — crop vertically
        new_h = int(w / target_ar)
        y0 = (h - new_h) // 2
        return bgr[y0:y0 + new_h, :]


def main():
    parser = argparse.ArgumentParser(
        description='Create a rosbag from a single RGB image for FoundationPose.'
    )
    parser.add_argument('--image', type=Path, required=True,
                        help='Input RGB image')
    parser.add_argument('--out-dir', type=Path,
                        default=Path('data/tless_test_bag'),
                        help='Output bag directory (default: data/tless_test_bag)')
    parser.add_argument('--depth', type=float, default=0.5,
                        help='Constant synthetic depth in metres (default: 0.5)')
    parser.add_argument('--frames', type=int, default=60,
                        help='Number of frames to write (default: 60)')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='Frame rate in Hz (default: 10.0)')
    parser.add_argument('--width', type=int, default=640,
                        help='Output image width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Output image height (default: 480)')
    # T-LESS Primesense Carmine 1.09 intrinsics at 640×480
    parser.add_argument('--fx', type=float, default=1075.65)
    parser.add_argument('--fy', type=float, default=1073.90)
    parser.add_argument('--cx', type=float, default=324.38)
    parser.add_argument('--cy', type=float, default=242.04)
    parser.add_argument('--crop', action='store_true',
                        help='Center-crop to output aspect ratio before resizing '
                             '(recommended for portrait phone photos)')
    args = parser.parse_args()

    rclpy.init()

    # Load image
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        print(f'ERROR: Cannot read image: {args.image}')
        return

    orig_h, orig_w = bgr.shape[:2]
    print(f'Input image:  {orig_w}×{orig_h}')

    if args.crop:
        bgr = center_crop_to_aspect(bgr, args.width, args.height)
        print(f'After crop:   {bgr.shape[1]}×{bgr.shape[0]}')

    bgr = cv2.resize(bgr, (args.width, args.height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f'After resize: {args.width}×{args.height}')

    orig_ar = orig_w / orig_h
    target_ar = args.width / args.height
    if abs(orig_ar - target_ar) > 0.1 and not args.crop:
        print(f'WARNING: aspect ratio mismatch ({orig_ar:.2f} vs {target_ar:.2f}). '
              f'Use --crop to avoid distortion.')

    # Open bag writer
    # rosbag2_py creates the directory itself and refuses pre-existing ones.
    out_dir = args.out_dir
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
        print(f'Removed existing bag: {out_dir}')

    writer = rosbag2_py.SequentialWriter()
    storage_opts = rosbag2_py.StorageOptions(uri=str(out_dir), storage_id='sqlite3')
    converter_opts = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )
    writer.open(storage_opts, converter_opts)

    for topic_id, (topic_name, msg_type) in enumerate([
        ('image_rect',       'sensor_msgs/msg/Image'),
        ('camera_info_rect', 'sensor_msgs/msg/CameraInfo'),
        ('depth',            'sensor_msgs/msg/Image'),
    ]):
        writer.create_topic(rosbag2_py.TopicMetadata(
            id=topic_id,
            name=topic_name,
            type=msg_type,
            serialization_format='cdr',
        ))

    interval_ns = int(1e9 / args.fps)
    t0_ns = 1_714_425_714_000_000_000  # same epoch as NVIDIA quickstart bags

    print(f'Writing {args.frames} frames at {args.fps} Hz '
          f'(~{args.frames / args.fps:.0f} s) ...')

    for i in range(args.frames):
        stamp_ns = t0_ns + i * interval_ns

        rgb_msg   = make_rgb_msg(rgb, stamp_ns)
        depth_msg = make_depth_msg(args.height, args.width, args.depth, stamp_ns)
        ci_msg    = make_camera_info_msg(
            args.height, args.width, stamp_ns,
            args.fx, args.fy, args.cx, args.cy)

        writer.write('image_rect',       serialize_message(rgb_msg),   stamp_ns)
        writer.write('camera_info_rect', serialize_message(ci_msg),    stamp_ns)
        writer.write('depth',            serialize_message(depth_msg), stamp_ns)

    del writer  # flush + close
    rclpy.shutdown()

    print(f'\nBag written to: {out_dir.resolve()}')
    print(f'  {args.frames} frames  |  {args.fps} Hz  |  depth={args.depth} m  '
          f'|  {args.width}×{args.height}')
    print()
    print('Next: launch FoundationPose (after trtexec conversion — see README):')
    print(f'  ros2 launch {Path(__file__).resolve().parent.parent}/launch/tless_foundationpose.launch.py \\')
    print(f'    bag_path:={out_dir.resolve()} \\')
    print(f'    object_id:=1 \\')
    print(f'    rt_detr_engine_file_path:=${{ISAAC_ROS_WS}}/isaac_ros_assets/models/tless/tless_rtdetr.plan')


if __name__ == '__main__':
    main()
