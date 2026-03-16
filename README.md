# T-LESS RT-DETR Detector for FoundationPose

Fine-tuned RT-DETR object detector for [T-LESS](https://cmp.felk.cvut.cz/t-less/) industrial objects,
intended to be used as the 2D detection front-end for
[NVIDIA Isaac ROS FoundationPose](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html).

## What this repo does

1. Downloads the T-LESS BOP dataset (PBR synthetic images)
2. Converts BOP annotations to COCO format for RT-DETR training
3. Fine-tunes RT-DETR (r50vd backbone) on T-LESS objects (30 classes)
4. Exports the trained model to ONNX with the tensor names expected by `isaac_ros_rtdetr`

## Requirements

- Training: run on **pfcalcul LAAS-CNRS** (see `cluster/`)
- ONNX → TRT conversion: run inside the **Isaac ROS Docker container** on your robot machine
- Local dev (optional): conda environment in `environment.yml`

## Quick start

See `cluster/README_cluster.md` for the full step-by-step guide.

## Repository structure

```
tless_detector/
├── environment.yml              # local conda env (optional, for dev)
├── data/
│   └── download_tless.py        # run on pfcalcul frontal
├── scripts/
│   ├── prepare_dataset.py       # BOP PBR → COCO JSON
│   ├── verify_dataset.py        # visual sanity check (JupyterLab)
│   ├── export_onnx.py           # ONNX export + tensor name fix
│   └── convert_meshes.py        # T-LESS .ply → centered .obj
├── configs/
│   └── rtdetr_r50vd_tless.yml   # RT-DETR training config
├── cluster/
│   ├── tless_detector.def       # Apptainer image definition
│   ├── train.sbatch             # SLURM job script (pfcalcul-specific)
│   └── README_cluster.md        # step-by-step cluster guide
└── third_party/
    └── RT-DETR/                 # git submodule (lyuwenyu/RT-DETR)
```

## After training: Isaac ROS integration

```bash
# On the Isaac ROS machine, inside the container:
/usr/src/tensorrt/bin/trtexec \
    --onnx=tless_rtdetr.onnx \
    --saveEngine=tless_rtdetr.plan \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640 \
    --fp16

# Launch FoundationPose with your trained detector:
ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
    launch_fragments:=realsense_mono_rect_depth,foundationpose_tracking \
    mesh_file_path:=<path_to_obj_XXXXXX.obj> \
    score_engine_file_path:=score_trt_engine.plan \
    refine_engine_file_path:=refine_trt_engine.plan \
    rt_detr_engine_file_path:=tless_rtdetr.plan \
    interface_specs_file:=${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_foundationpose/quickstart_interface_specs.json
```

## Credits

- [RT-DETR](https://github.com/lyuwenyu/RT-DETR) by lyuwenyu
- [T-LESS dataset](https://cmp.felk.cvut.cz/t-less/) by Hodaň et al.
- [BOP Benchmark](https://bop.felk.cvut.cz/)
