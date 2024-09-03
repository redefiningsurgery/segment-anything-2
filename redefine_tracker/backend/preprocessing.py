import gc
import os
import cv2
import torch
import ffmpeg
from utils import clear_folder

def reset_app(tracker_state):
    """Resets the application state and clears GPU memory."""
    if tracker_state is not None:
        predictor, inference_state, image_predictor = tracker_state
        predictor.reset_state(inference_state)
        del predictor, inference_state, image_predictor, tracker_state
        gc.collect()
        torch.cuda.empty_cache()
    return None, ({}, {}), None, None, 0, None, None, None, 0

def preprocess_video(tracker_state, input_video, scale_slider, model_size):
    """Preprocesses the input video by extracting frames and setting up the tracker."""
    output_dir = 'output_frames'
    output_masks_dir = 'output_masks'
    output_combined_dir = 'output_combined'
    clear_folder(output_dir)
    clear_folder(output_masks_dir)
    clear_folder(output_combined_dir)

    if input_video is None:
        return reset_app(tracker_state)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    output_frames = int(total_frames * scale_slider)
    frame_interval = max(1, total_frames // output_frames)
    ffmpeg.input(input_video, hwaccel='cuda').output(
        os.path.join(output_dir, '%07d.jpg'), q=2, start_number=0, 
        vf=f'select=not(mod(n\,{frame_interval}))', vsync='vfr'
    ).run()

    first_frame_path = os.path.join(output_dir, '0000000.jpg')
    first_frame = cv2.imread(first_frame_path)
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    if tracker_state is not None:
        tracker_state = None
        gc.collect()
        torch.cuda.empty_cache()

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint_map = {
        "tiny": ("checkpoints/sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"),
        "small": ("checkpoints/sam2_hiera_small.pt", "sam2_hiera_s.yaml"),
        "base-plus": ("checkpoints/sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"),
        "large": ("checkpoints/sam2_hiera_large.pt", "sam2_hiera_l.yaml")
    }

    sam2_checkpoint, model_cfg = checkpoint_map[model_size]
    
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    image_predictor = SAM2ImagePredictor(sam2_model)
    inference_state = predictor.init_state(video_path=output_dir)
    predictor.reset_state(inference_state)
    return (predictor, inference_state, image_predictor), ({}, {}), first_frame_rgb, first_frame_rgb, 0, None, None, None, 0

