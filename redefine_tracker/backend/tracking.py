import os
import cv2
import ffmpeg
from utils import draw_mask, clear_folder, zip_folder

def track_objects_in_video(tracker_state, frame_num, input_video):
    """Tracks objects in the input video starting from the given frame number."""
    output_dir = 'output_frames'
    output_masks_dir = 'output_masks'
    output_combined_dir = 'output_combined'
    output_video_path = 'output_video.mp4'
    output_zip_path = 'output_masks.zip'

    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    video_segments = {}
    predictor, inference_state, image_predictor = tracker_state

    start_frame_idx = frame_num
    max_frame_num_to_track = 60  # Adjustable tracking range

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=start_frame_idx, max_frame_num_to_track=max_frame_num_to_track):
        
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    for frame_file in frame_files:
        frame_idx = int(os.path.splitext(frame_file)[0])
        if start_frame_idx <= frame_idx < start_frame_idx + max_frame_num_to_track:
            frame_path = os.path.join(output_dir, frame_file)
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masked_frame = image.copy()

            if frame_idx in video_segments:
                for obj_id, mask in video_segments[frame_idx].items():
                    masked_frame = draw_mask(mask, image=masked_frame, obj_id=obj_id)
                    mask_output_path = os.path.join(output_masks_dir, f'{obj_id}_{frame_idx:07d}.png')
                    cv2.imwrite(mask_output_path, draw_mask(mask))

            combined_output_path = os.path.join(output_combined_dir, f'{frame_idx:07d}.png')
            combined_image_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(combined_output_path, combined_image_bgr)

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    output_frames = len([name for name in os.listdir(output_combined_dir) if name.endswith('.png')])
    out_fps = fps * output_frames / total_frames
    ffmpeg.input(os.path.join(output_combined_dir, '%07d.png'), framerate=out_fps).output(output_video_path, vcodec='h264_nvenc', pix_fmt='yuv420p').run()
    zip_folder(output_masks_dir, output_zip_path)

    return masked_frame, masked_frame, output_video_path, output_video_path, output_zip_path
