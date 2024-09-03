import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sam2.sam2_video_predictor import SAM2VideoPredictor

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
video_dir = "output_frames"
ann_frame_idx, ann_obj_id = 100, 1
points = np.array([[210, 250], [300, 600]], dtype=np.float32)
labels = np.array([1, 1], np.int32) # `1` means positive click and `0` means negative click

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state = predictor.init_state(video_path=video_dir)
    frame_idx, object_ids, mask_logits = predictor.add_new_points_or_box(inference_state=inference_state,
                                                                         frame_idx=ann_frame_idx,
                                                                         obj_id=ann_obj_id,
                                                                         points=points,
                                                                         labels=labels,
                                                                         clear_old_points=True)

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(inference_state=inference_state,
                                                                           reverse=False,
                                                                           start_frame_idx=50,
                                                                           max_frame_num_to_track=80):
        video_segments[frame_idx] = {
        out_obj_id: (mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(object_ids)
    }

# print(inference_state["num_frames"])
# print(inference_state["frames_already_tracked"])


frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

def apply_mask_to_image(base_image, mask, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

vis_frame_stride = 1
output_dir = 'output_masks'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for out_frame_idx in sorted(video_segments.keys()):
    frame_path = os.path.join(video_dir, frame_names[out_frame_idx])
    image = np.array(Image.open(frame_path), dtype=np.float32)  # Ensure image is in float format to blend properly
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        mask_image = apply_mask_to_image(image, out_mask, obj_id=out_obj_id)
        image = image * (1 - mask_image[..., 3, None]) + mask_image[..., :3] * mask_image[..., 3, None]  # Blend original and mask
    output_image_path = os.path.join(output_dir, f"frame_{out_frame_idx}.png")
    Image.fromarray(np.clip(image, 0, 255).astype('uint8')).save(output_image_path)  # Clip to valid range and convert to uint8


print('done!')