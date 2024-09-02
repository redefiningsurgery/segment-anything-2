import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

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

video_dir = "output_frames"

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=video_dir) # frame loading (JPEG): ...
predictor.reset_state(inference_state)

# import inspect
# print(inspect.getsource(predictor.init_state))

ann_frame_idx = 0
ann_obj_id = 1
points = np.array([[210, 250], [200, 600]], dtype=np.float32)
# points = np.array([[210, 350], [250, 220]], dtype=np.float32)
labels = np.array([1, 0], np.int32) # `1` means positive click and `0` means negative click

_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

ann_frame_idx = 140
ann_obj_id = 1
points = np.array([[44, 366]], dtype=np.float32)
labels = np.array([1], np.int32)
_, _, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    # for obj_id, mask in video_segments[out_frame_idx].items():
    #     print(f"Object ID: {obj_id}, Mask Shape: {mask.shape}")

vis_frame_stride = 1
output_dir = 'output_masks'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

t_start = time.time()
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    frame_path = os.path.join(video_dir, frame_names[out_frame_idx])
    image = np.array(Image.open(frame_path), dtype=np.float32)  # Ensure image is in float format to blend properly
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        mask_image = apply_mask_to_image(image, out_mask, obj_id=out_obj_id)
        image = image * (1 - mask_image[..., 3, None]) + mask_image[..., :3] * mask_image[..., 3, None]  # Blend original and mask
    output_image_path = os.path.join(output_dir, f"frame_{out_frame_idx}.png")
    Image.fromarray(np.clip(image, 0, 255).astype('uint8')).save(output_image_path)  # Clip to valid range and convert to uint8
print(time.time() - t_start)