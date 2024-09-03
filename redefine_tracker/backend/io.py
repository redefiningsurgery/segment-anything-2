import cv2
import numpy as np
from utils import mask_to_bbox, draw_mask, draw_rect

def apply_stroke(tracker_state, drawing_board, last_draw, frame_num, object_id):
    """Applies the stroke-based segmentation on the given frame."""
    predictor, inference_state, image_predictor = tracker_state
    image_path = f'output_frames/{frame_num:07d}.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image = drawing_board["image"]

    image_predictor.set_image(image)
    input_mask = drawing_board["mask"]
    input_mask[input_mask != 0] = 255

    if last_draw is not None:
        diff_mask = cv2.absdiff(input_mask, last_draw)
        input_mask = diff_mask

    bbox, has_mask = mask_to_bbox(input_mask[:, :, 0])
    if not has_mask:
        return tracker_state, display_image, display_image

    masks, scores, logits = image_predictor.predict(
        point_coords=None, point_labels=None, box=bbox[None, :], multimask_output=False
    )
    mask = masks > 0.0
    masked_frame = draw_mask(mask, display_image, object_id)
    masked_with_rect = draw_rect(masked_frame, bbox, object_id)
    predictor.add_new_mask(inference_state, frame_idx=frame_num, obj_id=object_id, mask=mask[0])

    last_draw = drawing_board["mask"]
    return tracker_state, masked_with_rect, masked_with_rect, last_draw

def handle_click(tracker_state, frame_num, point_mode, click_stack, object_id, evt):
    """Handles click events for point-based segmentation."""
    points_dict, labels_dict = click_stack
    predictor, inference_state, image_predictor = tracker_state
    annotation_frame_idx = frame_num

    point = np.array([[evt.index[0], evt.index[1]]], dtype=np.float32)
    label = np.array([1], np.int32) if point_mode == "Positive" else np.array([0], np.int32)

    if annotation_frame_idx not in points_dict:
        points_dict[annotation_frame_idx] = {}
    if annotation_frame_idx not in labels_dict:
        labels_dict[annotation_frame_idx] = {}

    if object_id not in points_dict[annotation_frame_idx]:
        points_dict[annotation_frame_idx][object_id] = np.empty((0, 2), dtype=np.float32)
    if object_id not in labels_dict[annotation_frame_idx]:
        labels_dict[annotation_frame_idx][object_id] = np.empty((0,), dtype=np.int32)

    points_dict[annotation_frame_idx][object_id] = np.append(points_dict[annotation_frame_idx][object_id], point, axis=0)
    labels_dict[annotation_frame_idx][object_id] = np.append(labels_dict[annotation_frame_idx][object_id], label, axis=0)

    click_stack = (points_dict, labels_dict)

    predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=annotation_frame_idx,
        obj_id=object_id,
        points=points_dict[annotation_frame_idx][object_id],
        labels=labels_dict[annotation_frame_idx][object_id],
    )

    image_path = f'output_frames/{annotation_frame_idx:07d}.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masked_frame = image.copy()
    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        masked_frame = draw_mask(mask, image=masked_frame, obj_id=obj_id)
    masked_frame_with_markers = draw_markers(masked_frame, points_dict[annotation_frame_idx], labels_dict[annotation_frame_idx])

    return tracker_state, masked_frame_with_markers, masked_frame_with_markers, click_stack

def increment_object_id(object_id):
    """Increments the annotation object ID."""
    return object_id + 1
