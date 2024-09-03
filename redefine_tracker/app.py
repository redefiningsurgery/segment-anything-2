import gradio as gr
from backend.preprocessing import preprocess_video, reset_app
from backend.io import apply_stroke, handle_click, increment_object_id
from backend.tracking import track_objects_in_video
from utils import initialize_drawing_board

def create_gradio_interface():
    """Sets up and returns the Gradio interface for the video processing application."""
    
    css = """
    #input_output_video video {
        max-height: 550px;
        max-width: 100%;
        height: auto;
    }
    """

    app = gr.Blocks(css=css)

    with app:
        gr.Markdown(
            '''
            <div style="text-align:center;">
                <span style="font-size:3em; font-weight:bold;">Redefine Surgery SAMTracker 🦾</span>
            </div>
            1. Upload video file 
            2. Select model size and downsample frame rate, then run `Preprocess`
            3. Use `Stroke to Box Prompt` to draw a box on the first frame or `Point Prompt` to click on the first frame
            4. Click `Segment` to get the segmentation result
            5. Click `Add New Object` to add a new object
            6. Click `Start Tracking` to track objects in the video
            7. Click `Reset` to reset the app
            8. Download the video with segmentation result
            '''
        )

        state_clicks = gr.State(({}, {}))
        state_tracker = gr.State(None)
        state_frame_num = gr.State(value=(int(0)))
        state_object_id = gr.State(value=(int(0)))
        state_last_draw = gr.State(None)

        with gr.Row():
            with gr.Column(scale=0.5):
                with gr.Tab(label="Video input"):
                    input_video = gr.Video(label='Input video', elem_id="input_output_video")
                    with gr.Row():
                        model_size = gr.Dropdown(label="Model Size", choices=["tiny", "small", "base-plus", "large"], value="tiny")
                        frame_rate_slider = gr.Slider(
                            label="Downsample Frame Rate",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.25,
                            value=1.0,
                            interactive=True
                        )
                        preprocess_button = gr.Button(value="Preprocess", interactive=True)

                with gr.Tab(label="Stroke to Box Prompt"):
                    drawing_board = gr.Image(label='Drawing Board', tool="sketch", brush_radius=10, interactive=True)
                    seg_button = gr.Button(value="Segment", interactive=True)
                            
                with gr.Tab(label="Point Prompt"):
                    first_frame_image = gr.Image(label='Segment result of first frame', interactive=True).style(height=550)
                    point_mode = gr.Radio(
                        choices=["Positive",  "Negative"],
                        value="Positive",
                        label="Point Prompt",
                        interactive=True
                    )
                            
                frame_slider = gr.Slider(
                    label="Number of Frames Viewed",
                    minimum=0,
                    maximum=200,
                    step=1,
                    value=0,
                )
                add_object_button = gr.Button(value="Add New Object", interactive=True)
                track_button = gr.Button(value="Start Tracking", interactive=True)
                reset_button = gr.Button(value="Reset", interactive=True)

            with gr.Column(scale=0.5):
                output_video = gr.Video(label='Visualize Results', elem_id="input_output_video")
                output_mp4 = gr.File(label="Predicted video")
                output_mask = gr.File(label="Predicted masks")

        preprocess_button.click(
            fn=preprocess_video,
            inputs=[state_tracker, input_video, frame_rate_slider, model_size],
            outputs=[state_tracker, state_clicks, first_frame_image, drawing_board, frame_slider, output_video, output_mp4, output_mask, state_object_id]
        )

        frame_slider.release(
            fn=show_results_by_slider,
            inputs=[frame_slider, state_clicks],
            outputs=[first_frame_image, drawing_board, state_frame_num]
        )

        first_frame_image.select(
            fn=handle_click,
            inputs=[state_tracker, state_frame_num, point_mode, state_clicks, state_object_id],
            outputs=[state_tracker, first_frame_image, drawing_board, state_clicks]
        )

        track_button.click(
            fn=track_objects_in_video,
            inputs=[state_tracker, state_frame_num, input_video],
            outputs=[first_frame_image, drawing_board, output_video, output_mp4, output_mask]
        )

        reset_button.click(
            fn=reset_app,
            inputs=[state_tracker],
            outputs=[state_tracker, state_clicks, first_frame_image, drawing_board, frame_slider, output_video, output_mp4, output_mask, state_object_id]
        )

        add_object_button.click(
            fn=increment_object_id,
            inputs=[state_object_id],
            outputs=[state_object_id]
        )

        seg_button.click(
            fn=apply_stroke,
            inputs=[state_tracker, drawing_board, state_last_draw, state_frame_num, state_object_id],
            outputs=[state_tracker, first_frame_image, drawing_board, state_last_draw]
        )

        drawing_board.set(fn=initialize_drawing_board, inputs=[first_frame_image], outputs=[drawing_board])

    return app

if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(debug=True, enable_queue=True, share=True)
