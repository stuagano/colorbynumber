import os

# Patch gradio_client bug: schema functions crash when schema is a bool
# (from additionalProperties: true in gr.State's JSON schema).
import gradio_client.utils as _gc_utils
_orig_inner = _gc_utils._json_schema_to_python_type
def _safe_inner(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_inner(schema, defs)
_gc_utils._json_schema_to_python_type = _safe_inner

import gradio as gr

from colorbynumber.config import default_config
from gradio_server import callbacks

MAX_NUM_COLORS = 50

with gr.Blocks(title = "Color by number") as demo:
    with gr.Row():
        # Inputs
        with gr.Column():
            image_path = gr.Image(type="filepath")
            image_examples = gr.Examples(
                examples=[
                    ["ExampleImages/Macaw.jpeg"],
                    ["ExampleImages/Grids.png"],
                ],
                inputs=[image_path]
            )

            with gr.Accordion(label="Crop (optional)", open=False):
                gr.Markdown(
                    "Trim the image to the region you want as a color-by-number. "
                    "Values are percentages of the original image (0–100)."
                )
                with gr.Row():
                    crop_left = gr.Slider(label="Left %", minimum=0, maximum=99, step=1, value=0)
                    crop_right = gr.Slider(label="Right %", minimum=1, maximum=100, step=1, value=100)
                with gr.Row():
                    crop_top = gr.Slider(label="Top %", minimum=0, maximum=99, step=1, value=0)
                    crop_bottom = gr.Slider(label="Bottom %", minimum=1, maximum=100, step=1, value=100)
                background_blur = gr.Slider(
                    label="Background blur",
                    minimum=0, maximum=100, step=5, value=0,
                    info="Softens the edges so the palette focuses on the centred subject.",
                )
                crop_preview = gr.Image(label="Crop preview", interactive=False)
                scan_button = gr.Button("Scan image for rendering issues")
                scan_results = gr.Markdown(value="")

                def _update_crop_preview(path, left, top, right, bottom, blur):
                    if not path:
                        return None
                    return callbacks.preview_crop(path, left, top, right, bottom, blur)

                preview_inputs = [image_path, crop_left, crop_top, crop_right, crop_bottom, background_blur]
                for slider in (crop_left, crop_top, crop_right, crop_bottom, background_blur):
                    slider.change(
                        fn=_update_crop_preview,
                        inputs=preview_inputs,
                        outputs=crop_preview,
                    )
                image_path.change(
                    fn=_update_crop_preview,
                    inputs=preview_inputs,
                    outputs=crop_preview,
                )

            output_style = gr.Radio(
                label="Output style",
                choices=["Pixel grid", "Islands"],
                value="Pixel grid",
                info="Islands: free-form color regions. Pixel grid: square cells like cross-stitch.",
            )

            number_of_colors = gr.Slider(
                label="Number of colors",
                minimum=2,
                maximum=30,
                step=1,
                value=10,
                info="More colors = more detail but harder to color.",
            )

            title = gr.Textbox(
                label="Title (optional)",
                placeholder="e.g. Macaw - by Sarah",
                info="Printed between the grid and the legend.",
            )

            with gr.Row():
                paper_size = gr.Radio(
                    label="Paper",
                    choices=["Letter", "A4"],
                    value="Letter",
                )
                orientation = gr.Radio(
                    label="Orientation",
                    choices=["Portrait", "Landscape"],
                    value="Portrait",
                )

            is_automatic_colors = gr.Checkbox(label="Automatic colors", value=True)

            # Manual color pickers (hidden by default)
            color_pickers = []
            with gr.Row(visible=False) as color_picker_row:
                for i in range(MAX_NUM_COLORS):
                    color_pickers.append(gr.ColorPicker(label=str(i + 1)))

            def _change_number_of_colors(number_of_colors):
                return [gr.update(visible=True)]*number_of_colors + \
                    [gr.update(visible=False)]*(MAX_NUM_COLORS - number_of_colors)
            def _get_color_selection_ui(is_automatic_colors_checked, number_of_colors):
                if is_automatic_colors_checked:
                    return [gr.update(visible=False)] + _change_number_of_colors(0)
                else:
                    return [gr.update(visible=True)] + _change_number_of_colors(number_of_colors)

            is_automatic_colors.change(
                _get_color_selection_ui,
                inputs=[is_automatic_colors, number_of_colors],
                outputs=[color_picker_row] + color_pickers,
            )
            number_of_colors.change(
                fn=_change_number_of_colors,
                inputs=[number_of_colors],
                outputs=color_pickers,
            )

            submit_button = gr.Button("Submit", variant="primary")

            # Advanced settings — collapsed by default
            with gr.Accordion(label="Advanced settings", open=False):
                # Pixel grid options
                with gr.Group(visible=True) as pixel_grid_group:
                    pixel_grid_max_dim = gr.Slider(
                        label="Grid resolution (cells along longest side)",
                        minimum=8,
                        maximum=100,
                        step=1,
                        value=default_config["pixel_grid_max_dim"],
                        info="Higher = more detail, more cells to color.",
                    )
                    pixel_show_grid = gr.Checkbox(
                        label="Show gridlines",
                        value=default_config["pixel_show_grid"],
                    )

                # Islands options
                with gr.Group(visible=False) as islands_group:
                    denoise_flag = gr.Checkbox(
                        label="Denoise image",
                        value=default_config["denoise"],
                        info="Smooths the image before processing. Disable for images with sharp edges.",
                    )
                    blur_size = gr.Slider(
                        label="Denoise strength",
                        minimum=3,
                        maximum=99,
                        step=2,
                        value=default_config["blur_size"],
                        info="Higher = smoother but less detail.",
                    )
                    open_kernel_size = gr.Slider(
                        label="Simplification",
                        minimum=3,
                        maximum=51,
                        step=2,
                        value=default_config["open_kernel_size"],
                        info="Higher = cleaner shapes but may lose small details.",
                    )

            # Hidden components that still need to exist for the callback signature
            denoise_order = gr.Textbox(value=default_config["denoise_order"], visible=False)
            denoise_type = gr.Textbox(value=default_config["denoise_type"], visible=False)
            denoise_h = gr.Number(value=default_config["denoise_h"], visible=False)
            area_perc_threshold = gr.Number(value=default_config["area_perc_threshold"], visible=False)
            check_shape_validity = gr.Checkbox(value=default_config["check_shape_validity"], visible=False)
            arc_length_area_ratio_threshold = gr.Number(value=default_config["arc_length_area_ratio_threshold"], visible=False)
            font_size = gr.Number(value=default_config["font_size"], visible=False)
            font_color = gr.ColorPicker(value="#8c8c8c", visible=False)
            font_thickness = gr.Number(value=default_config["font_thickness"], visible=False)
            pixel_cell_size = gr.Number(value=default_config["pixel_cell_size"], visible=False)

        # Outputs
        with gr.Column():
            color_by_number_image = gr.Image(label="Color by number (with legend)")
            pdf_download = gr.File(label="Download printable PDF", interactive=False)
            simplified_image = gr.Image(label="Simplified image", visible=False)
            islands_image = gr.Image(label="Islands (no numbers)", visible=False)
            data = gr.State()
            current_mode = gr.State(value="Pixel grid")

    scan_button.click(
        fn=callbacks.scan_image,
        inputs=[image_path, pixel_grid_max_dim, crop_left, crop_top, crop_right, crop_bottom],
        outputs=scan_results,
    )

    # ---- Output-style switching ----
    def _switch_output_style(style):
        is_pixel = (style == "Pixel grid")
        return (
            gr.update(visible=is_pixel),
            gr.update(visible=not is_pixel),
            style,
        )
    output_style.change(
        fn=_switch_output_style,
        inputs=[output_style],
        outputs=[pixel_grid_group, islands_group, current_mode],
    )

    # ---- Submit dispatch ----
    def _on_submit(
        style,
        image_path, number_of_colors,
        is_automatic_colors, num_colors,
        title,
        crop_left, crop_top, crop_right, crop_bottom, background_blur,
        paper_size, orientation,
        denoise_flag, denoise_order, denoise_type,
        blur_size, denoise_h,
        open_kernel_size, area_perc_threshold,
        check_shape_validity, arc_length_area_ratio_threshold,
        font_size, font_color, font_thickness,
        pixel_grid_max_dim, pixel_cell_size, pixel_show_grid,
        *color_list,
    ):
        image_path = callbacks.apply_crop_to_path(
            image_path, crop_left, crop_top, crop_right, crop_bottom, background_blur
        )
        paper = (paper_size or "Letter").lower()
        orient = (orientation or "Portrait").lower()
        if style == "Pixel grid":
            return callbacks.get_pixel_grid_color_by_number(
                image_path,
                number_of_colors,
                is_automatic_colors,
                num_colors,
                pixel_grid_max_dim,
                pixel_cell_size,
                pixel_show_grid,
                font_color,
                font_thickness,
                title,
                paper,
                orient,
                *color_list,
            )
        return callbacks.get_color_by_number(
            image_path, number_of_colors,
            is_automatic_colors, num_colors,
            denoise_flag, denoise_order, denoise_type,
            blur_size, denoise_h,
            open_kernel_size, area_perc_threshold,
            check_shape_validity, arc_length_area_ratio_threshold,
            font_size, font_color, font_thickness,
            title,
            paper,
            orient,
            *color_list,
        )

    submit_button.click(
        fn=_on_submit,
        inputs=[
            output_style,
            image_path,
            number_of_colors,
            is_automatic_colors,
            number_of_colors,
            title,
            crop_left,
            crop_top,
            crop_right,
            crop_bottom,
            background_blur,
            paper_size,
            orientation,
            denoise_flag,
            denoise_order,
            denoise_type,
            blur_size,
            denoise_h,
            open_kernel_size,
            area_perc_threshold,
            check_shape_validity,
            arc_length_area_ratio_threshold,
            font_size,
            font_color,
            font_thickness,
            pixel_grid_max_dim,
            pixel_cell_size,
            pixel_show_grid,
            *color_pickers,
        ],
        outputs=[color_by_number_image, simplified_image, islands_image, data, pdf_download],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
