import os

import gradio as gr

from colorbynumber.config import default_config
from gradio_server import callbacks
from gradio_server import doc

MAX_NUM_COLORS = 50 # Mostly for UI purposes

with gr.Blocks(title = "Color by number") as demo:
    with gr.Row():
        # Inputs
        with gr.Column(elem_id="inputColumn"):
            image_path = gr.Image(type="filepath")
            image_examples = gr.Examples(
                examples=[
                    ["ExampleImages/Macaw.jpeg"],
                    ["ExampleImages/Grids.png"],
                ],
                inputs=[image_path]
            )

            # Output style selection
            gr.Markdown(doc.output_style_block())
            output_style = gr.Radio(
                label="Output style",
                choices=["Islands", "Pixel grid"],
                value="Islands",
            )

            # Color selection
            gr.Markdown(doc.color_selection_block())
            is_automatic_colors = gr.Checkbox(label = "Automatic colors", value = True)
            number_of_colors = gr.Number(precision=0, label = "Number of colors", value=10)

            # Color pickers
            color_pickers = []
            with gr.Row(visible=False) as color_picker_row:
                for i in range(MAX_NUM_COLORS):
                    color_pickers.append(gr.ColorPicker(label = str(i + 1)))

            # Toggle visibility of color pickers
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
                inputs = [is_automatic_colors, number_of_colors],
                outputs=[color_picker_row] + color_pickers,
            )
            number_of_colors.change(
                fn=_change_number_of_colors,
                inputs=[number_of_colors],
                outputs=color_pickers,
            )

            # Config UI
            gr.Markdown(doc.parameters_block_header())
            with gr.Accordion(label="Configuration") as config_accordion:
                # ---- Pixel grid configuration ----
                with gr.Group(visible=False) as pixel_grid_group:
                    gr.Markdown(doc.pixel_grid_parameters())
                    pixel_grid_max_dim = gr.Slider(
                        label="Grid resolution (cells along longest side)",
                        minimum=8,
                        maximum=100,
                        step=1,
                        value=default_config["pixel_grid_max_dim"],
                        info="Higher = more detail, but more cells to color."
                    )
                    pixel_cell_size = gr.Slider(
                        label="Cell size (pixels)",
                        minimum=8,
                        maximum=80,
                        step=1,
                        value=default_config["pixel_cell_size"],
                        info="Pixel size of each cell in the rendered output."
                    )
                    pixel_show_grid = gr.Checkbox(
                        label="Show gridlines",
                        value=default_config["pixel_show_grid"],
                    )

                # ---- Islands configuration (original) ----
                with gr.Group(visible=True) as islands_group:
                    with gr.Tab(label="Denoise") as denoise_tab:
                        # Denoise parameters
                        gr.Markdown(doc.denoise_block_header())
                        denoise_flag = gr.Checkbox(
                                label = "Denoise",
                                value = default_config["denoise"]
                                )
                        with gr.Group() as denoise_params:
                            with gr.Row():
                                denoise_order = gr.Dropdown(
                                    label = "Denoise order",
                                    choices = ["before_simplify", "after_simplify"],
                                    value = default_config["denoise_order"],
                                    )
                                denoise_type = gr.Dropdown(
                                    label = "Denoise type",
                                    choices = ["fastNlMeansDenoisingColored", "gaussianBlur", "blur"],
                                    value = default_config["denoise_type"],
                                    info="Algorithm to be used for denoising"
                                    )
                            show_denoise_h = False
                            if default_config["denoise_type"] == "fastNlMeansDenoisingColored":
                                show_denoise_h = True

                            with gr.Row():
                                blur_size = gr.Slider(
                                    label = "Blur size",
                                    minimum = 3,
                                    maximum = 100,
                                    step=2,
                                    value = default_config["blur_size"],
                                    info="Larger values will denoise more",
                                    visible=(not show_denoise_h)
                                    )
                                denoise_h = gr.Slider(
                                    label = "h",
                                    minimum = 0,
                                    maximum = 500,
                                    value = default_config["denoise_h"],
                                    info="Larger values will denoise more",
                                    visible=show_denoise_h
                                    )

                        def _toggle_h_blur_size_visibility(event: gr.SelectData):
                            if event.value == "fastNlMeansDenoisingColored":
                                # Show denoise_h, hide blur_size
                                return [gr.update(visible=False), gr.update(visible=True)]
                            else:
                                # Show blur_size, hide denoise_h
                                return [gr.update(visible=True), gr.update(visible=False)]
                        denoise_type.select(
                            fn = _toggle_h_blur_size_visibility,
                            inputs = None,
                            outputs = [blur_size, denoise_h]
                            )

                        denoise_flag.change(
                            fn = lambda x: gr.update(visible=x),
                            inputs = [denoise_flag],
                            outputs = denoise_params
                        )

                    with gr.Tab(label = "Simplify") as simplify_tab:

                        # Simplification parameters
                        gr.Markdown(doc.simplify_islands_parameters())
                        open_kernel_size = gr.Slider(
                            label = "Open kernel size",
                            minimum = 3,
                            maximum = 51,
                            step=2,
                            value = default_config["open_kernel_size"],
                            info="Larger the value, cleaner the image. But too large values can remove important details."
                        )
                        area_perc_threshold = gr.Slider(
                            label = "Area Percentage threshold",
                            minimum = 0,
                            maximum = 10,
                            step=0.01,
                            value = default_config["area_perc_threshold"],
                            info="Islands which cover a percentage area less than this threshold will be removed."
                        )

                        check_shape_validity = gr.Checkbox(
                            label = "Remove thin islands",
                            value = default_config["check_shape_validity"],
                        )
                        arc_length_area_ratio_threshold = gr.Slider(
                            label = "Arc length to Area ratio",
                            minimum = 0,
                            maximum = 10,
                            step=0.01,
                            value = default_config["arc_length_area_ratio_threshold"],
                            info="Smaller value removes more islands.",
                            visible=default_config["check_shape_validity"]
                        )
                        check_shape_validity.change(
                            fn = lambda x: gr.update(visible=x),
                            inputs = [check_shape_validity],
                            outputs = [arc_length_area_ratio_threshold]
                        )

            # Submit button
            submit_button = gr.Button("Submit")

        # Outputs
        with gr.Column():
            color_by_number_image = gr.Image(label = "Color by number")

            # Edit coloring page
            with gr.Row():
                font_size = gr.Slider(
                    label = "Font size",
                    minimum = 0.1,
                    maximum = 10,
                    value = default_config["font_size"],
                )
                font_thickness = gr.Slider(
                    label = "Font thickness",
                    minimum = 1,
                    maximum = 10,
                    step=1,
                    value = default_config["font_thickness"],
                )
                font_color = gr.ColorPicker(
                    label = "Font color",
                    value = "#8c8c8c",
                    visible=False,
                    )
            legend_image = gr.Image(label = "Legend")
            simplified_image = gr.Image(label = "Simplified image")
            islands_image = gr.Image(label = "Islands (no numbers)", visible=False)
            data = gr.State() # To store the data for font change
            current_mode = gr.State(value="Islands")

        # ---- Output-style switching ----
        def _switch_output_style(style):
            is_pixel = (style == "Pixel grid")
            return (
                gr.update(visible=is_pixel),       # pixel_grid_group
                gr.update(visible=not is_pixel),   # islands_group
                style,                             # current_mode
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
            denoise_flag, denoise_order, denoise_type,
            blur_size, denoise_h,
            open_kernel_size, area_perc_threshold,
            check_shape_validity, arc_length_area_ratio_threshold,
            font_size, font_color, font_thickness,
            pixel_grid_max_dim, pixel_cell_size, pixel_show_grid,
            *color_list,
        ):
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
            outputs=[color_by_number_image, legend_image, simplified_image, islands_image, data],
        )

        # ---- Live font/grid edits ----
        def _on_font_change(style, image, data, font_size, font_color, font_thickness, cell_size):
            if style == "Pixel grid":
                return callbacks.change_pixel_grid_font(
                    data, cell_size, font_color, font_thickness
                )
            return callbacks.change_font_on_image(
                image, data, font_size, font_color, font_thickness
            )

        gr.on(
                triggers=[font_size.change, font_thickness.change, pixel_cell_size.change],
                fn = _on_font_change,
                inputs = [
                    current_mode,
                    islands_image,
                    data,
                    font_size,
                    font_color,
                    font_thickness,
                    pixel_cell_size,
                    ],
                outputs = color_by_number_image
            )

if __name__ == "__main__":
    # Render / Fly / Railway / any PaaS sets $PORT; fall back to Gradio's default.
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
