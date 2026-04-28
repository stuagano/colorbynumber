import cv2 as cv
import numpy as np

from colorbynumber.config import default_config
from colorbynumber.main import ColorByNumber
from colorbynumber.numbered_islands import add_numbers_to_image
from colorbynumber.pixel_grid import PixelColorByNumber


def _render_title(text, width, font_scale=1.5, thickness=2):
    """Render a title string centred on a white strip matching `width`."""
    if not text or not text.strip():
        return None
    font = cv.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv.getTextSize(text.strip(), font, font_scale, thickness)
    # Auto-shrink if title is wider than the image
    while tw > width - 40 and font_scale > 0.4:
        font_scale -= 0.1
        (tw, th), baseline = cv.getTextSize(text.strip(), font, font_scale, thickness)
    strip_h = th + baseline + 40  # padding above and below
    strip = np.ones((strip_h, width, 3), dtype=np.uint8) * 255
    x = (width - tw) // 2
    y = 20 + th
    cv.putText(strip, text.strip(), (x, y), font, font_scale, (0, 0, 0), thickness, cv.LINE_AA)
    return strip


def _combine_grid_and_legend(grid, legend, title=None, padding=30):
    """Stack the coloring grid, title, and legend vertically into one printable image."""
    gh, gw = grid.shape[:2]
    lh, lw = legend.shape[:2]
    width = max(gw, lw)

    parts = [grid]
    if title and title.strip():
        title_strip = _render_title(title, width)
        if title_strip is not None:
            parts.append(title_strip)
    parts.append(legend)

    # Stack all parts with padding between them
    total_h = sum(p.shape[0] for p in parts) + padding * (len(parts) - 1)
    combined = np.ones((total_h, width, 3), dtype=np.uint8) * 255
    y = 0
    for i, part in enumerate(parts):
        ph, pw = part.shape[:2]
        x_offset = (width - pw) // 2
        combined[y:y + ph, x_offset:x_offset + pw] = part
        y += ph + padding
    return combined


def _hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_color_by_number(image_path, number_of_colors,
                        is_automatic_colors, num_colors,
                        denoise_flag, denoise_order, denoise_type,
                        blur_size, denoise_h,
                        open_kernel_size, area_perc_threshold,
                        check_shape_validity, arc_length_area_ratio_threshold,
                        font_size, font_color, font_thickness,
                        title,
                        *color_list):
    # Convert each color to r,g,b tuple
    color_list = color_list[:num_colors]
    color_list = [_hex_to_rgb(h) for h in color_list]

    # Update config
    config = default_config.copy()
    config["denoise"] = denoise_flag
    config["denoise_order"] = denoise_order
    config["denoise_type"] = denoise_type
    config["blur_size"] = blur_size
    config["denoise_h"] = denoise_h
    config["open_kernel_size"] = open_kernel_size
    config["area_perc_threshold"] = area_perc_threshold
    config["check_shape_validity"] = check_shape_validity
    config["arc_length_area_ratio_threshold"] = arc_length_area_ratio_threshold
    config["font_size"] = font_size
    config["font_color"] = _hex_to_rgb(font_color)
    config["font_thickness"] = font_thickness

    if is_automatic_colors:
        colorbynumber_obj = ColorByNumber(
            image_path = image_path,
            num_colors = number_of_colors,
            config = config,
        )
    else:
        colorbynumber_obj = ColorByNumber(
            image_path = image_path,
            color_list = color_list,
            config = config,
        )

    numbered_islands = colorbynumber_obj.create_color_by_number()
    num_colors = len(colorbynumber_obj.color_list)
    legend = colorbynumber_obj.generate_color_legend(cols=num_colors)
    combined = _combine_grid_and_legend(numbered_islands, legend, title=title)
    data = {
        "centroid_coords_list": colorbynumber_obj.centroid_coords_list,
        "color_id_list": [color_id for color_id, _ in colorbynumber_obj.island_borders_list]
    }
    return combined, \
        colorbynumber_obj.simplified_image, \
        colorbynumber_obj.islands_image, \
        data

def change_font_on_image(image, data, font_size, font_color, font_thickness):
    if image is None:
        return None

    centroid_coords_list = data["centroid_coords_list"]
    color_id_list = data["color_id_list"]

    font_color = _hex_to_rgb(font_color)
    return add_numbers_to_image(
        image = image,
        centroid_coords_list = centroid_coords_list,
        color_id_list = color_id_list,
        font_size = font_size,
        font_color = font_color,
        font_thickness = font_thickness
    )


def get_pixel_grid_color_by_number(
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
    *color_list,
):
    """Pixel-grid output style. No island/denoise params used."""
    color_list = color_list[:num_colors]
    color_list = [_hex_to_rgb(h) for h in color_list]

    config = default_config.copy()
    # The pixel-grid path doesn't need denoising — each cell is already an
    # average of its source pixels. Keeping it on actively hurts detail.
    config["denoise"] = False
    config["pixel_grid_max_dim"] = int(pixel_grid_max_dim)
    config["pixel_cell_size"] = int(pixel_cell_size)
    config["pixel_show_grid"] = bool(pixel_show_grid)
    config["font_color"] = _hex_to_rgb(font_color)
    config["font_thickness"] = int(font_thickness)

    if is_automatic_colors:
        obj = PixelColorByNumber(
            image_path=image_path,
            num_colors=int(number_of_colors),
            config=config,
        )
    else:
        obj = PixelColorByNumber(
            image_path=image_path,
            color_list=color_list,
            config=config,
        )

    numbered = obj.create_color_by_number()
    num_colors = len(obj.color_list)
    legend = obj.generate_color_legend(cols=num_colors)
    combined = _combine_grid_and_legend(numbered, legend, title=title)

    data = {
        "mode": "pixel_grid",
        "indices": obj.indices.tolist(),
        "palette": [list(int(v) for v in c) for c in obj.palette],
        "cell_size": int(pixel_cell_size),
        "show_grid": bool(pixel_show_grid),
    }
    return combined, obj.filled_image, obj.blank_grid_image, data


def change_pixel_grid_font(data, cell_size, font_color, font_thickness):
    """Live re-render of the pixel grid when font controls change."""
    if not data or data.get("mode") != "pixel_grid":
        return None

    import numpy as np

    from colorbynumber.pixel_grid import render_pixel_grid

    indices = np.array(data["indices"], dtype=np.int32)
    palette = np.array(data["palette"], dtype=np.uint8)
    return render_pixel_grid(
        indices,
        palette,
        cell_size=int(cell_size),
        show_grid=bool(data.get("show_grid", True)),
        fill_cells=False,
        show_numbers=True,
        font_color=_hex_to_rgb(font_color),
        font_thickness=int(font_thickness),
    )
