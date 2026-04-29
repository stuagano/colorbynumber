"""Pixel-grid ("pixel art") color-by-number generator.

This is an alternative output style to the island-based one in `main.py`.
The image is downsampled to a small grid (e.g. 48 cells on the longest
side), each cell is quantized to a single color from a palette (provided
by the user or learned via k-means), and the result is rendered as a
large grid of square cells with their color number printed inside.

Everything runs locally with OpenCV + NumPy — no AI / API key required.
"""

import cv2 as cv
import numpy as np

from .config import default_config
from .simplify_image import _choose_closest_colors, _kmeans_simplify_image


def _resize_to_grid(image, grid_max_dim):
    """Resize so the longest side equals `grid_max_dim` cells.

    Aspect ratio is preserved. INTER_AREA averages source pixels into each
    cell, which is what we want for the pixel-art look.
    """
    h, w = image.shape[:2]
    if w >= h:
        grid_w = int(grid_max_dim)
        grid_h = max(1, int(round(h * (grid_w / w))))
    else:
        grid_h = int(grid_max_dim)
        grid_w = max(1, int(round(w * (grid_h / h))))
    return cv.resize(image, (grid_w, grid_h), interpolation=cv.INTER_AREA)


def _quantize_to_palette(small_image, color_list, num_colors, apply_kmeans):
    """Quantize a small image to either a fixed palette or k-means colors.

    Returns:
        indices: (h, w) int array, values 1..N (1-indexed color id)
        palette: (N, 3) uint8 array of RGB colors
    """
    if color_list is None:
        _, indices, palette = _kmeans_simplify_image(small_image, num_colors)
        return indices, np.asarray(palette, dtype=np.uint8)

    # Fixed palette path. Optionally pre-cluster with k-means so each cell
    # snaps to a representative average before being mapped to the palette.
    image = small_image
    if apply_kmeans:
        image, _, _ = _kmeans_simplify_image(small_image, len(color_list))
    _, indices = _choose_closest_colors(image, color_list)
    palette = np.asarray(color_list, dtype=np.uint8)
    return indices, palette


def _auto_font_scale(cell_size, max_digits):
    """Pick a font scale so `max_digits` characters fit in a cell."""
    # cv.FONT_HERSHEY_SIMPLEX glyphs at scale=1 are ~22px wide. Aim for
    # ~70% of the cell width to leave a little padding.
    target_width = cell_size * 0.7
    glyph_width = 22 * max_digits
    return max(0.3, target_width / glyph_width)


def render_pixel_grid(
    indices,
    palette,
    cell_size=30,
    show_grid=True,
    grid_color=(181, 181, 181),
    fill_cells=False,
    show_numbers=True,
    font_color=(80, 80, 80),
    font_scale=None,
    font_thickness=1,
):
    """Render the pixel-grid color-by-number canvas.

    Args:
        indices: (h, w) int array, 1-indexed color ids.
        palette: (N, 3) uint8 RGB palette.
        cell_size: pixel width/height of each cell in the rendered image.
        show_grid: draw thin gridlines between cells.
        grid_color: RGB color of gridlines.
        fill_cells: if True, fill each cell with its palette color (the
            "answer key" / pixel-art preview). Otherwise cells are white.
        show_numbers: if True, draw the color id in each cell.
        font_color: RGB color of the numbers.
        font_scale: optional override; otherwise auto-sized to the cell.
        font_thickness: text stroke thickness.

    Returns:
        (h_out, w_out, 3) uint8 RGB image.
    """
    gh, gw = indices.shape[:2]
    out_h = gh * cell_size
    out_w = gw * cell_size
    canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)

    if fill_cells:
        # Fill cells with palette colors.
        cell_colors = palette[indices - 1]  # (gh, gw, 3)
        # Upsample by repeating each cell.
        canvas = np.repeat(np.repeat(cell_colors, cell_size, axis=0), cell_size, axis=1)
        canvas = canvas.astype(np.uint8)

    if show_grid:
        for r in range(gh + 1):
            y = min(r * cell_size, out_h - 1)
            canvas[y, :] = grid_color
        for c in range(gw + 1):
            x = min(c * cell_size, out_w - 1)
            canvas[:, x] = grid_color

    if show_numbers:
        max_digits = len(str(int(indices.max())))
        if font_scale is None:
            font_scale = _auto_font_scale(cell_size, max_digits)
        font = cv.FONT_HERSHEY_SIMPLEX

        for r in range(gh):
            for c in range(gw):
                color_id = int(indices[r, c])
                text = str(color_id)
                (tw, th), _ = cv.getTextSize(text, font, font_scale, font_thickness)
                x = c * cell_size + (cell_size - tw) // 2
                y = r * cell_size + (cell_size + th) // 2
                cv.putText(
                    canvas,
                    text,
                    (x, y),
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                    cv.LINE_AA,
                )

    return canvas


def generate_color_legend(
    color_list,
    cols=7,
    square_size=100,
    margin=10,
    gap_horizontal=5,
    gap_vertical=30,
    font=cv.FONT_HERSHEY_SIMPLEX,
    font_size=1,
    border_color=(0, 0, 0),
):
    """Standalone version of the legend grid (works without a class)."""
    color_list = list(color_list)
    cols = min(cols, len(color_list))
    rows = int(np.ceil(len(color_list) / cols))

    total_width = 2 * margin + (cols + 1) * square_size + (cols - 1) * gap_horizontal
    total_height = 2 * margin + (rows + 1) * square_size + (rows - 1) * gap_vertical
    image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    for i, color in enumerate(color_list):
        row = i // cols
        col = i % cols
        start_col = margin + col * (square_size + gap_horizontal)
        end_col = start_col + square_size
        start_row = margin + row * (square_size + gap_vertical)
        end_row = start_row + square_size

        image[start_row:end_row, start_col:end_col] = color
        image[start_row, start_col:end_col] = border_color
        image[end_row, start_col:end_col] = border_color
        image[start_row:end_row, start_col] = border_color
        image[start_row:end_row, end_col] = border_color

        text = str(i + 1)
        text_size, _ = cv.getTextSize(text, font, font_size, 1)
        text_row = (end_row + text_size[1]) + 5
        text_col = start_col + (square_size // 2) - (text_size[0] // 2)
        cv.putText(image, text, (text_col, text_row), font, font_size, (0, 0, 0), 1)

        r, g, b = (int(v) for v in color)
        hex_text = f"#{r:02X}{g:02X}{b:02X}"
        hex_scale = font_size * 0.45
        hex_size, _ = cv.getTextSize(hex_text, font, hex_scale, 1)
        hex_row = text_row + hex_size[1] + 6
        hex_col = start_col + (square_size // 2) - (hex_size[0] // 2)
        cv.putText(image, hex_text, (hex_col, hex_row), font, hex_scale, (90, 90, 90), 1, cv.LINE_AA)

    return image


class PixelColorByNumber:
    """Pixel-grid color-by-number generator.

    Usage:
        obj = PixelColorByNumber(image_path, num_colors=12)
        numbered = obj.create_color_by_number()
        legend   = obj.generate_color_legend()
        preview  = obj.filled_image  # answer-key pixel-art preview
    """

    def __init__(
        self,
        image_path,
        color_list=None,
        num_colors=None,
        config=default_config,
    ):
        assert color_list is not None or num_colors is not None, \
            "Either color_list or num_colors must be provided."

        self.image_path = image_path
        self.config = config
        self.color_list = color_list
        self.num_colors = num_colors

        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.image = image

    def create_color_by_number(self):
        cfg = self.config
        small = _resize_to_grid(self.image, cfg["pixel_grid_max_dim"])
        indices, palette = _quantize_to_palette(
            small,
            color_list=self.color_list,
            num_colors=self.num_colors,
            apply_kmeans=cfg.get("apply_kmeans", True),
        )

        self.indices = indices
        self.palette = palette
        # Keep parity with ColorByNumber — color_list is the active palette.
        self.color_list = [tuple(int(v) for v in c) for c in palette]

        font_color = cfg.get("font_color", (80, 80, 80))
        grid_color = cfg.get("border_color", (181, 181, 181))

        self.numbered_image = render_pixel_grid(
            indices,
            palette,
            cell_size=cfg["pixel_cell_size"],
            show_grid=cfg.get("pixel_show_grid", True),
            grid_color=grid_color,
            fill_cells=False,
            show_numbers=True,
            font_color=font_color,
            font_thickness=int(cfg.get("font_thickness", 1)),
        )

        self.filled_image = render_pixel_grid(
            indices,
            palette,
            cell_size=cfg["pixel_cell_size"],
            show_grid=cfg.get("pixel_show_grid", True),
            grid_color=grid_color,
            fill_cells=True,
            show_numbers=False,
        )

        self.blank_grid_image = render_pixel_grid(
            indices,
            palette,
            cell_size=cfg["pixel_cell_size"],
            show_grid=cfg.get("pixel_show_grid", True),
            grid_color=grid_color,
            fill_cells=False,
            show_numbers=False,
        )

        return self.numbered_image

    def render_with_font(self, cell_size=None, font_thickness=None, font_color=None):
        """Re-render the numbered grid with new font settings."""
        cfg = self.config
        return render_pixel_grid(
            self.indices,
            self.palette,
            cell_size=cell_size or cfg["pixel_cell_size"],
            show_grid=cfg.get("pixel_show_grid", True),
            grid_color=cfg.get("border_color", (181, 181, 181)),
            fill_cells=False,
            show_numbers=True,
            font_color=font_color or cfg.get("font_color", (80, 80, 80)),
            font_thickness=int(font_thickness or cfg.get("font_thickness", 1)),
        )

    def generate_color_legend(self, **kwargs):
        return generate_color_legend(self.color_list, **kwargs)
