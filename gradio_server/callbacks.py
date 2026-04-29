import os
import tempfile

import cv2 as cv
import numpy as np

from colorbynumber.config import default_config
from colorbynumber.main import ColorByNumber
from colorbynumber.numbered_islands import add_numbers_to_image
from colorbynumber.pixel_grid import PixelColorByNumber


def _normalize_crop_box(left, top, right, bottom):
    """Clamp/normalize crop percentages and ensure left<right, top<bottom."""
    left = max(0.0, min(99.0, float(left)))
    top = max(0.0, min(99.0, float(top)))
    right = max(1.0, min(100.0, float(right)))
    bottom = max(1.0, min(100.0, float(bottom)))
    if right <= left:
        right = min(100.0, left + 1.0)
    if bottom <= top:
        bottom = min(100.0, top + 1.0)
    return left, top, right, bottom


def _crop_image_pct(image, left, top, right, bottom):
    h, w = image.shape[:2]
    left, top, right, bottom = _normalize_crop_box(left, top, right, bottom)
    x1 = int(round(w * left / 100))
    x2 = int(round(w * right / 100))
    y1 = int(round(h * top / 100))
    y2 = int(round(h * bottom / 100))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return image[y1:y2, x1:x2]


def _is_full_crop(left, top, right, bottom):
    return float(left) <= 0 and float(top) <= 0 and float(right) >= 100 and float(bottom) >= 100


def preview_crop(image_path, left, top, right, bottom):
    """Return the cropped region as an RGB array for Gradio preview."""
    if not image_path:
        return None
    img = cv.imread(image_path)
    if img is None:
        return None
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return _crop_image_pct(img, left, top, right, bottom)


def apply_crop_to_path(image_path, left, top, right, bottom):
    """If a non-trivial crop is set, write the cropped image to a tempfile and
    return that path; otherwise return the original path."""
    if not image_path or _is_full_crop(left, top, right, bottom):
        return image_path
    img = cv.imread(image_path)
    if img is None:
        return image_path
    cropped = _crop_image_pct(img, left, top, right, bottom)
    fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(image_path)[1] or ".png")
    os.close(fd)
    cv.imwrite(tmp_path, cropped)
    return tmp_path


def scan_image(image_path, grid_max_dim, left, top, right, bottom):
    """Heuristic checks for a likely-poor color-by-number rendering.

    Looks at face size relative to grid cells, edge density, contrast, and
    aspect ratio. Returns a Markdown bullet list.
    """
    if not image_path:
        return "_Upload an image first._"
    img = cv.imread(image_path)
    if img is None:
        return "_Couldn't read the image file._"
    img = _crop_image_pct(img, left, top, right, bottom)
    h, w = img.shape[:2]
    grid_max_dim = max(8, int(grid_max_dim))
    pixels_per_cell = max(h, w) / grid_max_dim

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    issues, ok = [], []

    faces = []
    try:
        cascade_dir = getattr(getattr(cv, "data", None), "haarcascades", "")
        cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
        if cascade_dir and os.path.exists(cascade_path):
            face_cascade = cv.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
            )
    except Exception:
        faces = []

    for (fx, fy, fw, fh) in faces:
        face_cells = min(fw, fh) / pixels_per_cell
        if face_cells < 8:
            issues.append(
                f"Face detected ({fw}×{fh}px) spans only ~{face_cells:.0f} cells — "
                "facial features will be lost. Crop closer or raise grid resolution."
            )
        elif face_cells < 14:
            issues.append(
                f"Face detected ({fw}×{fh}px) is borderline (~{face_cells:.0f} cells). "
                "Cropping tighter to the face will help."
            )
    if len(faces) > 0 and not any("Face" in m for m in issues):
        ok.append(f"{len(faces)} face(s) detected — should render fine at this resolution.")

    edges = cv.Canny(gray, 100, 200)
    edge_ratio = float(edges.sum()) / 255.0 / (h * w)
    if edge_ratio > 0.18:
        issues.append(
            f"High detail density ({edge_ratio*100:.0f}% edge pixels) — fine textures will flatten. "
            "Consider cropping to the subject or fewer colors."
        )

    contrast = float(gray.std())
    if contrast < 25:
        issues.append(
            f"Low contrast (stddev={contrast:.0f}) — adjacent regions may merge into one color. "
            "Try increasing the number of colors or pick a more contrasty image."
        )

    aspect = max(w, h) / max(1, min(w, h))
    if aspect > 2.5:
        issues.append(
            f"Aspect ratio is extreme ({aspect:.1f}:1) — the grid will be very wide or tall. "
            "Crop closer to a square if you want a balanced page."
        )

    if min(h, w) / pixels_per_cell < 10:
        issues.append(
            "The shorter side of the image is fewer than ~10 cells — the result will look "
            "blocky. Crop closer or lower the grid resolution."
        )

    lines = []
    if issues:
        lines.append("**Potential issues:**")
        lines.extend(f"- {m}" for m in issues)
    if ok:
        if lines:
            lines.append("")
        lines.append("**Looks fine:**")
        lines.extend(f"- {m}" for m in ok)
    if not lines:
        lines.append("No issues detected — should render cleanly.")
    return "\n".join(lines)


def save_as_pdf(image_array, paper_size="letter", margin_in=0.5):
    """Render the deliverable image to a single-page PDF and return the path."""
    from PIL import Image

    if paper_size == "a4":
        page_w_in, page_h_in = 8.27, 11.69
    else:
        page_w_in, page_h_in = 8.5, 11.0
    dpi = 300
    page_w = int(page_w_in * dpi)
    page_h = int(page_h_in * dpi)
    margin = int(margin_in * dpi)

    img = Image.fromarray(image_array.astype(np.uint8))
    avail_w = page_w - 2 * margin
    avail_h = page_h - 2 * margin
    iw, ih = img.size
    scale = min(avail_w / iw, avail_h / ih)
    new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    page = Image.new("RGB", (page_w, page_h), "white")
    page.paste(img, ((page_w - new_w) // 2, (page_h - new_h) // 2))

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    page.save(path, "PDF", resolution=float(dpi))
    return path


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


def _resize_to_width(image, target_width):
    """Resize an image to a target width, preserving aspect ratio."""
    h, w = image.shape[:2]
    if w == target_width:
        return image
    target_h = max(1, int(round(h * (target_width / w))))
    return cv.resize(image, (target_width, target_h), interpolation=cv.INTER_AREA)


def _combine_grid_and_legend(grid, legend, title=None, thumbnail=None, padding=30):
    """Compose the deliverable: grid on top, footer below with title+legend on
    the left and an optional colored thumbnail on the right."""
    gh, gw = grid.shape[:2]
    lh, lw = legend.shape[:2]

    if thumbnail is not None:
        thumb_w = min(thumbnail.shape[1], max(200, gw // 4))
        thumb = _resize_to_width(thumbnail, thumb_w)
        th, tw = thumb.shape[:2]
    else:
        thumb = None
        th = tw = 0

    title_strip = _render_title(title, lw) if title and title.strip() else None

    left_h = lh + (title_strip.shape[0] + padding if title_strip is not None else 0)
    left_w = lw

    if thumb is not None:
        footer_w = left_w + padding * 2 + tw
        footer_h = max(left_h, th)
    else:
        footer_w = left_w
        footer_h = left_h

    width = max(gw, footer_w)
    total_h = gh + padding + footer_h
    canvas = np.ones((total_h, width, 3), dtype=np.uint8) * 255

    grid_x = (width - gw) // 2
    canvas[:gh, grid_x:grid_x + gw] = grid

    footer_x = (width - footer_w) // 2
    footer_y = gh + padding

    y = footer_y + (footer_h - left_h) // 2
    if title_strip is not None:
        ts_h, ts_w = title_strip.shape[:2]
        canvas[y:y + ts_h, footer_x:footer_x + ts_w] = title_strip
        y += ts_h + padding
    canvas[y:y + lh, footer_x:footer_x + lw] = legend

    if thumb is not None:
        thumb_x = footer_x + left_w + padding * 2
        thumb_y = footer_y + (footer_h - th) // 2
        canvas[thumb_y:thumb_y + th, thumb_x:thumb_x + tw] = thumb

    return canvas


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
    combined = _combine_grid_and_legend(
        numbered_islands,
        legend,
        title=title,
        thumbnail=colorbynumber_obj.simplified_image,
    )
    data = {
        "centroid_coords_list": colorbynumber_obj.centroid_coords_list,
        "color_id_list": [color_id for color_id, _ in colorbynumber_obj.island_borders_list]
    }
    pdf_path = save_as_pdf(combined)
    return combined, \
        colorbynumber_obj.simplified_image, \
        colorbynumber_obj.islands_image, \
        data, \
        pdf_path

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
    combined = _combine_grid_and_legend(
        numbered,
        legend,
        title=title,
        thumbnail=obj.filled_image,
    )

    data = {
        "mode": "pixel_grid",
        "indices": obj.indices.tolist(),
        "palette": [list(int(v) for v in c) for c in obj.palette],
        "cell_size": int(pixel_cell_size),
        "show_grid": bool(pixel_show_grid),
    }
    pdf_path = save_as_pdf(combined)
    return combined, obj.filled_image, obj.blank_grid_image, data, pdf_path


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
