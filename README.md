---
title: Color By Number
sdk: gradio
sdk_version: 4.25.0
app_file: app.py
license: mit
---

# Color by number

"[Color by number](https://en.wikipedia.org/wiki/Paint_by_number)" is a coloring activity in which the canvas is delineated into sections labelled with numbers and the user fills in these sections with colors corresponding to their numbers, in a manner similar to a coloring book. This project helps you create your own Color By Number kit for an image of your choice.

Two output styles are supported:

- **Islands** (original): free-form regions of color with smooth contours.
- **Pixel grid**: pixel-art / cross-stitch style — the image is downsampled to a grid of square cells, each labeled with its color number.

Everything runs **locally** with OpenCV + NumPy + k-means. **No AI API key (OpenAI, Anthropic, etc.) is required** — there is no model call anywhere in the pipeline.

Most existing Color By Number tools require buying art supplies specific to an image which makes it an expensive activity and can potentially result in wastage of paint. However, for many people, the goal is to create a rough approximation of an image and have fun painting/coloring without having to spend $30 on new art supplies for every new image. In this tool, you can specify the limited set of colors you already have at home and it generates a custom Color By Number kit for your image using only the colors you mention. You can print it and color it, all for free. See the [notebook](ColorByNumber.ipynb) for an example.

Choose images carefully and ensure that the limited art supplies you already have at home are indeed sufficient to create a good approximation of the image.

## Installing the environment

Download and install [Anaconda](https://www.anaconda.com/download).

```
conda env create -f environment.yml
conda activate ColorByNumber
```

## Generating your own Color By Number

Launch jupyter lab by running: `jupyter lab`. Open the `ColorByNumber.ipynb` notebook.

- Change `image_path` to your own image.
- Change `color_list` to the list of colors you have at home. You can get the RGB values of your colors by snapping a picture of your color set and using an online color picker (such as [this](https://imagecolorpicker.com/)).

Running the code in the notebook generates a "Color by number" for your image using your color palette. If the result is not satisfactory, try changing the `config` parameters. See [config.py](colorbynumber/config.py) for an explanation of the parameters.

## Pixel-grid (pixel-art) mode

Launch the Gradio app:

```
python app.py
```

In the UI, switch **Output style** to **Pixel grid**. Adjust:

- **Grid resolution** — number of cells along the longest side (more cells = more detail).
- **Cell size** — pixel size of each cell in the rendered output.
- **Show gridlines** — draw thin lines between cells.

You can also use it directly in Python:

```python
from colorbynumber.pixel_grid import PixelColorByNumber

obj = PixelColorByNumber(
    image_path="ExampleImages/Macaw.jpeg",
    num_colors=12,
)
numbered = obj.create_color_by_number()  # the printable color-by-number grid
preview  = obj.filled_image              # answer-key pixel-art preview
legend   = obj.generate_color_legend()
```
