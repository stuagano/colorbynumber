def color_selection_block():
    return """
    ---
    ## Color selection
    Limit the colors to the ones you already have in your paint palette, 
    or let the algorithm choose the best colors for you.
    """

def edit_coloring_page_block_header():
    return """
    ---
    ## Edit Coloring Page
    Change the font size and thickness of the numbers on the coloring page.
    """

def parameters_block_header():
    return """
    ---
    Configuration influences how the image is processed and thereby
    affects the complexity of the coloring page. You can use the default values
    to get started. Tweak them if nexessary to get the desired results.
    """

# Denoise parameters
def denoise_block_header():
    return """
    ## Denoise
    Removing noise from the image can help in simplifying the image. 
    Not recommended for some images where sharp edges are to be preserved 
    (e.g. Grid image in above example).
    """

def output_style_block():
    return """
    ---
    ## Output style
    **Islands**: free-form regions of color, the original style.
    **Pixel grid**: pixel-art style — the image is downsampled to a grid of
    square cells, each labeled with a color number. Great for cross-stitch,
    bead patterns, or just a chunky pixel-art coloring page.
    """

def pixel_grid_parameters():
    return """
    ## Pixel Grid
    Tune the resolution and look of the pixel-art grid.
    """

def simplify_islands_parameters():
    return """
    ## Simplify Islands
    The individual blocks of colors are _Islands_. 
    The parameters below influence the simplification of these islands 
    and determine the complexity of the coloring page.
    """



