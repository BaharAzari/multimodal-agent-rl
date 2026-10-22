"""Deterministic crop and zoom with explicit pixel coordinates."""

from math import isfinite
from numbers import Real

from PIL import Image

_MAX_IMAGE_DIMENSION = 2**31 - 1


def crop_and_zoom(
    image: Image.Image,
    box: tuple[int, int, int, int],
    zoom_factor: float = 2.0,
) -> Image.Image:
    """Return an independent crop enlarged with bilinear resampling.

    Only ``L``, ``RGB``, and ``RGBA`` images are supported. The image mode is
    preserved; this tool never implicitly converts colors or image modes.

    ``box`` is ``(x1, y1, x2, y2)`` in original-image pixels, with an
    inclusive upper-left corner and exclusive lower-right corner. Coordinates
    must be integers (not booleans); the box must have positive area and lie
    entirely inside the image. Out-of-bounds coordinates are never clamped.

    ``zoom_factor`` must be a finite real number greater than or equal to one.
    Each output dimension is ``max(1, round(crop_dimension * zoom_factor))``;
    Python's round-to-even rule applies. An output with the original crop size
    is returned without resampling. Output dimensions must fit Pillow's signed
    32-bit size representation. The input image is never modified.

    To map continuous output coordinates back to original-image coordinates,
    use the realized scale in each axis: ``x = x1 + u / (output_width /
    (x2 - x1))`` and ``y = y1 + v / (output_height / (y2 - y1))``. These are
    pixel-edge coordinates; rounding means the realized scale may differ from
    ``zoom_factor``. Bilinear sampling itself uses pixel-center conventions.

    Raises:
        TypeError: If the image, box, coordinates, or factor have invalid types.
        ValueError: If the mode, box shape/bounds, or factor value is invalid,
            or the requested output dimensions cannot be represented.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if image.mode not in {"L", "RGB", "RGBA"}:
        raise ValueError("image mode must be L, RGB, or RGBA; convert explicitly")
    if not isinstance(box, tuple):
        raise TypeError("box must be a tuple of four integer coordinates")
    if len(box) != 4:
        raise ValueError("box must contain exactly four coordinates")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in box):
        raise TypeError("box coordinates must be integers, not booleans")

    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
        raise ValueError("box must have positive area and lie within the image")

    if isinstance(zoom_factor, bool) or not isinstance(zoom_factor, Real):
        raise TypeError("zoom_factor must be a real number, not a boolean")
    try:
        finite_factor = isfinite(zoom_factor)
    except OverflowError:
        raise ValueError(
            "zoom_factor is too large to represent output dimensions"
        ) from None
    if not finite_factor or zoom_factor < 1:
        raise ValueError("zoom_factor must be finite and greater than or equal to 1")

    try:
        scaled_size = ((x2 - x1) * zoom_factor, (y2 - y1) * zoom_factor)
        if not all(isfinite(dimension) for dimension in scaled_size):
            raise ValueError("zoom_factor produces nonfinite output dimensions")
        output_size = tuple(max(1, round(dimension)) for dimension in scaled_size)
    except OverflowError:
        raise ValueError(
            "zoom_factor is too large to represent output dimensions"
        ) from None
    if any(dimension > _MAX_IMAGE_DIMENSION for dimension in output_size):
        raise ValueError("output dimensions must be at most 2**31 - 1 pixels")

    cropped = image.crop(box)
    if output_size == cropped.size:
        return cropped
    return cropped.resize(output_size, resample=Image.Resampling.BILINEAR)
