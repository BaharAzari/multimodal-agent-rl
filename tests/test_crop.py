"""Synthetic, offline tests for deterministic image inspection."""

import pytest
from PIL import Image

from multimodal_agent_rl.tools import crop_and_zoom


@pytest.fixture
def rgb_image() -> Image.Image:
    image = Image.new("RGB", (6, 5))
    image.putdata([(x * 30, y * 40, x + y) for y in range(5) for x in range(6)])
    return image


def test_crop_uses_exclusive_right_and_bottom_edges(rgb_image: Image.Image) -> None:
    result = crop_and_zoom(rgb_image, (1, 2, 4, 5), zoom_factor=1)

    assert result.size == (3, 3)
    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (30, 80, 3)
    assert result.getpixel((2, 2)) == (90, 160, 7)


def test_crop_is_independent_and_does_not_mutate_input(rgb_image: Image.Image) -> None:
    original_pixels = rgb_image.tobytes()

    result = crop_and_zoom(rgb_image, (0, 0, 6, 5), zoom_factor=1)

    assert result is not rgb_image
    result.putpixel((0, 0), (255, 255, 255))
    assert rgb_image.tobytes() == original_pixels


def test_default_zoom_uses_bilinear_interpolation() -> None:
    image = Image.new("L", (2, 1))
    image.putdata([0, 100])

    result = crop_and_zoom(image, (0, 0, 2, 1))

    assert result.mode == "L"
    assert result.size == (4, 2)
    assert [result.getpixel((x, 0)) for x in range(4)] == [0, 25, 75, 100]
    assert [image.getpixel((x, 0)) for x in range(2)] == [0, 100]


def test_zoom_rounds_each_dimension_and_is_repeatable(rgb_image: Image.Image) -> None:
    result = crop_and_zoom(rgb_image, (1, 1, 4, 2), zoom_factor=1.5)
    repeated = crop_and_zoom(rgb_image, (1, 1, 4, 2), zoom_factor=1.5)

    assert result.size == (4, 2)  # round(4.5) == 4; round(1.5) == 2.
    assert result.tobytes() == repeated.tobytes()


def test_zoom_preserves_constant_rgb_pixels() -> None:
    image = Image.new("RGB", (3, 2), (20, 40, 60))

    result = crop_and_zoom(image, (1, 0, 3, 2), zoom_factor=3)

    assert result.size == (6, 6)
    assert result.mode == "RGB"
    assert result.getextrema() == ((20, 20), (40, 40), (60, 60))


def test_zoom_preserves_rgba_mode_and_alpha() -> None:
    image = Image.new("RGBA", (3, 2), (20, 40, 60, 128))

    result = crop_and_zoom(image, (1, 0, 3, 2), zoom_factor=3)

    assert result.size == (6, 6)
    assert result.mode == "RGBA"
    assert result.getchannel("A").getextrema() == (128, 128)
    assert image.getpixel((0, 0)) == (20, 40, 60, 128)


@pytest.mark.parametrize("mode", ["P", "1", "I;16"])
def test_unsupported_image_modes_require_explicit_conversion(mode: str) -> None:
    image = Image.new(mode, (3, 2))

    with pytest.raises(ValueError, match="image mode.*convert explicitly"):
        crop_and_zoom(image, (0, 0, 2, 2))


@pytest.mark.parametrize(
    "box",
    [
        (-1, 0, 2, 2),
        (0, -1, 2, 2),
        (0, 0, 7, 2),
        (0, 0, 2, 6),
        (2, 0, 2, 2),
        (3, 0, 2, 2),
        (0, 2, 2, 2),
        (0, 3, 2, 2),
    ],
)
def test_invalid_box_values_are_rejected(
    rgb_image: Image.Image, box: tuple[int, int, int, int]
) -> None:
    with pytest.raises(ValueError, match="positive area.*within"):
        crop_and_zoom(rgb_image, box)


@pytest.mark.parametrize("box", [(), (0, 0, 1), (0, 0, 1, 1, 2)])
def test_invalid_box_lengths_are_rejected(
    rgb_image: Image.Image, box: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="exactly four"):
        crop_and_zoom(rgb_image, box)  # type: ignore[arg-type]


@pytest.mark.parametrize("box", [[0, 0, 2, 2], "0,0,2,2", None])
def test_invalid_box_types_are_rejected(rgb_image: Image.Image, box: object) -> None:
    with pytest.raises(TypeError, match="tuple"):
        crop_and_zoom(rgb_image, box)  # type: ignore[arg-type]


@pytest.mark.parametrize("coordinate", [1.0, True, "1", None])
def test_noninteger_coordinates_are_rejected(
    rgb_image: Image.Image, coordinate: object
) -> None:
    with pytest.raises(TypeError, match="coordinates must be integers"):
        crop_and_zoom(rgb_image, (coordinate, 0, 3, 3))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factor", [0, -1, 0.99, float("inf"), float("-inf"), float("nan")]
)
def test_invalid_zoom_values_are_rejected(
    rgb_image: Image.Image, factor: float
) -> None:
    with pytest.raises(ValueError, match="finite.*greater than or equal to 1"):
        crop_and_zoom(rgb_image, (0, 0, 2, 2), factor)


@pytest.mark.parametrize("factor", [True, False, "2", None, 2 + 0j])
def test_invalid_zoom_types_are_rejected(
    rgb_image: Image.Image, factor: object
) -> None:
    with pytest.raises(TypeError, match="real number"):
        crop_and_zoom(rgb_image, (0, 0, 2, 2), factor)  # type: ignore[arg-type]


@pytest.mark.parametrize("factor", [1e308, 10**1000, 2**31])
def test_unrepresentable_output_sizes_are_rejected_before_allocation(
    rgb_image: Image.Image, factor: float
) -> None:
    with pytest.raises(ValueError, match="output dimensions"):
        crop_and_zoom(rgb_image, (0, 0, 2, 2), factor)


@pytest.mark.parametrize("image", [None, "image.png", [[0, 0], [0, 0]]])
def test_invalid_image_types_are_rejected(image: object) -> None:
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        crop_and_zoom(image, (0, 0, 1, 1))  # type: ignore[arg-type]
