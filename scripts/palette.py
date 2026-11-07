"""Bounded, deterministic source colors for the original Redthread palette roles.

This module chooses colors only. The renderer must retain the original photograph,
the canonical subject pixels, and the bundled paper texture.
"""
from __future__ import annotations

import colorsys
import statistics

from PIL import Image


def _hex(rgb: tuple[int, int, int]) -> str:
    return '#%02X%02X%02X' % rgb


def _rgb(hue: float, lightness: float, saturation: float) -> tuple[int, int, int]:
    return tuple(round(channel * 255) for channel in colorsys.hls_to_rgb(hue % 1, lightness, saturation))


def _luma(rgb: tuple[int, int, int]) -> float:
    return .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2]


def _median(pixels: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    return tuple(round(statistics.median(pixel[channel] for pixel in pixels)) for channel in range(3))


def _parse_theme(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str):
        raise ValueError('theme must be a hexadecimal #RRGGBB color')
    text = value.strip().removeprefix('#')
    if len(text) != 6 or any(char not in '0123456789abcdefABCDEF' for char in text):
        raise ValueError('theme must be a hexadecimal #RRGGBB color')
    return tuple(int(text[offset:offset + 2], 16) for offset in (0, 2, 4))


def _contrast(first: tuple[int, int, int], second: tuple[int, int, int]) -> float:
    def luminance(rgb):
        channels = [value / 255 for value in rgb]
        linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4 for value in channels]
        return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2]
    low, high = sorted((luminance(first), luminance(second)))
    return (high + .05) / (low + .05)


def derive_palette(source: Image.Image, mask: Image.Image | None = None, theme: str | None = None) -> dict:
    """Return ``theme``, ``accent``, ``silhouette``, ``glow`` and provenance.

    ``source`` must already have its display orientation applied. ``mask`` uses
    white/opaque pixels for the subject and must occupy the same source canvas.
    An explicit ``theme`` is preserved exactly; lighting still comes from source.
    Neither input image is changed. Sampling uses at most 128 x 128 pixels.
    """
    if source.width < 1 or source.height < 1:
        raise ValueError('source image must have nonzero dimensions')
    if mask is not None and mask.size != source.size:
        raise ValueError('palette mask must match the source image dimensions')
    sample = source.convert('RGB')
    sample.thumbnail((128, 128), Image.Resampling.BOX)
    raw = sample.tobytes()
    pixels = list(zip(raw[0::3], raw[1::3], raw[2::3]))
    selected = pixels
    region = 'whole-source'
    if mask is not None:
        alpha = mask.getchannel('A') if 'A' in mask.getbands() else mask.convert('L')
        values = alpha.resize(sample.size, Image.Resampling.NEAREST).tobytes()
        background = [pixel for pixel, value in zip(pixels, values) if value < 64]
        if len(background) >= max(1, round(len(pixels) * .03)):
            selected = background
            region = 'source-background'
    # Near-black voids and clipped white pixels describe exposure, not a useful
    # paper tint. Keep them only when the photograph has no midtone samples.
    midtones = [pixel for pixel in selected if 18 <= _luma(pixel) <= 238]
    representative = _median(midtones or selected)
    hue, lightness, saturation = colorsys.rgb_to_hls(*(value / 255 for value in representative))
    if theme is None:
        paper_rgb = _rgb(hue, min(.78, max(.65, .64 + .16 * lightness)), min(.22, saturation * .38))
        theme_source = region
    else:
        paper_rgb = _parse_theme(theme)
        theme_source = 'explicit-theme'

    # Stay in the vintage red family while moving toward brick/brown for warm
    # sources and wine red for cool sources. The accent is shared by all text
    # and by the authored connector, rather than recoloring those separately.
    warmth = (paper_rgb[0] - paper_rgb[2]) / max(1, max(paper_rgb) - min(paper_rgb))
    accent_hue = (5 + 13 * warmth) / 360
    accent_saturation = .44 + .10 * min(1, saturation)
    accent_lightness = .34
    accent = _rgb(accent_hue, accent_lightness, accent_saturation)
    # Darken in a fixed small search only if a user-supplied theme needs it.
    # Very dark explicit themes can remain unsuitable; report the measured
    # contrast instead of changing an explicit theme behind the user's back.
    for step in range(1, 11):
        if _contrast(paper_rgb, accent) >= 3:
            break
        accent = _rgb(accent_hue, accent_lightness - step * .02, accent_saturation)

    ordered = sorted(pixels, key=_luma)
    highlights = ordered[max(0, len(ordered) * 3 // 4):]
    neutral_highlights = [pixel for pixel in highlights if max(pixel) - min(pixel) <= 80]
    light_rgb = _median(neutral_highlights or highlights)
    temperature = (light_rgb[0] - light_rgb[2]) / max(32, _luma(light_rgb))
    if temperature >= .16:
        lighting, silhouette, glow = 'cream-white', '#FFF6DE', '#FFF6DA'
    elif temperature >= .05:
        lighting, silhouette, glow = 'warm-white', '#FDF7E9', '#FFF8EA'
    else:
        lighting, silhouette, glow = 'neutral-white', '#FAFAF8', '#FFFFFC'
    return {
        'theme': _hex(paper_rgb),
        'accent': _hex(accent),
        'silhouette': silhouette,
        'glow': glow,
        'metadata': {
            'method': 'source-median-muted-v1',
            'theme_source': theme_source,
            'source_representative': _hex(representative),
            'lighting': lighting,
            'lighting_sample': _hex(light_rgb),
            'sample_size': list(sample.size),
            'sampled_pixels': len(pixels),
            'theme_accent_contrast': round(_contrast(paper_rgb, accent), 3),
        },
    }
