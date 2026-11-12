#!/usr/bin/env python3
"""Render classic Redthread; generate only missing upper-panel background."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat, __version__ as PILLOW_VERSION
except ImportError:
    print('DEPENDENCY_MISSING: Install Pillow: python -m pip install Pillow', file=sys.stderr)
    raise SystemExit(2)

import check_font_glyphs as glyphs
import recolor_theme_paper as paper
import render_connector as connector
from palette import derive_palette, _contrast as palette_contrast

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / 'assets/fonts/dartsfont/DartsFont-Regular.ttf'
PAPER = ROOT / 'assets/paper/theme-fiber-paper.jpg'
TEMPLATES = ROOT / 'assets/connectors/templates.json'
CANVAS = (1200, 1600)
PANEL = (1200, 800)
ACCENT = '#8F2634'


class RenderError(ValueError):
    def __init__(self, code, message, remedy='', details=None):
        self.code, self.message, self.remedy = code, message, remedy
        self.details = details or {}
        super().__init__(f'{code}: {message}' + (f' {remedy}' if remedy else ''))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_point(value, field):
    if (not isinstance(value, list) or len(value) != 2 or
            any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1 for v in value)):
        raise RenderError('INVALID_CONFIG', f'{field} must contain two numbers between 0 and 1.')
    return value


def normalized_box(value, field):
    if (not isinstance(value, list) or len(value) != 4 or
            any(not isinstance(v, (int, float)) or isinstance(v, bool) or
                not math.isfinite(v) or not 0 <= v <= 1 for v in value) or
            value[0] >= value[2] or value[1] >= value[3]):
        raise RenderError('INVALID_CONFIG', f'{field} must be [left, top, right, bottom] within 0–1, with positive width and height.')
    return value


def read_config(path):
    try:
        if str(path) == '-':
            text, base_dir = sys.stdin.read(), Path.cwd()
        else:
            path = Path(path).expanduser().resolve()
            text, base_dir = path.read_text(encoding='utf-8'), path.parent
        data = json.loads(text)
    except (OSError, ValueError) as exc:
        raise RenderError('INVALID_CONFIG', str(exc), 'Provide a readable JSON brief, or pass a JSON object on stdin with --config -.') from exc
    return validate_config(data, base_dir)


def read_config_json(text):
    """Parse an inline JSON brief as data; resolve image paths from the cwd."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise RenderError('INVALID_CONFIG', str(exc), 'Pass a valid JSON object to --config-json.') from exc
    return validate_config(data, Path.cwd())


def artifact_relative_path(path):
    """Return a cwd-relative POSIX path, or None for a file outside the cwd."""
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return None


def validate_config(data, base_dir):
    if not isinstance(data, dict):
        raise RenderError('INVALID_CONFIG', 'The brief must be a JSON object.')
    unknown = set(data) - {'source', 'mask', 'english', 'japanese', 'theme', 'accent', 'silhouette', 'glow', 'seed', 'anchor', 'subject_point', 'segmentation', 'head_box', 'upper_body_box', 'protected_boxes', 'background'}
    if unknown:
        raise RenderError('INVALID_CONFIG', 'Unknown fields: ' + ', '.join(sorted(unknown)))
    if not isinstance(data.get('source'), str) or not data['source']:
        raise RenderError('INVALID_CONFIG', 'source must be an image path.')
    for name, allowed in [('english', (3, 4)), ('japanese', (2,))]:
        parts = data.get(name)
        if not isinstance(parts, list) or len(parts) not in allowed or any(not isinstance(t, str) or not t.strip() for t in parts):
            raise RenderError('INVALID_CONFIG', f'{name} requires {allowed} nonempty text fragments.')
        data[name] = [t.strip() for t in parts]
    if any(len(t) > 40 or '\n' in t for t in data['english']):
        raise RenderError('COPY_TOO_LONG', 'Each English fragment must be one line, at most 40 characters.', 'Use 3–4 short phrases; 2–3 words per phrase usually fit.')
    if any(len(t) > 40 or len(t.splitlines()) > 3 or any(len(s) > 12 for s in t.splitlines()) for t in data['japanese']):
        raise RenderError('COPY_TOO_LONG', 'Each Japanese block supports up to three short lines, at most 12 characters per line.', 'Use explicit \\n line breaks or shorter text.')
    for name in ('source', 'mask'):
        if name in data:
            if not isinstance(data[name], str) or not data[name]:
                raise RenderError('INVALID_CONFIG', f'{name} must be an image path.')
            p = Path(data[name]).expanduser()
            data[name] = p.resolve() if p.is_absolute() else (base_dir / p).resolve()
    if 'background' in data:
        import background_fill
        try:
            data['background'] = background_fill.validate_config(data['background'], base_dir)
        except ValueError as exc:
            raise RenderError(getattr(exc, 'code', 'INVALID_CONFIG'), getattr(exc, 'message', str(exc)),
                              getattr(exc, 'remedy', ''), getattr(exc, 'details', {})) from exc
        if 'image' in data['background']:
            image_path = Path(data['background']['image']).expanduser()
            data['background']['image'] = image_path.resolve() if image_path.is_absolute() else (base_dir / image_path).resolve()
    for name in ('anchor', 'subject_point'):
        if name in data:
            normalized_point(data[name], name)
    if 'head_box' in data:
        normalized_box(data['head_box'], 'head_box')
    if 'upper_body_box' in data:
        normalized_box(data['upper_body_box'], 'upper_body_box')
        body, head = data['upper_body_box'], data.get('head_box')
        if head is not None and any(body[i] > head[i] if i < 2 else body[i] < head[i] for i in range(4)):
            raise RenderError('INVALID_CONFIG', 'upper_body_box must include the complete head_box and the visible body down to at least the waist, when present.')
    if 'protected_boxes' in data:
        if not isinstance(data['protected_boxes'], list) or len(data['protected_boxes']) > 8:
            raise RenderError('INVALID_CONFIG', 'protected_boxes must be a list of at most eight rectangles.')
        for box in data['protected_boxes']:
            normalized_box(box, 'protected_boxes entry')
    if 'theme' in data:
        try:
            paper.parse_color(data['theme'])
        except (ValueError, TypeError, AttributeError, argparse.ArgumentTypeError) as exc:
            raise RenderError('INVALID_CONFIG', 'theme must be a hexadecimal color such as #B4B9A8.') from exc
    for field in ('accent', 'silhouette', 'glow'):
        if field in data:
            value = data[field]
            if (not isinstance(value, str) or len(value) != 7 or value[0] != '#' or
                    any(character not in '0123456789abcdefABCDEF' for character in value[1:])):
                raise RenderError('INVALID_CONFIG', f'{field} must be a six-digit hexadecimal color such as #8F2634.')
    if not isinstance(data.get('seed', 29), int) or isinstance(data.get('seed'), bool):
        raise RenderError('INVALID_CONFIG', 'seed must be an integer.')
    data.setdefault('seed', 29)
    if 'mask' in data:
        # A supplied mask completely bypasses segmentation configuration, including
        # a stale or invalid host setting. Record only the route used for replay.
        data.pop('segmentation', None)
    else:
        if 'segmentation' not in data and 'REDTHREAD_SEGMENTATION' in os.environ:
            try:
                data['segmentation'] = json.loads(os.environ['REDTHREAD_SEGMENTATION'])
            except (ValueError, TypeError) as exc:
                raise RenderError('INVALID_CONFIG', 'REDTHREAD_SEGMENTATION must contain a segmentation JSON object.',
                                  'Configure the existing segmenter command, or provide mask.') from exc
        if 'segmentation' in data:
            data['segmentation'] = validate_segmentation(data['segmentation'], data.get('subject_point'))
    return data


def validate_segmentation(value, subject_point=None):
    if not isinstance(value, dict) or set(value) - {'command', 'output', 'timeout'}:
        raise RenderError('INVALID_CONFIG', 'segmentation must be an object with command, output and timeout fields only.')
    command = value.get('command')
    if (not isinstance(command, list) or not command or
            any(not isinstance(arg, str) or not arg or '\0' in arg for arg in command)):
        raise RenderError('INVALID_CONFIG', 'segmentation.command must be a nonempty list of nonempty argument strings.',
                          'Use the trusted host interface as argv; no shell is started.')
    for token in ('{source}', '{output}'):
        if not any(token in arg for arg in command):
            raise RenderError('INVALID_CONFIG', f'segmentation.command must contain {token}.')
    point_tokens = [any(token in arg for arg in command) for token in ('{subject_x}', '{subject_y}')]
    if any(point_tokens) and (not all(point_tokens) or subject_point is None):
        raise RenderError('INVALID_CONFIG', 'Both {subject_x} and {subject_y} require subject_point in the brief.')
    if subject_point is not None and not all(point_tokens):
        raise RenderError('INVALID_CONFIG', 'A configured command must accept both {subject_x} and {subject_y} when subject_point is supplied.',
                          'Use an adapter that selects the requested subject, or provide an explicit mask. For a fixed-subject segmenter, omit subject_point only when its selected subject is already correct.')
    output = value.get('output', 'mask')
    if output not in ('mask', 'alpha'):
        raise RenderError('INVALID_CONFIG', 'segmentation.output must be mask or alpha.')
    timeout = value.get('timeout', 180)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 1800:
        raise RenderError('INVALID_CONFIG', 'segmentation.timeout must be an integer from 1 to 1800 seconds.')
    return {'command': list(command), 'output': output, 'timeout': timeout}


def configured_mask(config, source, work):
    adapter = config['segmentation']
    # A fresh output path prevents an unsuccessful command from reusing a mask
    # left by an earlier run. Temporary model RGB pixels never enter composition.
    with tempfile.TemporaryDirectory(prefix='segmentation-', dir=work) as directory:
        srcfile = Path(directory) / 'source.png'
        maskfile = Path(directory) / 'output.png'
        source.save(srcfile)
        replacements = {'{source}': str(srcfile), '{output}': str(maskfile)}
        if 'subject_point' in config:
            replacements.update(zip(('{subject_x}', '{subject_y}'), map(str, config['subject_point'])))
        command = []
        for argument in adapter['command']:
            for token, replacement in replacements.items():
                argument = argument.replace(token, replacement)
            command.append(argument)
        try:
            result = subprocess.run(command, capture_output=True, text=True, errors='replace',
                                    timeout=adapter['timeout'], shell=False)
        except FileNotFoundError as exc:
            raise RenderError('SEGMENTATION_COMMAND_MISSING', 'The configured segmentation executable was not found.',
                              'Correct segmentation.command to the installed host interface; this does not mean the sandbox lacks a model.') from exc
        except subprocess.TimeoutExpired as exc:
            raise RenderError('SEGMENTATION_TIMEOUT', f'The configured segmenter exceeded {adapter["timeout"]} seconds.',
                              'Check the host segmenter, adjust its timeout, or provide mask.') from exc
        except OSError as exc:
            raise RenderError('SEGMENTATION_FAILED', f'The configured segmenter could not start: {exc}',
                              'Check executable permissions and the host command interface.') from exc
        if result.returncode:
            detail = result.stderr.strip()[-1800:]
            raise RenderError('SEGMENTATION_FAILED', f'The configured segmenter exited with code {result.returncode}. {detail}',
                              'Check the existing segmenter command and model configuration; do not substitute image generation.')
        if not maskfile.is_file():
            raise RenderError('SEGMENTATION_OUTPUT_MISSING', 'The configured segmenter did not write its {output} file.',
                              'Use the installed command interface that writes a same-size mask or RGBA image to {output}.')
        try:
            with Image.open(maskfile) as result_image:
                if result_image.size != source.size:
                    raise RenderError('MASK_SIZE', f'Segmentation output {result_image.size} does not match oriented source {source.size}.',
                                      'Return output at the EXIF-oriented source dimensions; do not resize it silently.')
                if adapter['output'] == 'alpha':
                    if result_image.mode != 'RGBA':
                        raise RenderError('SEGMENTATION_OUTPUT_MODE', 'alpha output requires an RGBA image with a subject alpha channel.',
                                          'Return RGBA, or configure mask output with a grayscale image.')
                    mask = result_image.getchannel('A')
                else:
                    if result_image.mode not in ('1', 'L'):
                        raise RenderError('SEGMENTATION_OUTPUT_MODE', f'mask output requires an L or 1 grayscale image, received {result_image.mode}.',
                                          'Configure the segmenter to export a white-foreground grayscale mask, or use alpha for RGBA.')
                    mask = result_image.convert('L')
        except OSError as exc:
            raise RenderError('SEGMENTATION_OUTPUT_INVALID', 'The configured segmenter output is not a readable image.') from exc
        if mask.getextrema()[1] < 128:
            raise RenderError('EMPTY_MASK', 'The configured segmenter did not return a foreground subject.')
        if mask.getextrema() == (255, 255):
            raise RenderError('FULL_MASK', 'The configured segmenter returned an entirely opaque mask.',
                              'Return a foreground subject mask with background pixels, not the full image rectangle.')
        return mask, {'method': 'configured-command', 'output': adapter['output'],
                      'sha256': sha256(maskfile),
                      'subject_selection': {'normalized_point': config['subject_point']} if 'subject_point' in config else 'host-segmenter default; inspect mask visually'}


def check_glyphs(texts):
    if not FONT.is_file():
        raise RenderError('ASSET_MISSING', str(FONT), 'Restore the original assets directory.')
    missing = []
    try:
        cmaps = glyphs.unicode_cmaps(FONT)
        for char in ''.join(texts):
            if char.isspace():
                continue
            if not any((fmt == 4 and glyphs.glyph4(table, ord(char))) or
                       (fmt == 12 and glyphs.glyph12(table, ord(char))) for fmt, table in cmaps):
                missing.append(char)
    except (OSError, ValueError, IndexError, struct.error, SystemExit) as exc:
        raise RenderError('ASSET_INVALID', f'The bundled DartsFont cannot be parsed: {exc}', 'Restore assets/fonts/dartsfont/DartsFont-Regular.ttf from the original package.') from exc
    if missing:
        raise RenderError('MISSING_GLYPHS', 'DartsFont does not contain: ' + ''.join(dict.fromkeys(missing)), 'Rewrite these characters using the original font’s supported letters or kana; do not substitute the font.')


def supplied_mask(path, source_size):
    try:
        with Image.open(path) as original:
            im = ImageOps.exif_transpose(original)
            if im.size != source_size:
                raise RenderError('MASK_SIZE', f'Mask {im.size} does not match oriented source {source_size}.',
                                  'Ask the host segmentation tool for a full-size mask or transparent PNG. Do not crop, resize or guess its position.')
            alpha = None
            if im.mode in ('RGBA', 'LA'):
                alpha = im.getchannel('A')
            elif im.mode == 'P' and 'transparency' in im.info:
                alpha = im.convert('RGBA').getchannel('A')
            if alpha is not None and alpha.getextrema()[0] < 255:
                # A host cutout contributes only alpha, never its modified RGB.
                mask, channel = alpha, 'alpha'
            elif im.mode in ('1', 'L', 'LA'):
                mask, channel = im.convert('L'), 'grayscale'
            elif im.mode in ('RGB', 'RGBA', 'P'):
                red, green, blue = im.convert('RGB').split()
                if ImageChops.difference(red, green).getbbox() or ImageChops.difference(red, blue).getbbox():
                    raise RenderError('MASK_FORMAT', 'The supplied mask is a color image without usable transparency.',
                                      'Use the host tool\'s full-size transparent cutout or a grayscale mask with white foreground. An opaque color photo is not a mask.')
                mask, channel = red, 'grayscale'
            else:
                raise RenderError('MASK_FORMAT', f'Unsupported mask image mode: {im.mode}.',
                                  'Supply an 8-bit grayscale mask or a full-size transparent PNG from the host segmentation tool.')
    except OSError as exc:
        raise RenderError('MASK_UNREADABLE', str(exc), 'Supply a readable grayscale mask or full-size transparent PNG.') from exc
    extrema = mask.getextrema()
    if extrema[1] < 128:
        raise RenderError('EMPTY_MASK', 'The supplied mask or transparent cutout does not contain a foreground subject.')
    if extrema == (255, 255):
        raise RenderError('FULL_MASK', 'The supplied mask is entirely white.',
                          'Use a subject mask with background pixels; a full image rectangle does not select a subject.')
    return mask, channel


def get_mask(config, source, work):
    if 'mask' in config:
        mask, channel = supplied_mask(config['mask'], source.size)
        return mask, {'method': 'supplied-mask', 'sha256': sha256(config['mask']), 'mask_channel': channel, 'subject_selection': 'caller-supplied; inspect mask visually'}
    if 'segmentation' in config:
        return configured_mask(config, source, work)
    if platform.system() != 'Darwin':
        raise RenderError('SEGMENTATION_NOT_CONFIGURED', 'This invocation has no supplied mask or configured local segmentation command.',
                          'Use the host\'s existing subject-segmentation tool and pass its full-size grayscale mask or transparent PNG in mask. An existing command adapter is optional. This error does not establish that the sandbox lacks a segmentation model.')
    if not shutil.which('swift'):
        raise RenderError('SWIFT_MISSING', 'The built-in Apple Vision route cannot start because swift is unavailable.',
                          'Use the host\'s existing subject-segmentation tool and pass its full-size grayscale mask or transparent PNG in mask. An existing command adapter is optional.')
    srcfile = work / 'normalized-source.png'
    source.save(srcfile)
    maskfile = work / 'source-mask.png'
    command = ['swift', str(Path(__file__).with_name('segment_macos.swift')), str(srcfile), str(maskfile)]
    if 'subject_point' in config:
        command += [str(v) for v in config['subject_point']]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired as exc:
        raise RenderError('SEGMENTATION_TIMEOUT', 'Vision segmentation did not finish within 180 seconds.', 'Provide an external mask.') from exc
    if result.returncode:
        detail = result.stderr.strip()[-1800:]
        code = 'MULTIPLE_SUBJECTS' if 'MULTIPLE_SUBJECTS' in detail else 'SEGMENTATION_FAILED'
        if 'SUBJECT_POINT_BACKGROUND' in detail:
            code = 'SUBJECT_POINT_BACKGROUND'
        raise RenderError(code, detail, 'Set subject_point to a visible point inside the intended person, or provide a mask.')
    try:
        metadata = json.loads(result.stdout.strip().splitlines()[-1])
        mask = Image.open(maskfile).convert('L')
    except (ValueError, OSError, IndexError) as exc:
        raise RenderError('SEGMENTATION_FAILED', 'Segmenter did not return a valid mask and report.') from exc
    return mask, {'method': 'apple-vision-single-instance', **metadata, 'sha256': sha256(maskfile)}


EDGE_ALPHA_THRESHOLD = 5


def _edge_evidence_bbox(alpha):
    """Read cut evidence without changing the alpha used to render or protect."""
    return alpha.point(lambda value: 255 if value >= EDGE_ALPHA_THRESHOLD else 0).getbbox()


def framing_candidates(source, mask, head_box=None, protected_boxes=None, anchor=None, upper_body_box=None,
                       *, _margin_fraction=1, _background_scale=None):
    """Prefer the original crop; fit protected source pixels before filling gaps."""
    if source.size != mask.size:
        raise RenderError('MASK_SIZE', f'Mask {mask.size} does not match oriented source {source.size}.', 'Create a mask from the EXIF-oriented source, at the same pixel dimensions.')
    if mask.getbbox() is None or mask.getextrema()[1] < 128:
        raise RenderError('EMPTY_MASK', 'The mask does not contain a foreground subject.')
    cover_scale = max(PANEL[0] / source.width, PANEL[1] / source.height)
    scale = cover_scale if _background_scale is None else _background_scale
    resized = (math.ceil(source.width * scale), math.ceil(source.height * scale))
    # Integer image dimensions can differ from the ideal scale by under one
    # pixel. Safety checks must use the transform Pillow actually applies.
    raster_scale = [resized[i] / source.size[i] for i in (0, 1)]
    source_bbox = mask.getbbox()
    edge_bbox = _edge_evidence_bbox(mask)
    source_edge_cuts = [edge_bbox[0] == 0, edge_bbox[1] == 0,
                        edge_bbox[2] == source.width, edge_bbox[3] == source.height]
    touches_bottom = edge_bbox[3] >= source.height - 2
    fit_scale = min(PANEL[0] / (source_bbox[2] - source_bbox[0]), PANEL[1] / (source_bbox[3] - source_bbox[1]))
    geometry = {'source_size': list(source.size), 'subject_bbox': list(source_bbox),
                'panel_size': list(PANEL), 'alpha_support': 'all nonzero source alpha', 'minimum_cover_scale': cover_scale,
                'edge_evidence_alpha_threshold': EDGE_ALPHA_THRESHOLD,
                'source_edge_cuts': source_edge_cuts, 'safety_margins_reduced': _margin_fraction == 0,
                'maximum_subject_scale': fit_scale,
                'subject_size_at_cover': [(source_bbox[2] - source_bbox[0]) * cover_scale,
                                          (source_bbox[3] - source_bbox[1]) * cover_scale],
                'background_size_at_subject_fit': [source.width * fit_scale, source.height * fit_scale]}
    def crop_bounds(box):
        result = []
        for i in (0, 1):
            low = math.ceil(box[i + 2] * raster_scale[i] - PANEL[i] - 1e-8)
            high = math.floor(box[i] * raster_scale[i] + 1e-8)
            # Visible source cuts must touch the canvas, or remain outside it
            # after an allowed body crop. They must never float inside a gap.
            if _background_scale is None or source_edge_cuts[i]:
                low = max(0, low)
            if _background_scale is None or source_edge_cuts[i + 2]:
                high = min(resized[i] - PANEL[i], high)
            result.append((low, high))
        return result

    def feasible(bounds):
        return all(low <= high for low, high in bounds)

    def union(a, b):
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

    def pixels(box):
        return [box[i] * source.size[i % 2] for i in range(4)]

    head_pixels, head_margin, required = None, 0, None
    if head_box is not None:
        normalized_box(head_box, 'head_box')
        head_pixels = pixels(head_box)
        region = (math.floor(head_pixels[0]), math.floor(head_pixels[1]),
                  math.ceil(head_pixels[2]), math.ceil(head_pixels[3]))
        if mask.crop(region).getbbox() is None:
            raise RenderError('HEAD_REGION_INVALID', 'The supplied head box contains no foreground alpha.',
                              'Check the complete-head annotation and aligned mask; this is not semantic head detection.')
        head_margin = _margin_fraction * max(2, .02 * max(head_pixels[2] - head_pixels[0], head_pixels[3] - head_pixels[1]))
        panel_margin = head_margin * max(raster_scale)
        required = [max(0, head_pixels[0] - panel_margin / raster_scale[0]),
                    max(0, head_pixels[1] - panel_margin / raster_scale[1]),
                    min(source.width, head_pixels[2] + panel_margin / raster_scale[0]),
                    min(source.height, head_pixels[3] + panel_margin / raster_scale[1])]
        geometry.update({'head_box_source': head_pixels, 'head_margin_source_pixels': head_margin})
    if upper_body_box is not None:
        normalized_box(upper_body_box, 'upper_body_box')
        body_pixels = pixels(upper_body_box)
        if head_pixels and any(body_pixels[i] > head_pixels[i] if i < 2 else body_pixels[i] < head_pixels[i] for i in range(4)):
            raise RenderError('INVALID_CONFIG', 'upper_body_box must include head_box.')
    else:
        # Older callers remain safe: without a waist annotation, keep the
        # entire visible subject instead of guessing an anatomical waist.
        body_pixels = list(source_bbox)
    required = body_pixels if required is None else union(required, body_pixels)
    geometry['upper_body_box_source'] = body_pixels
    geometry['body_protection_source'] = 'upper_body_box' if upper_body_box is not None else 'complete-alpha-bounds'
    for box in protected_boxes or []:
        normalized_box(box, 'protected_boxes entry')
        values = pixels(box)
        padded = [max(0, values[0] - 2 * _margin_fraction), max(0, values[1] - 2 * _margin_fraction),
                  min(source.width, values[2] + 2 * _margin_fraction), min(source.height, values[3] + 2 * _margin_fraction)]
        required = padded if required is None else union(required, padded)
    if anchor is not None:
        normalized_point(anchor, 'anchor')
        ax, ay = anchor[0] * source.width, anchor[1] * source.height
        point_box = [max(0, ax - 8 * _margin_fraction / raster_scale[0]), max(0, ay - 8 * _margin_fraction / raster_scale[1]),
                     min(source.width, ax + 8 * _margin_fraction / raster_scale[0]), min(source.height, ay + 8 * _margin_fraction / raster_scale[1])]
        required = point_box if required is None else union(required, point_box)
    if required[1] > edge_bbox[1]:
        # An underestimated annotation must not introduce a new top cut.
        # This is read-only alpha evidence, not semantic head detection.
        geometry['mask_top_guard'] = {'source_y': edge_bbox[1], 'previous_required_y': required[1]}
        required = [required[0], edge_bbox[1], required[2], required[3]]
    geometry.update({'protected_boxes': protected_boxes or [], 'requested_anchor': anchor,
                     'required_region_source': required})
    if required is not None and (required[0] < 0 or required[1] < 0 or required[2] > source.width or
                                 required[3] > source.height or not feasible(crop_bounds(required))):
        if _margin_fraction:
            # Padding is preferred context, never a reason to cut or reject a
            # real head, body or prop that fits without that optional padding.
            return framing_candidates(source, mask, head_box, protected_boxes, anchor, upper_body_box,
                                      _margin_fraction=0, _background_scale=_background_scale)
        if _background_scale is None:
            # Two output pixels cover integer resize/offset rounding. The same
            # source and mask transform is used; no generated subject is used.
            fit = min((PANEL[i] - 2) / (required[i + 2] - required[i]) for i in (0, 1))
            return framing_candidates(source, mask, head_box, protected_boxes, anchor, upper_body_box,
                                      _background_scale=min(cover_scale, fit))
        opposite = any(source_edge_cuts[i] and source_edge_cuts[i + 2] and resized[i] < PANEL[i]
                       for i in (0, 1))
        raise RenderError('SOURCE_EDGE_CONFLICT' if opposite else 'ORIGINAL_LAYOUT_CONFLICT',
                          'Opposite source-edge cuts cannot both meet the canvas after the protected region is fitted.' if opposite else
                          'The protected source region and existing source-edge cuts cannot share a valid panel transform.',
                          'Keep the original subject pixels. Report this measured limit; background generation cannot complete a cut body or prop.', geometry)
    complete = source_bbox if required is None else union(source_bbox, required)
    bounds = crop_bounds(complete)
    body_crop = not feasible(bounds)
    if body_crop:
        bounds = crop_bounds(required)
    preferred = [(resized[0] - PANEL[0]) // 2,
                 resized[1] - PANEL[1] if touches_bottom else (resized[1] - PANEL[1]) // 2]
    if body_crop:
        # Favor useful headroom and torso, then try the two legal extremes.
        preferred[1] = round(required[1] * raster_scale[1] - 48)
    if _background_scale is not None:
        for i in (0, 1):
            if source_edge_cuts[i]:
                preferred[i] = 0
            elif source_edge_cuts[i + 2]:
                preferred[i] = resized[i] - PANEL[i]
            elif resized[i] < PANEL[i]:
                # One missing strip needs one seam instead of two. Keep the
                # existing scale and bounds; only prefer a legal flush edge.
                edges = [offset for offset in (0, resized[i] - PANEL[i])
                         if bounds[i][0] <= offset <= bounds[i][1]]
                if edges:
                    center = (required[i] + required[i + 2]) * raster_scale[i] / 2
                    preferred[i] = min(edges, key=lambda offset: abs(center - offset - PANEL[i] / 2))
    axes = [list(dict.fromkeys((max(low, min(high, wanted)), low, high)))
            for (low, high), wanted in zip(bounds, preferred)]
    offsets = [(x, y) for x in axes[0] for y in axes[1]]
    offsets.sort(key=lambda p: abs(p[0] - preferred[0]) + abs(p[1] - preferred[1]))
    frames = []
    for x, y in offsets:
        frame = {'uniform_scale': scale, 'raster_scale': raster_scale, 'resized': list(resized), 'crop_offset': [x, y],
                 'bottom_aligned': touches_bottom and y == resized[1] - PANEL[1],
                 'geometry': geometry, 'crop_candidates': len(offsets), 'body_crop': body_crop,
                 'layout_mode': 'full-bleed' if _background_scale is None else 'background-fill',
                 'photo_rect': [max(0, -x), max(0, -y), min(PANEL[0], resized[0] - x), min(PANEL[1], resized[1] - y)],
                 'source_edges': [edge_bbox[0] * raster_scale[0] <= x, edge_bbox[1] * raster_scale[1] <= y,
                                  edge_bbox[2] * raster_scale[0] >= x + PANEL[0], edge_bbox[3] * raster_scale[1] >= y + PANEL[1]]}
        if head_pixels is not None:
            box = [head_pixels[i] * raster_scale[i % 2] - (x, y)[i % 2] for i in range(4)]
            clearance = [box[0], box[1], PANEL[0] - box[2], PANEL[1] - box[3]]
            margins = [min(panel_margin, head_pixels[0] * raster_scale[0]),
                       min(panel_margin, head_pixels[1] * raster_scale[1]),
                       min(panel_margin, (source.width - head_pixels[2]) * raster_scale[0]),
                       min(panel_margin, (source.height - head_pixels[3]) * raster_scale[1])]
            frame['head_crop_check'] = {'head_box_source': head_pixels, 'head_box_panel': box,
                                       'head_inside': all(c >= m - 1e-8 for c, m in zip(clearance, margins)),
                                       'clearance_pixels': clearance, 'margin_pixels': min(margins), 'margins_pixels': margins}
        frame['upper_body_check'] = {'box_source': body_pixels,
                                    'box_panel': [body_pixels[i] * raster_scale[i % 2] - (x, y)[i % 2] for i in range(4)],
                                    'annotation_source': geometry['body_protection_source']}
        frames.append(frame)
    return frames


def prepare_subject(source, mask, framing=None):
    framing = dict(framing or framing_candidates(source, mask)[0])
    scale, resized = framing['uniform_scale'], tuple(framing['resized'])
    left, top = framing['crop_offset']
    rect = framing.get('photo_rect', [0, 0, *PANEL])
    crop = (left + rect[0], top + rect[1], left + rect[2], top + rect[3])
    photo = Image.new('RGBA', PANEL)
    alpha = Image.new('L', PANEL)
    photo.paste(source.convert('RGB').resize(resized, Image.Resampling.LANCZOS).crop(crop), tuple(rect[:2]))
    alpha.paste(mask.resize(resized, Image.Resampling.LANCZOS).crop(crop), tuple(rect[:2]))
    bbox = alpha.getbbox()
    if bbox is None or bbox[2] - bbox[0] < 90 or bbox[3] - bbox[1] < 180:
        raise RenderError('SUBJECT_TOO_SMALL', 'The subject is too small after the fixed panel crop.', 'Use a closer portrait with enough torso area for the English text.')
    canonical = photo.crop(bbox)
    canonical.putalpha(alpha.crop(bbox))
    # Keep all alpha in the canonical layer. Only cut classification ignores
    # near-transparent edge noise, including fringes from raster resampling.
    panel_edges = (0, 0, *PANEL)
    edge_bbox = _edge_evidence_bbox(alpha)
    visible_edges = [edge_bbox is not None and edge_bbox[i] == panel_edges[i] for i in range(4)]
    source_boundaries = (-left, -top, resized[0] - left, resized[1] - top)
    original_cuts = framing.get('geometry', {}).get('source_edge_cuts')
    if original_cuts is None:
        original_bbox = _edge_evidence_bbox(mask)
        original_cuts = [original_bbox is not None and original_bbox[i] == (0, 0, *source.size)[i]
                         for i in range(4)]
    framing['source_cut_edges'] = [visible_edges[i] and original_cuts[i] and
                                   source_boundaries[i] == panel_edges[i] for i in range(4)]
    framing['crop_cut_edges'] = [visible_edges[i] and
                                 (source_boundaries[i] < panel_edges[i] if i < 2 else
                                  source_boundaries[i] > panel_edges[i]) for i in range(4)]
    framing['source_edges'] = [original or cropped for original, cropped in
                               zip(framing['source_cut_edges'], framing['crop_cut_edges'])]
    return photo, alpha, canonical, bbox, framing


def fill_photo_background(config, photo, transform, canonical, work, report):
    """Fill only outside the transformed source rectangle, after layout selection."""
    rect = transform['photo_rect']
    coverage = Image.new('L', PANEL)
    coverage.paste(255, tuple(rect))
    missing = PANEL[0] * PANEL[1] - (rect[2] - rect[0]) * (rect[3] - rect[1])
    metadata = {'required': bool(missing), 'filled_pixels': missing,
                'coverage_rect': list(rect), 'generated_foreground_used': False,
                'source_pixels_unchanged': True, 'canonical_unchanged': True}
    report['background'] = metadata
    if not missing:
        metadata.update({'method': 'not-required', 'image_generation_used': False})
        report['image_generation_used'] = False
        return photo
    if 'prefer_side' in config.get('background', {}):
        metadata['requested_prefer_side'] = config['background']['prefer_side']
    import background_fill
    coverage.save(work / 'background-coverage.png')
    canonical_bytes = canonical.tobytes()
    original = photo.convert('RGB')
    try:
        filled, provider_metadata = background_fill.fill_missing(original.copy(), coverage.copy(), work, config.get('background'))
    except ValueError as exc:
        metadata.update({'status': 'needs_background' if getattr(exc, 'code', None) == 'NEED_BACKGROUND' else 'failed',
                         **getattr(exc, 'details', {})})
        if metadata.get('request_image'):
            metadata['reference_image'] = {'type': 'image', 'path': str(metadata['request_image']),
                                           'relative_path': artifact_relative_path(metadata['request_image']),
                                           'mime_type': 'image/png', 'width': PANEL[0], 'height': PANEL[1]}
        raise RenderError(getattr(exc, 'code', 'BACKGROUND_FAILED'), getattr(exc, 'message', str(exc)),
                          getattr(exc, 'remedy', 'Check the background provider or supply background.image.'),
                          getattr(exc, 'details', {})) from exc
    if not isinstance(filled, Image.Image) or filled.size != PANEL or filled.mode != 'RGB':
        raise RenderError('BACKGROUND_OUTPUT_INVALID', 'The background provider must return an RGB image at the exact panel size.')
    result = Image.composite(original, filled, coverage).convert('RGBA')
    if canonical.tobytes() != canonical_bytes:
        raise RenderError('CANONICAL_CHANGED', 'The canonical subject changed during background filling.')
    metadata.update(provider_metadata)
    metadata.update({'coverage_rect': list(rect), 'generated_foreground_used': False,
                     'source_pixels_unchanged': True, 'canonical_unchanged': True})
    report['image_generation_used'] = bool(metadata.get('image_generation_used'))
    return result


def contour_points(alpha):
    # The exact outer boundary of the canonical alpha includes resampling fringes.
    # The connector gate protects those same pixels, so snap to that same boundary.
    binary = alpha.point(lambda p: 255 if p else 0)
    edge = ImageChops.subtract(binary, binary.filter(ImageFilter.MinFilter(3)))
    pixels = edge.load()
    box = edge.getbbox()
    return [(x, y) for y in range(box[1], box[3]) for x in range(box[0], box[2]) if pixels[x, y]]


def choose_anchor(alpha, bbox, supplied=None):
    points = contour_points(alpha)
    if supplied is not None:
        chosen = min(points, key=lambda p: (p[0] - supplied[0]) ** 2 + (p[1] - supplied[1]) ** 2)
        distance = math.dist(chosen, supplied)
        if distance > 60:
            raise RenderError('ANCHOR_TOO_FAR', f'Anchor is {distance:.1f} panel pixels from the silhouette contour.', 'Place anchor on the intended visible contour tip in the source image.')
        return chosen, {'method': 'provided-point-snapped-to-contour', 'snap_distance': round(distance, 3)}
    h = bbox[3] - bbox[1]
    lower = [p for p in points if bbox[1] + h * .50 <= p[1] <= bbox[1] + h * .94 and 18 <= p[0] < 1182]
    if not lower:
        raise RenderError('ANCHOR_UNAVAILABLE', 'No lower contour point has enough canvas clearance.', 'Supply an anchor on a clear visible contour point.')
    # Prefer the side with more open space. Pick its contour extreme, then the lowest
    # near-extreme point, which is a deterministic outline tip, not a finger detector.
    side = 'left' if bbox[0] >= PANEL[0] - bbox[2] else 'right'
    extreme = min(p[0] for p in lower) if side == 'left' else max(p[0] for p in lower)
    near = [p for p in lower if abs(p[0] - extreme) <= 2]
    chosen = max(near, key=lambda p: p[1])
    return chosen, {'method': 'automatic-lower-contour-extreme', 'side': side, 'semantic_identity': 'unclassified contour tip; visually inspect'}


def anchor_candidates(alpha, bbox, supplied=None):
    """Keep explicit points fixed; try a small set of actual contour tips otherwise."""
    primary = choose_anchor(alpha, bbox, supplied)
    if supplied is not None:
        return [primary]
    choices, seen = [primary], {primary[0]}
    points = contour_points(alpha)
    height = bbox[3] - bbox[1]
    sides = [primary[1]['side'], 'right' if primary[1]['side'] == 'left' else 'left']
    # Keep successful lower-tip choices stable. A later upper-body band covers
    # real contour tips in short crops without inventing a connector path.
    bands = [(side, fraction) for side in sides for fraction in (.55, .72, .89)]
    bands.extend((side, .40) for side in sides)
    for side, fraction in bands:
        band = [p for p in points if bbox[1] + height * (fraction - .07) <= p[1] <= bbox[1] + height * (fraction + .07)
                and 18 <= p[0] < 1182]
        if not band:
            continue
        extreme = min(p[0] for p in band) if side == 'left' else max(p[0] for p in band)
        point = max((p for p in band if abs(p[0] - extreme) <= 2), key=lambda p: p[1])
        if point not in seen:
            seen.add(point)
            choices.append((point, {'method': 'automatic-contour-band-extreme', 'side': side,
                                     'height_fraction': fraction, 'semantic_identity': 'unclassified contour tip; visually inspect'}))
    return choices


def fallback_anchor_candidates(alpha, bbox):
    """Return at most four real low contour tips, excluding panel cut edges."""
    if alpha.getbbox() is None:
        return []
    points = contour_points(alpha)
    height = bbox[3] - bbox[1]
    choices, seen = [], set()
    for low, high in ((.96, .975), (.975, .99)):
        band = [p for p in points if bbox[1] + height * low <= p[1] < bbox[1] + height * high
                and 18 <= p[0] < 1182 and 2 <= p[1] < 798]
        if not band:
            continue
        for side in ('left', 'right'):
            extreme = min(p[0] for p in band) if side == 'left' else max(p[0] for p in band)
            point = max((p for p in band if abs(p[0] - extreme) <= 2), key=lambda p: p[1])
            if point not in seen:
                seen.add(point)
                choices.append((point, {'method': 'automatic-low-contour-band-extreme', 'side': side,
                                         'height_range': [low, high],
                                         'semantic_identity': 'unclassified contour tip; visually inspect'}))
    return choices


@lru_cache(maxsize=256)
def handtext(text, size, seed, angle=0, accent=ACCENT):
    rng = random.Random(f'{seed}:{text}:{size}')
    font = ImageFont.truetype(str(FONT), size)
    advances = [font.getlength(c) + size * .10 for c in text]
    result = Image.new('RGBA', (max(1, int(sum(advances) + size * 3)), size * 4))
    x = size
    for char, step in zip(text, advances):
        cell = Image.new('RGBA', (size * 3, size * 3))
        ImageDraw.Draw(cell).text((size, size), char, font=font, fill=accent, anchor='mm')
        cell = cell.rotate(rng.uniform(-2, 2), resample=Image.Resampling.BICUBIC)
        result.alpha_composite(cell, (int(x - size), int(size * .35 + rng.uniform(-1.3, 1.3))))
        x += step
    bounds = result.getbbox()
    if bounds is None:
        raise RenderError('EMPTY_TEXT', 'A text fragment contains no visible glyphs.')
    return result.crop(bounds).rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)


def english_layout(phrases, alpha, bbox, seed, accent=ACCENT):
    interior = alpha.filter(ImageFilter.MinFilter(17))
    bh = bbox[3] - bbox[1]
    def phrase_image(text, size, wrapped):
        words = text.split()
        rows = [text]
        if wrapped and len(words) > 1:
            splits = [(' '.join(words[:i]), ' '.join(words[i:])) for i in range(1, len(words))]
            rows = list(min(splits, key=lambda pair: max(handtext(row,size,seed,accent=accent).width for row in pair)))
        cells = [handtext(row,size,seed,accent=accent) for row in rows]
        if len(cells) == 1:
            return cells[0], rows
        gap = max(6, round(size * .18))
        image = Image.new('RGBA',(max(cell.width for cell in cells),sum(cell.height for cell in cells)+gap))
        y = 0
        for cell in cells:
            image.alpha_composite(cell,((image.width-cell.width)//2,y))
            y += cell.height + gap
        return image, rows

    # Try readable lettering, then natural word wrapping, before a smaller
    # single-line fallback. The words and their ordering are never rewritten.
    modes = [(size, False) for size in (48, 44, 40, 36)]
    modes += [(size, True) for size in (48, 44, 40, 36)]
    modes += [(size, False) for size in (32, 28, 24)]
    for size, wrapped in modes:
        blocks = [phrase_image(text,size,wrapped) for text in phrases]
        spacing = max(size * 2.5, bh * .10)
        fractions = (.30, .38, .24, .44) if wrapped else (.44, .50, .38, .56)
        for fraction in fractions:
            first_y = int(bbox[1] + bh * fraction)
            placements = []
            next_y = first_y
            for index, (phrase, (im, rows)) in enumerate(zip(phrases, blocks)):
                y = round(next_y if wrapped else first_y + spacing * index)
                if y + im.height > min(bbox[3] - 20, 780):
                    break
                choices = []
                for x in range(max(18, bbox[0]), min(bbox[2] - im.width, 1182 - im.width) + 1, 6):
                    region = interior.crop((x, y, x + im.width, y + im.height))
                    if ImageChops.subtract(im.getchannel('A'), region).getextrema()[1] <= 20:
                        choices.append(x)
                if not choices:
                    break
                x = min(choices, key=lambda value: abs(value + im.width / 2 - (bbox[0] + bbox[2]) / 2))
                placements.append((im, (x, y), {'text': phrase, 'rendered_lines': rows, 'position': [x, y], 'font_size': size, 'inside_silhouette': True}))
                next_y = y + im.height + max(16, round(size * .7))
            if len(placements) == len(phrases):
                return placements
    raise RenderError('ENGLISH_DOES_NOT_FIT', 'English fragments do not fit inside the silhouette at the supported 24–48 px sizes.', 'Shorten the phrases in JSON while keeping their meaning. Do not change the photo or protections.')


def japanese_rows(text):
    """Keep explicit breaks; wrap a long unbroken phrase without changing text."""
    rows = [line for line in text.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) <= 5:
        return rows
    value = rows[0]
    count = 2 if len(value) <= 8 else 3
    rows = []
    for remaining in range(count, 0, -1):
        length = math.ceil(len(value) / remaining)
        rows.append(value[:length])
        value = value[length:]
    return rows


@lru_cache(maxsize=96)
def japanese_block(text, size, seed, angle, accent=ACCENT):
    rows = [handtext(line, size, seed, accent=accent) for line in japanese_rows(text)]
    width, height = max(im.width for im in rows), sum(im.height for im in rows) + 20 * (len(rows) - 1)
    result = Image.new('RGBA', (width, height))
    y = 0
    for im in rows:
        result.alpha_composite(im, ((width - im.width) // 2, y))
        y += im.height + 20
    return result.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)


def japanese_layout(texts, size, lower_pos, canonical, seed, variant=0, accent=ACCENT, avoid=None, allow_same_side=False):
    lx, ly = lower_pos
    margin, clearance = 40, 32
    fractions = ((.54, .76), (.68, .48), (.42, .66))[variant]
    body = Image.new('L', CANVAS)
    body.paste(canonical.getchannel('A'), lower_pos)
    center = lx + canonical.width / 2

    def place_on_sides(sides):
        placements = []
        for index, (text, side) in enumerate(zip(texts, sides)):
            is_left = side == 'left'
            angle = (3 if is_left else -3) * (-1 if allow_same_side and index else 1)
            im = japanese_block(text, size, seed, angle, accent)
            cy = ly + canonical.height * fractions[index]
            preferred_y = max(835, min(1560 - im.height, round(cy - im.height / 2)))
            # Keep both modes beside the original contour. The fallback adds
            # only same-side placements; it never changes subject protection.
            ys = [preferred_y, round(ly + canonical.height * .12)]
            ys += [preferred_y + offset for offset in (-80, 80, -160, 160)]
            if avoid is not None or allow_same_side:
                ys += sorted(range(max(835, ly + 16), min(1560, ly + canonical.height) - im.height + 1, 24),
                             key=lambda y: abs(y - preferred_y))
            found = None
            for y in dict.fromkeys(ys):
                if y < max(835, ly) or y + im.height > min(1560, ly + canonical.height):
                    continue
                occupied = body.crop((0, y - 16, CANVAS[0], y + im.height + 16)).getbbox()
                if is_left:
                    low = margin
                    high = occupied[0] - clearance if occupied else center + im.width / 2 - 24
                else:
                    low = occupied[2] + clearance if occupied else center - im.width / 2 + 24
                    high = CANVAS[0] - margin
                if high - low < im.width:
                    continue
                near = round(high - im.width if is_left else low)
                xs = [near] if avoid is None else [near + (-1 if is_left else 1) * gap for gap in (0, 32, 64, 96, 128)]
                for x in xs:
                    if x < low or x + im.width > high:
                        continue
                    if avoid is not None and avoid.crop((x - 8, y - 8, x + im.width + 9, y + im.height + 9)).getbbox():
                        continue
                    if (is_left and x + im.width / 2 >= center) or (not is_left and x + im.width / 2 <= center):
                        continue
                    if any((record['side'] == side and not (y + im.height + 16 <= py or py + previous.height + 16 <= y)) or
                           not (x + im.width + 16 <= px or px + previous.width + 16 <= x or
                                y + im.height + 16 <= py or py + previous.height + 16 <= y)
                           for previous, (px, py), record in placements):
                        continue
                    found = (x, y)
                    break
                if found is not None:
                    break
            if found is None:
                return None
            x, y = found
            placements.append((im, (x, y), {'text': text, 'rendered_lines': japanese_rows(text), 'position': [x, y], 'font_size': size, 'rotation': angle, 'outside_subject_alpha': True, 'side': side}))
        return placements

    if not allow_same_side:
        return place_on_sides(('left', 'right'))
    # This phase runs only after all normal composition candidates fail.
    # Prefer the side with more room, with the same size and opposite tilts.
    sides = ('left', 'right') if lx >= CANVAS[0] - lx - canonical.width else ('right', 'left')
    for side in sides:
        placements = place_on_sides((side, side))
        if placements is not None:
            return placements
    return None


def protect(alpha, canonical, lower_pos, texts):
    mask = Image.new('L', CANVAS)
    mask.paste(alpha, (0, 0))
    mask.paste(canonical.getchannel('A'), lower_pos)
    draw = ImageDraw.Draw(mask)
    for im, (x, y), _ in texts:
        draw.rectangle((x - 8, y - 8, x + im.width + 8, y + im.height + 8), fill=255)
    return mask


def lower_positions(bbox, canonical, source_edges=None, head_check=None):
    """Prefer the source position; alternatives must improve left/right balance."""
    if source_edges is None:
        evidence = _edge_evidence_bbox(canonical.getchannel('A'))
        edges = [evidence is not None and evidence[i] + bbox[i % 2] == (0, 0, *PANEL)[i]
                 for i in range(4)]
    else:
        edges = source_edges
    # One translated cutout can meet opposite panel edges only when it spans
    # that panel dimension. Never let the second edge override the first.
    if ((edges[0] and edges[2] and canonical.width != PANEL[0]) or
            (edges[1] and edges[3] and canonical.height != PANEL[1])):
        return []
    margin_x = min(24, (CANVAS[0] - canonical.width) // 2)
    margin_y = min(24, (PANEL[1] - canonical.height) // 2)
    x_min, x_max = margin_x, CANVAS[0] - canonical.width - margin_x
    y_min = PANEL[1] if edges[1] else PANEL[1] + margin_y
    y_max = CANVAS[1] - canonical.height - (0 if edges[3] else margin_y)
    if edges[0]:
        x_min = x_max = 0
    if edges[2]:
        x_min = x_max = CANVAS[0] - canonical.width
    if edges[3]:
        # An existing photo-edge cut (for example a waist) must not float above
        # the paper's bottom edge. Complete subjects retain vertical freedom.
        y_min = y_max
    if edges[1]:
        y_max = y_min
    if head_check is not None:
        hx0, hy0, hx1, hy1 = head_check['head_box_panel']
        margins = head_check.get('margins_pixels', [head_check['margin_pixels']] * 4)
        x_min = max(x_min, math.ceil(bbox[0] - hx0 + margins[0] - 1e-8))
        x_max = min(x_max, math.floor(CANVAS[0] + bbox[0] - hx1 - margins[2] + 1e-8))
        y_min = max(y_min, math.ceil(PANEL[1] + bbox[1] - hy0 + margins[1] - 1e-8))
        y_max = min(y_max, math.floor(CANVAS[1] + bbox[1] - hy1 - margins[3] + 1e-8))
    if x_min > x_max or y_min > y_max:
        return []
    def axis(low, high, preferred):
        return list(dict.fromkeys((max(low, min(high, preferred)), round((low + high) / 2),
                                   low, high, round(low * .75 + high * .25))))
    xs = list(dict.fromkeys(max(x_min, min(x_max, x)) for x in
                           (bbox[0], bbox[0] - 120, bbox[0] + 120,
                            round((x_min + x_max) / 2), round(x_min * .75 + x_max * .25))))
    original_x = max(x_min, min(x_max, bbox[0]))
    def side_imbalance(x):
        return abs(x - (CANVAS[0] - x - canonical.width))
    xs = [x for x in xs if x == original_x or side_imbalance(x) < side_imbalance(original_x)]
    ys = axis(y_min, y_max, bbox[1] + PANEL[1])
    positions = [(x, y) for x in xs for y in ys]
    positions.sort(key=lambda p: abs(p[0] - bbox[0]) + abs(p[1] - bbox[1] - PANEL[1]))
    return positions


def choose_layout(config, alpha, canonical, bbox, anchor, english, report, accent=ACCENT, source_edges=None, head_check=None, allow_same_side=False, route_cache=None, extended_budget=False):
    connector.load_python_image_backend()
    try:
        templates = connector.load_templates(TEMPLATES)
    except (OSError, TypeError, KeyError, AttributeError, ValueError) as exc:
        raise RenderError('ASSET_INVALID', f'The connector template library cannot be parsed: {exc}', 'Restore assets/connectors/templates.json from the original package.') from exc
    attempts = []
    routes_without_japanese = {} if route_cache is None else route_cache
    local_routing_count = 0
    routing_limit = min(128 if extended_budget else 96, report.get('routing_attempt_limit', 96))
    # Every crop, anchor and placement shares the same request-wide budget.
    positions = lower_positions(bbox, canonical, source_edges, head_check)
    # Exhaust lettering options at the source position before translating the
    # subject. A safe red line alone must not take priority over composition.
    trials = ((size, lower, variant)
              for lower in positions
              for size in (40, 36, 32, 28)
              for variant in range(3))
    for size, lower, variant in trials:
        end = (anchor[0] + lower[0] - bbox[0], anchor[1] + lower[1] - bbox[1])
        if lower not in routes_without_japanese:
            if allow_same_side:
                # A fallback may use only positions that already have a body
                # route from the normal phase. It cannot expand the search.
                continue
            if local_routing_count >= 96 or report.get('routing_attempt_count', 0) >= routing_limit:
                report.setdefault('layout_attempts', []).extend(attempts)
                raise RenderError('NO_SAFE_LAYOUT', 'The bounded layout search did not find a safe original-template route.',
                                  'Stop this render. Report the constraint; do not modify the source, mask, templates or random seed.')
            routes_without_japanese[lower] = None
            local_routing_count += 1
            report['routing_attempt_count'] = report.get('routing_attempt_count', 0) + 1
            try:
                body_mask = protect(alpha, canonical, lower, english)
                route = connector.choose_safe_candidate(templates, anchor, end, CANVAS, 6.24, body_mask, config['seed'])
                stroke = connector.render_mask(route['points'], CANVAS, 6.24, 6, route['texture_seed'], .58)
                routes_without_japanese[lower] = route, stroke
            except ValueError as exc:
                if not str(exc).startswith('NO_SAFE_CONNECTOR'):
                    raise
                attempts.append({'lower_position': list(lower), 'result': 'no-safe-body-connector'})
        saved_route = routes_without_japanese[lower]
        # Japanese protection only adds obstacles. If no authored route can
        # avoid the complete subjects and English, lettering cannot fix it.
        if saved_route is None:
            continue
        lettering_options = {'allow_same_side': True} if allow_same_side else {}
        japanese = japanese_layout(config['japanese'], size, lower, canonical, config['seed'], variant, accent, **lettering_options)
        if (japanese is not None and local_routing_count < 96 and
                report.get('routing_attempt_count', 0) < routing_limit):
            report['routing_attempt_count'] = report.get('routing_attempt_count', 0) + 1
            local_routing_count += 1
            protected = protect(alpha, canonical, lower, english + japanese)
            try:
                candidate = connector.choose_safe_candidate(templates, anchor, end, CANVAS, 6.24, protected, config['seed'])
                attempts.append({'lower_position': list(lower), 'japanese_font_size': size, 'variant': variant, 'result': 'passed'})
                report.setdefault('layout_attempts', []).extend(attempts)
                report['japanese_layout_mode'] = 'same-side-staggered' if allow_same_side else 'opposite-sides'
                report['fallback_reason'] = 'opposite-side-space-exhausted' if allow_same_side else None
                return lower, japanese, protected, end, candidate
            except ValueError as exc:
                if not str(exc).startswith('NO_SAFE_CONNECTOR'):
                    raise
                attempts.append({'lower_position': list(lower), 'japanese_font_size': size, 'variant': variant, 'result': 'no-safe-connector'})
        # Reuse the prechecked route and move text to avoid its actual pixels.
        # A cached route costs no further routing attempt, even at the limit.
        if saved_route is not None:
            route, stroke = saved_route
            moved_text = japanese_layout(config['japanese'], size, lower, canonical, config['seed'], variant, accent, avoid=stroke, **lettering_options)
            if moved_text is not None:
                final_mask = protect(alpha, canonical, lower, english + moved_text)
                if connector.validate_rendered_mask(stroke, final_mask, anchor, end, route['contact_radius']) == 0:
                    attempts.append({'lower_position': list(lower), 'japanese_font_size': size, 'variant': variant, 'result': 'passed-route-then-text'})
                    report.setdefault('layout_attempts', []).extend(attempts)
                    report['japanese_layout_mode'] = 'same-side-staggered' if allow_same_side else 'opposite-sides'
                    report['fallback_reason'] = 'opposite-side-space-exhausted' if allow_same_side else None
                    return lower, moved_text, final_mask, end, route
    report.setdefault('layout_attempts', []).extend(attempts)
    raise RenderError('NO_SAFE_LAYOUT', 'No bounded layout satisfied subject, lettering, border and connector protection.',
                      'Stop this render. Report the constraint without retries, source padding or mask changes.')


def solve_composition(config, source, mask, report, accent):
    frames = framing_candidates(source, mask, config.get('head_box'), config.get('protected_boxes'),
                                config.get('anchor'), config.get('upper_body_box'))
    prefer_side = config.get('background', {}).get('prefer_side')
    if prefer_side is not None and frames[0]['layout_mode'] == 'background-fill':
        def side_priority(frame):
            left, right = frame['photo_rect'][0], PANEL[0] - frame['photo_rect'][2]
            preferred, opposite = (left, right) if prefer_side == 'left' else (right, left)
            return preferred == 0, opposite
        # A soft preference only reorders existing legal frames; ties retain
        # their authored order and all attempts share the existing route budget.
        frames = sorted(frames, key=side_priority)
    report['framing_attempts'] = []
    report['routing_attempt_limit'] = 96
    last_error = None
    had_route_failure = False
    deferred = []
    for frame in frames:
        supplied = None
        try:
            photo, alpha, canonical, bbox, transform = prepare_subject(source, mask, frame)
            if 'anchor' in config:
                supplied = tuple(min(source.size[i] - 1, config['anchor'][i] * source.size[i]) *
                                 transform['raster_scale'][i] - transform['crop_offset'][i] for i in (0, 1))
            english = english_layout(config['english'], alpha, bbox, config['seed'], accent)
            choices = anchor_candidates(alpha, bbox, supplied)
            # The same protected source region must remain visible below too.
            required = transform['geometry']['required_region_source']
            placement_check = {'head_box_panel': [required[i] * transform['raster_scale'][i % 2] - transform['crop_offset'][i % 2] for i in range(4)], 'margin_pixels': 0}
            for anchor, anchor_record in choices:
                if supplied is not None:
                    anchor_record.update({'requested_source_normalized': config['anchor'],
                                          'requested_panel_point': list(supplied)})
                route_cache = {}
                try:
                    lower, japanese, protected, end, candidate = choose_layout(config, alpha, canonical, bbox, anchor, english, report, accent, transform['source_edges'], placement_check, route_cache=route_cache)
                    report['framing_attempts'].append({'layout_mode': frame['layout_mode'], 'crop_offset': frame['crop_offset'], 'anchor': list(anchor), 'result': 'passed'})
                    return photo, alpha, canonical, bbox, transform, anchor, anchor_record, english, lower, japanese, protected, end, candidate
                except RenderError as exc:
                    if exc.code != 'NO_SAFE_LAYOUT':
                        raise
                    had_route_failure = True
                    last_error = exc
                    if any(route is not None for route in route_cache.values()):
                        deferred.append((photo, alpha, canonical, bbox, transform, anchor, anchor_record,
                                         english, placement_check, frame, route_cache))
                    report.setdefault('anchor_attempts', []).append({'crop_offset': frame['crop_offset'], 'anchor': list(anchor), 'result': exc.code})
                    if report.get('routing_attempt_count', 0) >= report['routing_attempt_limit']:
                        break
            raise last_error
        except RenderError as exc:
            if exc.code not in ('ENGLISH_DOES_NOT_FIT', 'NO_SAFE_LAYOUT', 'ANCHOR_UNAVAILABLE', 'SUBJECT_TOO_SMALL'):
                raise
            last_error = exc
            report['framing_attempts'].append({'layout_mode': frame['layout_mode'], 'crop_offset': frame['crop_offset'], 'result': exc.code})
            if report.get('routing_attempt_count', 0) >= report['routing_attempt_limit']:
                break
    # Exhaust normal left/right compositions globally before considering a
    # same-side exception. Reuse their body-safe routes and the same 96 budget.
    for photo, alpha, canonical, bbox, transform, anchor, anchor_record, english, placement_check, frame, route_cache in deferred:
        try:
            lower, japanese, protected, end, candidate = choose_layout(
                config, alpha, canonical, bbox, anchor, english, report, accent,
                transform['source_edges'], placement_check, allow_same_side=True, route_cache=route_cache)
            report['framing_attempts'].append({'layout_mode': frame['layout_mode'], 'crop_offset': frame['crop_offset'],
                                              'anchor': list(anchor), 'result': 'passed',
                                              'japanese_layout_mode': 'same-side-staggered'})
            return photo, alpha, canonical, bbox, transform, anchor, anchor_record, english, lower, japanese, protected, end, candidate
        except RenderError as exc:
            if exc.code != 'NO_SAFE_LAYOUT':
                raise
            last_error = exc
            report.setdefault('anchor_attempts', []).append({'crop_offset': frame['crop_offset'], 'anchor': list(anchor),
                                                            'result': exc.code, 'japanese_layout_mode': 'same-side-staggered'})
    if 'anchor' not in config and had_route_failure:
        return _solve_low_contour_fallback(config, source, mask, frames, report, accent, last_error)
    raise last_error


def _solve_low_contour_fallback(config, source, mask, frames, report, accent, last_error):
    previous = report.get('routing_attempt_count', 0)
    fallback = {'triggered': True, 'previous_stage_routing_attempts': previous,
                'additional_routing_attempt_limit': 32, 'additional_routing_attempts': 0}
    report['low_contour_fallback'] = fallback
    report['routing_attempt_limit'] = min(128, previous + 32)
    try:
        for frame in frames:
            try:
                photo, alpha, canonical, bbox, transform = prepare_subject(source, mask, frame)
                english = english_layout(config['english'], alpha, bbox, config['seed'], accent)
                required = transform['geometry']['required_region_source']
                placement_check = {'head_box_panel': [required[i] * transform['raster_scale'][i % 2] - transform['crop_offset'][i % 2] for i in range(4)], 'margin_pixels': 0}
                for anchor, anchor_record in fallback_anchor_candidates(alpha, bbox):
                    try:
                        lower, japanese, protected, end, candidate = choose_layout(
                            config, alpha, canonical, bbox, anchor, english, report, accent,
                            transform['source_edges'], placement_check, extended_budget=True)
                        report['framing_attempts'].append({'layout_mode': frame['layout_mode'], 'crop_offset': frame['crop_offset'],
                                                          'anchor': list(anchor), 'result': 'passed', 'stage': 'low-contour-fallback'})
                        return photo, alpha, canonical, bbox, transform, anchor, anchor_record, english, lower, japanese, protected, end, candidate
                    except RenderError as exc:
                        if exc.code != 'NO_SAFE_LAYOUT':
                            raise
                        last_error = exc
                        report.setdefault('anchor_attempts', []).append({'crop_offset': frame['crop_offset'], 'anchor': list(anchor),
                                                                        'result': exc.code, 'stage': 'low-contour-fallback'})
                    if report.get('routing_attempt_count', 0) >= report['routing_attempt_limit']:
                        break
                raise last_error
            except RenderError as exc:
                if exc.code not in ('ENGLISH_DOES_NOT_FIT', 'NO_SAFE_LAYOUT', 'ANCHOR_UNAVAILABLE', 'SUBJECT_TOO_SMALL'):
                    raise
                last_error = exc
                report['framing_attempts'].append({'layout_mode': frame['layout_mode'], 'crop_offset': frame['crop_offset'],
                                                  'result': exc.code, 'stage': 'low-contour-fallback'})
            if report.get('routing_attempt_count', 0) >= report['routing_attempt_limit']:
                break
        raise last_error
    finally:
        fallback['additional_routing_attempts'] = report.get('routing_attempt_count', 0) - previous


def render(config, out, report):
    work, outputs = out / 'work', out / 'outputs'
    work.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    final = outputs / 'final.png'
    # A failed retry must not expose an older result as a successful new result.
    final.unlink(missing_ok=True)
    for asset in (FONT, PAPER, TEMPLATES):
        if not asset.is_file():
            raise RenderError('ASSET_MISSING', str(asset), 'Restore the bundled original asset.')
    report['assets'] = {str(asset.relative_to(ROOT)): sha256(asset) for asset in (FONT, PAPER, TEMPLATES)}
    check_glyphs(config['english'] + config['japanese'])
    report['glyph_coverage'] = 'passed'
    try:
        with Image.open(config['source']) as im:
            source = ImageOps.exif_transpose(im).convert('RGB')
    except OSError as exc:
        raise RenderError('SOURCE_UNREADABLE', str(exc)) from exc
    report['source'] = {'path': str(config['source']), 'sha256': sha256(config['source']), 'oriented_size': list(source.size)}
    mask, segmentation = get_mask(config, source, work)
    report['segmentation'] = segmentation
    mask.save(work / 'source-mask.png')
    palette = derive_palette(source, mask, config.get('theme'))
    for field in ('accent', 'silhouette', 'glow'):
        if field in config:
            palette[field] = config[field]
            palette['metadata'][field + '_source'] = 'provided-reference-color'
    if 'accent' in config:
        palette['metadata']['theme_accent_contrast'] = round(
            palette_contrast(paper.parse_color(palette['theme']), paper.parse_color(palette['accent'])), 3)
    report['palette'] = palette
    photo, alpha, canonical, bbox, transform, anchor, anchor_record, english, lower, japanese, protected, end, candidate = solve_composition(config, source, mask, report, palette['accent'])
    canonical.save(work / 'canonical-subject.png')
    alpha.save(work / 'upper-alpha.png')
    report['transform'] = transform
    if 'head_crop_check' in transform:
        check = transform['head_crop_check']
        lower_box = [check['head_box_panel'][i] + lower[i % 2] - bbox[i % 2] for i in range(4)]
        clearance = [lower_box[0], lower_box[1] - PANEL[1], CANVAS[0] - lower_box[2], CANVAS[1] - lower_box[3]]
        report['head_crop_check'] = {**check, 'upper_head_inside': check['head_inside'],
                                     'lower_head_box': lower_box, 'lower_clearance_pixels': clearance,
                                     'lower_head_inside': all(c >= m - 1e-8 for c, m in zip(clearance, check.get('margins_pixels', [check['margin_pixels']] * 4))),
                                     'annotation_source': 'Agent visual annotation; not semantic head detection'}
        if not (report['head_crop_check']['upper_head_inside'] and report['head_crop_check']['lower_head_inside']):
            raise RenderError('HEAD_CROP_UNSUPPORTED', 'The annotated complete head would be clipped in one of the panels.')
    body = transform['upper_body_check']
    upper_box = body['box_panel']
    lower_box = [upper_box[i] + lower[i % 2] - bbox[i % 2] for i in range(4)]
    rect = transform['photo_rect']
    report['upper_body_check'] = {**body, 'lower_box': lower_box,
                                  'upper_inside': all(upper_box[i] >= rect[i] - 1e-8 if i < 2 else upper_box[i] <= rect[i] + 1e-8 for i in range(4)),
                                  'lower_inside': lower_box[0] >= -1e-8 and lower_box[1] >= PANEL[1] - 1e-8 and lower_box[2] <= CANVAS[0] + 1e-8 and lower_box[3] <= CANVAS[1] + 1e-8,
                                  'semantic_waist_detection': False}
    if not (report['upper_body_check']['upper_inside'] and report['upper_body_check']['lower_inside']):
        raise RenderError('BODY_REGION_CLIPPED', 'The protected visible upper body would be clipped.')
    report['canonical'] = {'bbox': list(bbox), 'sha256': sha256(work / 'canonical-subject.png'), 'alpha_sha256': hashlib.sha256(canonical.getchannel('A').tobytes()).hexdigest(), 'created_once': True}
    report['anchor'] = {**anchor_record, 'upper': list(anchor), 'lower': list(end), 'source_normalized': [(anchor[i] + transform['crop_offset'][i]) / transform['raster_scale'][i] / source.size[i] for i in (0, 1)]}
    # Geometry, lettering and a safe authored connector are fixed before the
    # only possible background request. The canonical RGBA remains untouched.
    photo = fill_photo_background(config, photo, transform, canonical, work, report)
    photo.save(work / 'upper-photo.png')
    theme, color_source = palette['theme'], palette['metadata']['theme_source']
    color = paper.parse_color(theme)
    paper.load_python_image_backend()
    texture = paper.recolor(paper.cover_crop(Image.open(PAPER), *PANEL), color)
    texture.save(work / 'paper.png')
    canvas = Image.new('RGBA', CANVAS)
    canvas.alpha_composite(photo, (0, 0))
    canvas.paste(texture, (0, PANEL[1]))
    glow = Image.new('RGBA', PANEL, palette['glow'])
    glow.putalpha(alpha.filter(ImageFilter.GaussianBlur(10)).point(lambda p: round(p * .4)))
    canvas.alpha_composite(glow, (0, 0))
    cream = Image.new('RGBA', canonical.size, palette['silhouette'])
    cream.putalpha(canonical.getchannel('A'))
    cream.save(work / 'upper-subject.png')
    canonical.save(work / 'lower-subject.png')
    with Image.open(work / 'lower-subject.png') as saved_lower:
        lower_rgb_verified = saved_lower.convert('RGB').tobytes() == canonical.convert('RGB').tobytes()
        lower_rgba_verified = saved_lower.convert('RGBA').tobytes() == canonical.tobytes()
    canvas.alpha_composite(cream, (bbox[0], bbox[1]))
    canvas.alpha_composite(canonical, lower)
    for im, pos, _ in english + japanese:
        canvas.alpha_composite(im, pos)
    protected.save(work / 'connector-protected.png')
    stroke_alpha = connector.render_mask(candidate['points'], CANVAS, 6.24, 6, candidate['texture_seed'], .58)
    collisions = connector.validate_rendered_mask(stroke_alpha, protected, anchor, end, candidate['contact_radius'])
    if collisions:
        raise RenderError('CONNECTOR_COLLISION', f'The complete rendered stroke overlaps {collisions} protected pixels.', 'Use another clear contour anchor or shorter copy.')
    stroke = Image.new('RGBA', CANVAS, palette['accent'])
    stroke.putalpha(stroke_alpha)
    stroke.save(work / 'connector.png')
    canvas.alpha_composite(stroke)
    canvas.convert('RGB').save(final)
    canvas.resize((400, 533), Image.Resampling.LANCZOS).convert('RGB').save(work / 'preview.jpg', quality=94)
    report.update({'status': 'success', 'canvas': list(CANVAS), 'split_y': 800, 'theme': theme, 'theme_source': color_source, 'accent': palette['accent'], 'upper_position': [bbox[0], bbox[1]], 'lower_position': list(lower), 'lower_scale_ratio': 1.0, 'alpha_reused_exactly': cream.getchannel('A').tobytes() == canonical.getchannel('A').tobytes(), 'lower_rgb_matches_canonical': lower_rgb_verified, 'lower_layer_rgba_matches_canonical': lower_rgba_verified, 'text': [record for _, _, record in english + japanese], 'connector': {'template': candidate['template']['id'], 'mirrored': candidate['mirrored'], 'reversed': candidate.get('reversed', False), 'spread': candidate['spread'], 'stroke_width': 6.24, 'contact_radius': candidate['contact_radius'], 'texture_seed': candidate['texture_seed'], 'collision_pixels': collisions, 'protection': 'complete upper and lower subject alpha plus padded text rectangles'}, 'output': str(final), 'output_sha256': sha256(final), 'visual_review_required': ['Chosen subject and props included by segmentation', 'Identity, anatomy and expression of source retained', 'Anchor refers to the intended visible tip; automatic choice is not semantic finger detection', 'English meaning and Japanese fluency', 'Overall editorial balance and stylistic similarity']})
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    config_input = parser.add_mutually_exclusive_group(required=True)
    config_input.add_argument('--config', help='JSON brief path, or - to read JSON from stdin')
    config_input.add_argument('--config-json', help='JSON brief as one argument; relative image paths use the current directory')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'failed', 'renderer': 'redthread-classic-simple', 'image_generation_used': False, 'runtime': {'python': platform.python_version(), 'pillow': PILLOW_VERSION}, 'visual_review': 'not_performed'}
    (out / 'outputs' / 'final.png').unlink(missing_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        config = read_config_json(args.config_json) if args.config_json is not None else read_config(args.config)
        report['seed'] = config['seed']
        brief_path = out / 'work' / 'brief.json'
        brief_path.parent.mkdir(parents=True, exist_ok=True)
        brief_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        report['brief'] = str(brief_path)
        final = render(config, out, report)
    except (RenderError, OSError, ValueError) as exc:
        (out / 'outputs' / 'final.png').unlink(missing_ok=True)
        report['status'] = 'needs_background' if getattr(exc, 'code', None) == 'NEED_BACKGROUND' else 'failed'
        report['error'] = {'code': getattr(exc, 'code', 'RENDER_FAILED'), 'message': getattr(exc, 'message', str(exc)), 'remedy': getattr(exc, 'remedy', 'Check the supplied images and brief.'), 'details': getattr(exc, 'details', {})}
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(str(exc), file=sys.stderr)
        print(f'Report: {out / "report.json"}', file=sys.stderr)
        print(json.dumps({'status': report['status'], 'report': str(out / 'report.json'),
                          'brief': report.get('brief'), 'background': report.get('background'),
                          'error': report['error'], 'visual_review': 'not_performed'}, ensure_ascii=False))
        return 2
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'status': 'success', 'output': str(final), 'report': str(out / 'report.json'),
        'brief': report['brief'],
        'segmentation': report.get('segmentation'),
        'background': report.get('background'),
        'image_generation_used': report['image_generation_used'],
        'composition': {
            'layout_mode': report.get('transform', {}).get('layout_mode'),
            'japanese_layout_mode': report.get('japanese_layout_mode'),
            'fallback_reason': report.get('fallback_reason'),
            'crop_offset': report.get('transform', {}).get('crop_offset'),
            'lower_position': report.get('lower_position'),
            'framing_attempts': len(report.get('framing_attempts', [])),
            'routing_attempts': report.get('routing_attempt_count', 0),
            'routing_attempt_limit': report.get('routing_attempt_limit', 96),
        },
        'checks': {
            'glyph_coverage': report['glyph_coverage'],
            'alpha_reused_exactly': report['alpha_reused_exactly'],
            'lower_layer_rgba_matches_canonical': report['lower_layer_rgba_matches_canonical'],
            'lower_scale_ratio': report['lower_scale_ratio'],
            'connector_collision_pixels': report['connector']['collision_pixels'],
            'head_crop_check': report.get('head_crop_check'),
            'upper_body_check': report.get('upper_body_check'),
        },
        'artifact': {
            'type': 'image', 'path': str(final), 'mime_type': 'image/png',
            'relative_path': artifact_relative_path(final),
            'width': report['canvas'][0], 'height': report['canvas'][1],
            'sha256': report['output_sha256'],
        },
        'visual_review': 'not_performed',
        'visual_review_required': report['visual_review_required'],
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
