#!/usr/bin/env python3
"""Render the classic Redthread style from one JSON brief; no image-generation API."""
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
from palette import derive_palette

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
    unknown = set(data) - {'source', 'mask', 'english', 'japanese', 'theme', 'seed', 'anchor', 'subject_point', 'segmentation', 'head_box', 'upper_body_box', 'protected_boxes'}
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


def framing_candidates(source, mask, head_box=None, protected_boxes=None, anchor=None, upper_body_box=None):
    """Prefer a full photo panel, protecting the visible head-to-waist region."""
    if source.size != mask.size:
        raise RenderError('MASK_SIZE', f'Mask {mask.size} does not match oriented source {source.size}.', 'Create a mask from the EXIF-oriented source, at the same pixel dimensions.')
    if mask.getbbox() is None or mask.getextrema()[1] < 128:
        raise RenderError('EMPTY_MASK', 'The mask does not contain a foreground subject.')
    scale = max(PANEL[0] / source.width, PANEL[1] / source.height)
    resized = (math.ceil(source.width * scale), math.ceil(source.height * scale))
    # Integer image dimensions can differ from the ideal scale by under one
    # pixel. Safety checks must use the transform Pillow actually applies.
    raster_scale = [resized[i] / source.size[i] for i in (0, 1)]
    source_bbox = mask.getbbox()
    touches_bottom = source_bbox[3] >= source.height - 2
    fit_scale = min(PANEL[0] / (source_bbox[2] - source_bbox[0]), PANEL[1] / (source_bbox[3] - source_bbox[1]))
    geometry = {'source_size': list(source.size), 'subject_bbox': list(source_bbox),
                'panel_size': list(PANEL), 'alpha_support': 'all nonzero source alpha', 'minimum_cover_scale': scale,
                'maximum_subject_scale': fit_scale,
                'subject_size_at_cover': [(source_bbox[2] - source_bbox[0]) * scale,
                                          (source_bbox[3] - source_bbox[1]) * scale],
                'background_size_at_subject_fit': [source.width * fit_scale, source.height * fit_scale]}
    def crop_bounds(box):
        return [(max(0, math.ceil(box[i + 2] * raster_scale[i] - PANEL[i] - 1e-8)),
                 min(resized[i] - PANEL[i], math.floor(box[i] * raster_scale[i] + 1e-8))) for i in (0, 1)]

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
        head_margin = max(2, .02 * max(head_pixels[2] - head_pixels[0], head_pixels[3] - head_pixels[1]))
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
        padded = [max(0, values[0] - 2), max(0, values[1] - 2),
                  min(source.width, values[2] + 2), min(source.height, values[3] + 2)]
        required = padded if required is None else union(required, padded)
    if anchor is not None:
        normalized_point(anchor, 'anchor')
        ax, ay = anchor[0] * source.width, anchor[1] * source.height
        point_box = [max(0, ax - 8 / raster_scale[0]), max(0, ay - 8 / raster_scale[1]),
                     min(source.width, ax + 8 / raster_scale[0]), min(source.height, ay + 8 / raster_scale[1])]
        required = point_box if required is None else union(required, point_box)
    geometry.update({'protected_boxes': protected_boxes or [], 'requested_anchor': anchor,
                     'required_region_source': required})
    if required is not None and (required[0] < 0 or required[1] < 0 or required[2] > source.width or
                                 required[3] > source.height or not feasible(crop_bounds(required))):
        raise RenderError('FULL_BLEED_UNAVAILABLE', 'The protected visible body needs the photo-card layout.',
                          'Use the internal photo-card framing; keep all protections.', geometry)
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
    axes = [list(dict.fromkeys((max(low, min(high, wanted)), low, high)))
            for (low, high), wanted in zip(bounds, preferred)]
    offsets = [(x, y) for x in axes[0] for y in axes[1]]
    offsets.sort(key=lambda p: abs(p[0] - preferred[0]) + abs(p[1] - preferred[1]))
    frames = []
    for x, y in offsets:
        frame = {'uniform_scale': scale, 'raster_scale': raster_scale, 'resized': list(resized), 'crop_offset': [x, y],
                 'bottom_aligned': touches_bottom and y == resized[1] - PANEL[1],
                 'geometry': geometry, 'crop_candidates': len(offsets), 'body_crop': body_crop,
                 'layout_mode': 'full-bleed', 'photo_rect': [0, 0, PANEL[0], PANEL[1]],
                 'source_edges': [source_bbox[0] * raster_scale[0] <= x, source_bbox[1] * raster_scale[1] <= y,
                                  source_bbox[2] * raster_scale[0] >= x + PANEL[0], source_bbox[3] * raster_scale[1] >= y + PANEL[1]]}
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


def photo_card_frames(source, mask, config):
    """Fit a protected original-photo crop on the existing paper, at three scales.

    All image coordinates still use source*raster_scale-crop_offset. Paper is
    outside photo_rect; it is never treated as synthesized source pixels.
    """
    try:
        geometry = framing_candidates(source, mask, config.get('head_box'), config.get('protected_boxes'),
                                      config.get('anchor'), config.get('upper_body_box'))[0]['geometry']
    except RenderError as exc:
        if exc.code != 'FULL_BLEED_UNAVAILABLE':
            raise
        geometry = exc.details
    required = geometry['required_region_source']
    rw, rh = required[2] - required[0], required[3] - required[1]
    width = min(source.width, math.ceil(max(rw * 1.16, rh * 1.5)))
    height = min(source.height, math.ceil(max(rh * 1.08, width / 1.5)))
    # Move the crop, never the annotation, when the source is near an edge.
    left = max(0, min(source.width - width, math.floor((required[0] + required[2] - width) / 2)))
    top = max(0, min(source.height - height, math.floor(required[1] - (height - rh) * .30)))
    left = min(left, math.floor(required[0]))
    top = min(top, math.floor(required[1]))
    width = max(width, math.ceil(required[2]) - left)
    height = max(height, math.ceil(required[3]) - top)
    frames = []
    for maximum in ((1120, 752), (980, 700), (840, 640)):
        scale = min(maximum[0] / width, maximum[1] / height)
        resized = [math.ceil(source.width * scale), math.ceil(source.height * scale)]
        scales = [resized[i] / source.size[i] for i in (0, 1)]
        crop = [math.floor(left * scales[0]), math.floor(top * scales[1]),
                min(resized[0], math.ceil((left + width) * scales[0])),
                min(resized[1], math.ceil((top + height) * scales[1]))]
        pw, ph = crop[2] - crop[0], crop[3] - crop[1]
        px, py = (PANEL[0] - pw) // 2, (PANEL[1] - ph) // 2
        offset = [crop[0] - px, crop[1] - py]
        frame = {'uniform_scale': scale, 'raster_scale': scales, 'resized': resized,
                 'crop_offset': offset, 'photo_rect': [px, py, px + pw, py + ph],
                 'layout_mode': 'paper-photo-card', 'card_limit': list(maximum),
                 'source_window': [left, top, left + width, top + height],
                 'bottom_aligned': False, 'geometry': geometry, 'crop_candidates': 3,
                 'body_crop': not (left <= geometry['subject_bbox'][0] and top <= geometry['subject_bbox'][1]
                                  and left + width >= geometry['subject_bbox'][2] and top + height >= geometry['subject_bbox'][3])}
        def mapped(box):
            return [box[i] * scales[i % 2] - offset[i % 2] for i in range(4)]
        body = geometry['upper_body_box_source']
        frame['upper_body_check'] = {'box_source': body, 'box_panel': mapped(body),
                                    'annotation_source': geometry['body_protection_source']}
        if 'head_box_source' in geometry:
            head = geometry['head_box_source']
            box = mapped(head)
            margin = geometry['head_margin_source_pixels']
            margins = [min(margin, head[0]) * scales[0], min(margin, head[1]) * scales[1],
                       min(margin, source.width - head[2]) * scales[0],
                       min(margin, source.height - head[3]) * scales[1]]
            clearance = [box[0] - px, box[1] - py, px + pw - box[2], py + ph - box[3]]
            frame['head_crop_check'] = {'head_box_source': head, 'head_box_panel': box,
                                       'head_inside': all(c >= m - 1e-8 for c, m in zip(clearance, margins)),
                                       'clearance_pixels': clearance, 'margin_pixels': min(margins), 'margins_pixels': margins}
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
    if framing.get('layout_mode') == 'paper-photo-card':
        # Sample only the crop. Enlarging a small subject in a large source
        # must not allocate a multi-gigabyte intermediate full-size image.
        scales = framing['raster_scale']
        source_box = tuple(crop[i] / scales[i % 2] for i in range(4))
        size = (rect[2] - rect[0], rect[3] - rect[1])
        photo.paste(source.convert('RGB').resize(size, Image.Resampling.LANCZOS, box=source_box), tuple(rect[:2]))
        alpha.paste(mask.resize(size, Image.Resampling.LANCZOS, box=source_box), tuple(rect[:2]))
    else:
        photo.paste(source.convert('RGB').resize(resized, Image.Resampling.LANCZOS).crop(crop), tuple(rect[:2]))
        alpha.paste(mask.resize(resized, Image.Resampling.LANCZOS).crop(crop), tuple(rect[:2]))
    bbox = alpha.getbbox()
    if bbox is None or bbox[2] - bbox[0] < 90 or bbox[3] - bbox[1] < 180:
        raise RenderError('SUBJECT_TOO_SMALL', 'The subject is too small after the fixed panel crop.', 'Use a closer portrait with enough torso area for the English text.')
    canonical = photo.crop(bbox)
    canonical.putalpha(alpha.crop(bbox))
    # Any body cut introduced by this crop belongs at the same paper edge below.
    framing['source_edges'] = [bbox[i] == rect[i] for i in range(4)]
    return photo, alpha, canonical, bbox, framing


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
    for side in sides:
        for fraction in (.55, .72, .89):
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
    for size in (32, 28, 24):
        images = [handtext(text, size, seed, accent=accent) for text in phrases]
        spacing = max(size * 2.5, bh * .10)
        for fraction in (.44, .50, .38, .56):
            first_y = int(bbox[1] + bh * fraction)
            placements = []
            for index, (phrase, im) in enumerate(zip(phrases, images)):
                y = round(first_y + spacing * index)
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
                placements.append((im, (x, y), {'text': phrase, 'position': [x, y], 'font_size': size, 'inside_silhouette': True}))
            if len(placements) == len(phrases):
                return placements
    raise RenderError('ENGLISH_DOES_NOT_FIT', 'English fragments do not fit inside the silhouette at the readable 24–32 px sizes.', 'Shorten the phrases in JSON while keeping their meaning. Do not change the photo or protections.')


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
            angle = 3 if is_left else -3
            im = japanese_block(text, size, seed, angle, accent)
            cy = ly + canonical.height * fractions[index]
            preferred_y = max(835, min(1560 - im.height, round(cy - im.height / 2)))
            # Text stays beside the contour. Adaptive layouts may use one side,
            # but cannot put both blocks at the same height or over the subject.
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

    sides_to_try = [('left', 'right')]
    if allow_same_side:
        sides_to_try += [('left', 'left'), ('right', 'right')]
    for sides in sides_to_try:
        placements = place_on_sides(sides)
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
    """Translate one layer; opposing card cuts use the translated paper boundary."""
    edges = source_edges if source_edges is not None else [bbox[0] == 0, bbox[1] == 0, bbox[2] == PANEL[0], bbox[3] == PANEL[1]]
    # In a full-bleed crop, opposite cuts imply a full panel-sized subject.
    # A smaller photo card can have both cuts too; render gives it a visible
    # paper rectangle, so neither cut must override the other canvas edge.
    paired_x = edges[0] and edges[2] and canonical.width < PANEL[0]
    paired_y = edges[1] and edges[3] and canonical.height < PANEL[1]
    margin_x = min(24, (CANVAS[0] - canonical.width) // 2)
    margin_y = min(24, (PANEL[1] - canonical.height) // 2)
    x_min, x_max = margin_x, CANVAS[0] - canonical.width - margin_x
    y_min = PANEL[1] if edges[1] and not paired_y else PANEL[1] + margin_y
    y_max = CANVAS[1] - canonical.height - (0 if edges[3] and not paired_y else margin_y)
    if edges[0] and not paired_x:
        x_min = x_max = 0
    if edges[2] and not paired_x:
        x_min = x_max = CANVAS[0] - canonical.width
    if edges[3] and not paired_y:
        # An existing photo-edge cut (for example a waist) must not float above
        # the paper's bottom edge. Complete subjects retain vertical freedom.
        y_min = y_max
    if edges[1] and not paired_y:
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
    ys = axis(y_min, y_max, bbox[1] + PANEL[1])
    positions = [(x, y) for x in xs for y in ys]
    positions.sort(key=lambda p: abs(p[0] - bbox[0]) + abs(p[1] - bbox[1] - PANEL[1]))
    return positions


def choose_layout(config, alpha, canonical, bbox, anchor, english, report, accent=ACCENT, source_edges=None, head_check=None):
    connector.load_python_image_backend()
    try:
        templates = connector.load_templates(TEMPLATES)
    except (OSError, TypeError, KeyError, AttributeError, ValueError) as exc:
        raise RenderError('ASSET_INVALID', f'The connector template library cannot be parsed: {exc}', 'Restore assets/connectors/templates.json from the original package.') from exc
    attempts = []
    routes_without_japanese = {}
    local_routing_count = 0
    lettering_options = {'allow_same_side': True} if config.get('_adaptive_layout') else {}
    routing_limit = min(288, report.get('routing_attempt_limit', 288))
    # Each crop/anchor gets at most 96 routing attempts. All candidates share
    # the request counter and the active layout stage's upper limit.
    positions = lower_positions(bbox, canonical, source_edges, head_check)
    # Search nearby placements first, but give every font size a turn before
    # spending the routing budget on distant positions.
    trials = ((size, lower, variant)
              for start in range(0, len(positions), 4)
              for size in (40, 36, 32, 28)
              for lower in positions[start:start + 4]
              for variant in range(3))
    for size, lower, variant in trials:
        japanese = japanese_layout(config['japanese'], size, lower, canonical, config['seed'], variant, accent, **lettering_options)
        if local_routing_count >= 96 or report.get('routing_attempt_count', 0) >= routing_limit:
            report.setdefault('layout_attempts', []).extend(attempts)
            raise RenderError('NO_SAFE_LAYOUT', 'The bounded layout search did not find a safe original-template route.',
                              'Stop this render. Report the constraint; do not modify the source, mask, templates or random seed.')
        end = (anchor[0] + lower[0] - bbox[0], anchor[1] + lower[1] - bbox[1])
        if japanese is not None:
            report['routing_attempt_count'] = report.get('routing_attempt_count', 0) + 1
            local_routing_count += 1
            protected = protect(alpha, canonical, lower, english + japanese)
            try:
                candidate = connector.choose_safe_candidate(templates, anchor, end, CANVAS, 6.24, protected, config['seed'])
                attempts.append({'lower_position': list(lower), 'japanese_font_size': size, 'variant': variant, 'result': 'passed'})
                report.setdefault('layout_attempts', []).extend(attempts)
                return lower, japanese, protected, end, candidate
            except ValueError as exc:
                if not str(exc).startswith('NO_SAFE_CONNECTOR'):
                    raise
                attempts.append({'lower_position': list(lower), 'japanese_font_size': size, 'variant': variant, 'result': 'no-safe-connector'})
        # Try the inverse order too: pick a body-safe authored route,
        # then move text along the contour to avoid its actual pixels.
        # This is an internal joint search, not another Agent command.
        if lower not in routes_without_japanese:
            routes_without_japanese[lower] = None
            if local_routing_count < 96 and report.get('routing_attempt_count', 0) < routing_limit:
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
        saved_route = routes_without_japanese[lower]
        if saved_route is not None:
            route, stroke = saved_route
            moved_text = japanese_layout(config['japanese'], size, lower, canonical, config['seed'], variant, accent, avoid=stroke, **lettering_options)
            if moved_text is not None:
                final_mask = protect(alpha, canonical, lower, english + moved_text)
                if connector.validate_rendered_mask(stroke, final_mask, anchor, end, route['contact_radius']) == 0:
                    attempts.append({'lower_position': list(lower), 'japanese_font_size': size, 'variant': variant, 'result': 'passed-route-then-text'})
                    report.setdefault('layout_attempts', []).extend(attempts)
                    return lower, moved_text, final_mask, end, route
    report.setdefault('layout_attempts', []).extend(attempts)
    raise RenderError('NO_SAFE_LAYOUT', 'No bounded layout satisfied subject, lettering, border and connector protection.',
                      'Stop this render. Report the constraint without retries, source padding or mask changes.')


def solve_composition(config, source, mask, report, accent):
    try:
        full_frames = framing_candidates(source, mask, config.get('head_box'), config.get('protected_boxes'),
                                         config.get('anchor'), config.get('upper_body_box'))
    except RenderError as exc:
        if exc.code != 'FULL_BLEED_UNAVAILABLE':
            raise
        full_frames = []
        report['adaptation_reason'] = exc.message
    cards = photo_card_frames(source, mask, config)
    stages = [(full_frames, 96)] + [([card], limit) for card, limit in zip(cards, (160, 224, 288))]
    report['framing_attempts'] = []
    last_error = None
    for frames, budget in stages:
        report['routing_attempt_limit'] = budget
        for frame in frames:
            supplied = None
            try:
                photo, alpha, canonical, bbox, transform = prepare_subject(source, mask, frame)
                if 'anchor' in config:
                    supplied = tuple(min(source.size[i] - 1, config['anchor'][i] * source.size[i]) *
                                     transform['raster_scale'][i] - transform['crop_offset'][i] for i in (0, 1))
                english = english_layout(config['english'], alpha, bbox, config['seed'], accent)
                choices = anchor_candidates(alpha, bbox, supplied)
                # Protect the whole requested region in the lower panel too,
                # including transparent parts of annotations outside alpha.
                required = transform['geometry']['required_region_source']
                placement_check = {'head_box_panel': [required[i] * transform['raster_scale'][i % 2] - transform['crop_offset'][i % 2] for i in range(4)], 'margin_pixels': 0}
                local_config = {**config, '_adaptive_layout': frame['layout_mode'] == 'paper-photo-card'}
                for anchor, anchor_record in choices:
                    if supplied is not None:
                        anchor_record.update({'requested_source_normalized': config['anchor'],
                                              'requested_panel_point': list(supplied)})
                    try:
                        lower, japanese, protected, end, candidate = choose_layout(local_config, alpha, canonical, bbox, anchor, english, report, accent, transform['source_edges'], placement_check)
                        report['framing_attempts'].append({'layout_mode': frame['layout_mode'], 'crop_offset': frame['crop_offset'], 'anchor': list(anchor), 'result': 'passed'})
                        return photo, alpha, canonical, bbox, transform, anchor, anchor_record, english, lower, japanese, protected, end, candidate
                    except RenderError as exc:
                        if exc.code != 'NO_SAFE_LAYOUT':
                            raise
                        last_error = exc
                        report.setdefault('anchor_attempts', []).append({'crop_offset': frame['crop_offset'], 'anchor': list(anchor), 'result': exc.code})
                        if report.get('routing_attempt_count', 0) >= budget:
                            break
                raise last_error
            except RenderError as exc:
                if exc.code not in ('ENGLISH_DOES_NOT_FIT', 'NO_SAFE_LAYOUT', 'ANCHOR_UNAVAILABLE', 'SUBJECT_TOO_SMALL'):
                    raise
                last_error = exc
                report['framing_attempts'].append({'layout_mode': frame['layout_mode'], 'crop_offset': frame['crop_offset'], 'result': exc.code})
                if report.get('routing_attempt_count', 0) >= budget:
                    break
    raise last_error


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
    report['palette'] = palette
    photo, alpha, canonical, bbox, transform, anchor, anchor_record, english, lower, japanese, protected, end, candidate = solve_composition(config, source, mask, report, palette['accent'])
    photo.save(work / 'upper-photo.png')
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
    theme, color_source = palette['theme'], palette['metadata']['theme_source']
    color = paper.parse_color(theme)
    paper.load_python_image_backend()
    texture = paper.recolor(paper.cover_crop(Image.open(PAPER), *PANEL), color)
    texture.save(work / 'paper.png')
    canvas = Image.new('RGBA', CANVAS)
    if transform['layout_mode'] == 'paper-photo-card':
        canvas.paste(texture, (0, 0))
    canvas.alpha_composite(photo, (0, 0))
    canvas.paste(texture, (0, PANEL[1]))
    edges = transform['source_edges']
    paired_cuts = {'horizontal': edges[0] and edges[2] and canonical.width < PANEL[0],
                   'vertical': edges[1] and edges[3] and canonical.height < PANEL[1]}
    if transform['layout_mode'] == 'paper-photo-card' and any(paired_cuts.values()):
        # The original photo's opposing cuts cannot both meet the larger
        # canvas edges. A visible piece of the same paper receives those cuts
        # at exactly the translated photo boundary; the subject stays intact.
        card = [transform['photo_rect'][i] + lower[i % 2] - bbox[i % 2] for i in range(4)]
        visible = [max(0, card[0]), max(PANEL[1], card[1]),
                   min(CANVAS[0], card[2]), min(CANVAS[1], card[3])]
        local = [visible[0], visible[1] - PANEL[1], visible[2], visible[3] - PANEL[1]]
        card_mask = Image.new('L', PANEL)
        ImageDraw.Draw(card_mask).rectangle((local[0], local[1], local[2] - 1, local[3] - 1), fill=255)
        shadow_mask = Image.new('L', PANEL)
        ImageDraw.Draw(shadow_mask).rectangle((local[0] + 3, local[1] + 5, local[2] + 2, local[3] + 4), fill=36)
        shadow = Image.new('RGBA', PANEL, (0, 0, 0, 0))
        shadow.putalpha(shadow_mask.filter(ImageFilter.GaussianBlur(5)))
        canvas.alpha_composite(shadow, (0, PANEL[1]))
        card_paper = Image.blend(texture.convert('RGB'), Image.new('RGB', PANEL, 'white'), .055).convert('RGBA')
        card_paper.putalpha(card_mask)
        canvas.alpha_composite(card_paper, (0, PANEL[1]))
        subject_rect = [lower[0], lower[1], lower[0] + canonical.width, lower[1] + canonical.height]
        report['lower_paper_card'] = {
            'translated_photo_rect': card, 'visible_rect': visible,
            'paired_source_edges': paired_cuts,
            'source_cuts_aligned': all(not edge or subject_rect[i] == visible[i] for i, edge in enumerate(edges)),
            'paper_asset': str(PAPER.relative_to(ROOT)), 'white_blend': .055,
            'subject_scale_ratio': 1.0,
        }
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
        report['error'] = {'code': getattr(exc, 'code', 'RENDER_FAILED'), 'message': getattr(exc, 'message', str(exc)), 'remedy': getattr(exc, 'remedy', 'Check the supplied images and brief.'), 'details': getattr(exc, 'details', {})}
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(str(exc), file=sys.stderr)
        print(f'Report: {out / "report.json"}', file=sys.stderr)
        print(json.dumps({'status': 'failed', 'report': str(out / 'report.json'), 'error': report['error'], 'visual_review': 'not_performed'}, ensure_ascii=False))
        return 2
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'status': 'success', 'output': str(final), 'report': str(out / 'report.json'),
        'brief': report['brief'],
        'segmentation': report.get('segmentation'),
        'composition': {
            'layout_mode': report.get('transform', {}).get('layout_mode'),
            'crop_offset': report.get('transform', {}).get('crop_offset'),
            'lower_position': report.get('lower_position'),
            'framing_attempts': len(report.get('framing_attempts', [])),
            'routing_attempts': report.get('routing_attempt_count', 0),
            'routing_attempt_limit': 288,
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
