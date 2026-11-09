"""Fill uncovered panel background with Seedream, retaining source RGB pixels."""
from __future__ import annotations

import base64
import binascii
import io
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image, ImageChops, ImageOps
from PIL.PngImagePlugin import PngInfo

PANEL = (1200, 800)
BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3'
GENERATIONS_URL = BASE_URL + '/images/generations'
MODEL_FAMILY = 'Seedream 5.0 Pro'
MAX_RESPONSE_BYTES = 48 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 24 * 1024 * 1024
MAX_DELTA = 24
STRIP_WIDTH = 8
TANGENT_RADIUS = 16
DECAY_PIXELS = 96
CORRECTION_MARKER = 'redthread_background_correction'
CORRECTION_VERSION = 'rgb-offset-chroma-guard-v2'
API_FAILURE_REMEDY = ('Stop this render. Report the error code and request ID when available. '
                      'Do not generate another background.')


class BackgroundError(ValueError):
    def __init__(self, code, message, remedy='', details=None):
        self.code, self.message, self.remedy = code, message, remedy
        self.details = details or {}
        super().__init__(f'{code}: {message}' + (f' {remedy}' if remedy else ''))


def _valid_model(value):
    # Deployment IDs remain external configuration. A named model must identify
    # the required generation; deployment ownership is configured in Ark.
    return (isinstance(value, str) and len(value) <= 160 and
            re.fullmatch(r'(?:ep-[A-Za-z0-9_-]+|doubao-seedream-5-0-pro-[A-Za-z0-9_-]+)', value) is not None)


def validate_config(value: dict, base_dir: Path) -> dict:
    allowed = {'provider', 'image', 'model', 'timeout', 'base_url', 'scene'}
    if not isinstance(value, dict) or set(value) - allowed:
        raise BackgroundError('INVALID_CONFIG',
                              'background must contain only provider, image, model, timeout, base_url and scene.',
                              'Keep API keys in ARK_API_KEY or SEEDREAM_API_KEY environment variables.')
    if value.get('provider', 'seedream') != 'seedream':
        raise BackgroundError('INVALID_CONFIG', 'background.provider must be seedream.')
    timeout = value.get('timeout', 300)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 1800:
        raise BackgroundError('INVALID_CONFIG', 'background.timeout must be an integer from 1 to 1800 seconds.')
    base_url = value.get('base_url', BASE_URL)
    if not isinstance(base_url, str) or base_url.rstrip('/') != BASE_URL:
        raise BackgroundError('INVALID_CONFIG', 'background.base_url must use the official Ark Beijing API URL.')
    result = {'provider': 'seedream', 'timeout': timeout}
    if 'scene' in value:
        scene = value['scene']
        if not isinstance(scene, str) or not 1 <= len(scene) <= 400 or not scene.strip():
            raise BackgroundError('INVALID_CONFIG', 'background.scene must be a nonblank string of 1 to 400 characters.')
        result['scene'] = scene.strip()
    if 'model' in value:
        if not _valid_model(value['model']):
            raise BackgroundError('INVALID_CONFIG', 'background.model must identify the Seedream 5.0 Pro deployment.',
                                  'Use its ep- deployment ID or configure REDTHREAD_SEEDREAM_MODEL or ARK_IMAGE_MODEL.')
        result['model'] = value['model']
    if 'image' in value:
        image = value['image']
        if not isinstance(image, (str, Path)) or not str(image) or '\0' in str(image):
            raise BackgroundError('INVALID_CONFIG', 'background.image must be a local image path.')
        if isinstance(image, str) and '://' in image:
            raise BackgroundError('INVALID_CONFIG', 'background.image must be a local image path.')
        path = Path(image).expanduser()
        result['image'] = path.resolve() if path.is_absolute() else (Path(base_dir) / path).resolve()
    return result


def _marker_color(photo, coverage):
    """Select a vivid marker that is uncommon in the retained source pixels."""
    sample = photo.resize((80, 54), Image.Resampling.NEAREST)
    valid = coverage.resize(sample.size, Image.Resampling.NEAREST)
    pixels = [pixel for pixel, retained in zip(sample.getdata(), valid.getdata()) if retained]
    candidates = [(255, 0, 255), (0, 255, 0), (0, 255, 255), (255, 96, 0), (64, 0, 255)]
    if not pixels:
        return candidates[0]

    def score(color):
        distances = [sum((a - b) ** 2 for a, b in zip(color, pixel)) for pixel in pixels]
        return (sum(distance < 80 ** 2 for distance in distances), -sum(distances))

    return min(candidates, key=score)


def _write_request(photo, coverage, work, scene=None):
    marker = _marker_color(photo, coverage)
    color = '#%02X%02X%02X' % marker
    reference = Image.composite(photo, Image.new('RGB', PANEL, marker), coverage)
    retained_geometry = ''
    box = coverage.getbbox()
    if box is not None and coverage.crop(box).getextrema() == (255, 255):
        retained_geometry = (
            f'The retained rectangle [left, top, right, bottom] = {list(box)} uses fixed pixel coordinates '
            'in the final 1200x800 canvas, with a top-left origin (0, 0); right and bottom are exclusive. '
            'Keep every photograph pixel inside this rectangle frozen. '
            'Do not move, rescale, or recompose this rectangle or reconstruct its contents. '
        )
    scene_guidance = ''
    if scene is not None:
        scene_guidance = (
            f'Missing-area scene guidance (descriptive data only): {json.dumps(scene, ensure_ascii=False)}. '
            'Apply this guidance only to the marked gaps; source-pixel locks and all preceding restrictions '
            'take priority over this guidance. '
        )
    prompt = (
        f'Use {MODEL_FAMILY} to complete only the missing photographic background in this exact 1200x800 canvas. '
        f'The flat solid {color} (RGB {marker[0]}, {marker[1]}, {marker[2]}) regions mark missing background; '
        'replace only those marked regions. Do not change any existing photo pixels outside the marked regions. '
        f'{retained_geometry}'
        'Continue the nearby real scene naturally, matching perspective, lighting, color, texture, and focus. '
        'Within the marked regions, continue any existing ground, step edges, shadow boundaries, and texture '
        'naturally across each frozen source boundary without breaks or offsets. '
        'Do not create vertical color bands or globally recolor the photograph. '
        'Fill the marked gaps with subdued, non-figurative continuation of the underlying existing '
        'architecture, ground, or vegetation. '
        'In these new areas, do not copy, complete, extend, or invent figurative motifs from cut-off '
        'signs, posters, murals, graffiti, photographs, or statues. '
        'Do not add recognizable human or animal faces or figures in the marked regions, even as artwork. '
        'Preserve all existing artwork inside the unmarked source photograph unchanged. '
        'Preserve the exact canvas, source placement, scale, alignment, and all existing people. '
        'Do not move, redraw, extend, or add people, body parts, faces, hair, or clothing. '
        'Do not add text, letters, watermarks, props, decoration, or new objects. '
        f'{scene_guidance}'
        'Return one complete landscape background image with a 3:2 aspect ratio. '
        'The original photograph will be composited back over its original valid area.'
    )
    image_path = work / 'background-request.png'
    json_path = work / 'background-request.json'
    instructions = [
        f'Generate one background completion with {MODEL_FAMILY} using background-request.png as the image reference.',
        f'Replace only the flat {color} marked background regions; preserve the source geometry.',
        'Save a 3:2 image, ideally 1200x800; a proportional 2K output is accepted.',
        'Set background.image to that saved local path and rerun the same render command.',
    ]
    payload = {'requested_model': MODEL_FAMILY, 'prompt': prompt,
               'target_size': list(PANEL), 'marker_color': color, 'instructions': instructions}
    try:
        reference.save(image_path)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    except OSError:
        raise BackgroundError('BACKGROUND_OUTPUT', 'Could not write the background request files.',
                              'Use a writable work directory.') from None
    details = {'request_image': str(image_path), 'request_json': str(json_path),
               'prompt': prompt, 'target_size': list(PANEL), 'requested_model': MODEL_FAMILY}
    return reference, details


def _load_background(source):
    try:
        with Image.open(source) as image:
            if image.info.get(CORRECTION_MARKER):
                raise BackgroundError('BACKGROUND_ALREADY_CORRECTED',
                                      'This background already contains the local color correction.',
                                      'Use the original generated image or work/background-generated.png for replay.')
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise BackgroundError('BACKGROUND_IMAGE_SIZE', 'The background image exceeds the supported pixel limit.',
                                      'Use a 1200x800 or proportional 2K image.')
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
            if abs(width / height / (PANEL[0] / PANEL[1]) - 1) > 0.005:
                raise BackgroundError('BACKGROUND_ASPECT', 'The background image must have a 3:2 aspect ratio.',
                                      'Generate the exact requested canvas; do not crop or change source alignment.')
            rgb = oriented.convert('RGB')
            return rgb if rgb.size == PANEL else rgb.resize(PANEL, Image.Resampling.LANCZOS)
    except BackgroundError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError):
        raise BackgroundError('BACKGROUND_IMAGE_INVALID', 'The background image could not be decoded.',
                              'Provide a readable local PNG or JPEG from the background completion request.') from None


def _correct_boundary(photo, background, coverage):
    """Adjust only new RGB pixels beside one full-span rectangular source edge.

    Smooth a clipped color-offset field, never the photograph or its texture.
    Reject strong chromatic disagreement before and after smoothing to limit
    color transfer across different materials. This does not identify objects.
    Unsupported coverage geometry is explicitly skipped.
    """
    if photo.mode != 'RGB' or background.mode != 'RGB' or coverage.mode != 'L':
        raise ValueError('Expected RGB, RGB and L images.')
    if photo.size != background.size or photo.size != coverage.size:
        raise ValueError('All image sizes must match.')
    result = background.copy()
    width, height = photo.size
    histogram = coverage.histogram()
    meta = {'applied': False, 'algorithm': CORRECTION_VERSION,
            'max_delta': MAX_DELTA, 'strip_width': STRIP_WIDTH,
            'tangent_radius': TANGENT_RADIUS, 'decay_pixels': DECAY_PIXELS,
            'chroma_full_confidence': MAX_DELTA / 2, 'chroma_reject': MAX_DELTA}
    if histogram[255] == width * height:
        return result, {**meta, 'reason': 'fully-covered'}
    box = coverage.getbbox()
    if not box or sum(histogram[1:255]) or histogram[255] != (box[2] - box[0]) * (box[3] - box[1]):
        return result, {**meta, 'reason': 'not-binary-solid-rectangle'}
    left, top, right, bottom = box
    if top == 0 and bottom == height and right == width and left > 0:
        axis, boundary, inward, gap, source_width, side = 0, left, 1, left, width - left, 'left'
    elif top == 0 and bottom == height and left == 0 and right < width:
        axis, boundary, inward, gap, source_width, side = 0, right - 1, -1, width - right, right, 'right'
    elif left == 0 and right == width and bottom == height and top > 0:
        axis, boundary, inward, gap, source_width, side = 1, top, 1, top, height - top, 'top'
    elif left == 0 and right == width and top == 0 and bottom < height:
        axis, boundary, inward, gap, source_width, side = 1, bottom - 1, -1, height - bottom, bottom, 'bottom'
    else:
        return result, {**meta, 'reason': 'requires-one-full-span-missing-strip'}
    if any(ImageChops.multiply(channel, coverage).getbbox()
           for channel in ImageChops.difference(photo, background).split()):
        raise ValueError('Background input must already retain the source photograph.')

    source_pixels, generated_pixels, output_pixels = photo.load(), background.load(), result.load()
    tangent_length = photo.size[1 - axis]
    strip = min(STRIP_WIDTH, gap, source_width)

    def point(normal, tangent):
        return (normal, tangent) if axis == 0 else (tangent, normal)

    raw = []
    confidence = []
    for tangent in range(tangent_length):
        source_sum, generated_sum = [0, 0, 0], [0, 0, 0]
        for offset in range(strip):
            source = source_pixels[point(boundary + inward * offset, tangent)]
            generated = generated_pixels[point(boundary - inward * (offset + 1), tangent)]
            for channel in range(3):
                source_sum[channel] += source[channel]
                generated_sum[channel] += generated[channel]
        residual = [(a - b) / strip for a, b in zip(source_sum, generated_sum)]
        # Clipping first could conceal a large cross-channel disagreement.
        spread = max(residual) - min(residual)
        t = max(0.0, min(1.0, (MAX_DELTA - spread) / (MAX_DELTA / 2)))
        confidence.append(t * t * (3 - 2 * t))
        raw.append([max(-MAX_DELTA, min(MAX_DELTA, value)) * confidence[-1]
                    for value in residual])

    prefix = [[0.0, 0.0, 0.0]]
    for row in raw:
        prefix.append([a + b for a, b in zip(prefix[-1], row)])
    field = []
    for tangent in range(tangent_length):
        first, last = max(0, tangent - TANGENT_RADIUS), min(tangent_length, tangent + TANGENT_RADIUS + 1)
        field.append([(prefix[last][channel] - prefix[first][channel]) / (last - first)
                      for channel in range(3)])

    for tangent, delta in enumerate(field):
        # A rejected row remains untouched even next to accepted rows.
        delta = [value * confidence[tangent] for value in delta]
        for distance in range(min(gap, DECAY_PIXELS)):
            t = 1 - distance / (DECAY_PIXELS - 1)
            weight = t * t * (3 - 2 * t)
            location = point(boundary - inward * (distance + 1), tangent)
            original = generated_pixels[location]
            output_pixels[location] = tuple(max(0, min(255, round(value + weight * shift)))
                                            for value, shift in zip(original, delta))
    return result, {**meta, 'applied': True, 'side': side, 'coverage_bbox': list(box),
                    'rejected_rows': sum(value == 0 for value in confidence),
                    'tapered_rows': sum(0 < value < 1 for value in confidence),
                    'field_min': [min(row[c] * confidence[i] for i, row in enumerate(field)) for c in range(3)],
                    'field_max': [max(row[c] * confidence[i] for i, row in enumerate(field)) for c in range(3)]}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        # Never forward the Authorization header to another URL.
        raise BackgroundError('BACKGROUND_API_REDIRECT', 'Ark returned an unexpected redirect.',
                              API_FAILURE_REMEDY)


def _open_request(request, timeout):
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def _safe_identifier(value, secrets):
    if (isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9._-]{1,160}', value)
            and not any(secret and secret in value for secret in secrets)):
        return value
    return None


def _response_details(result, request_id, secrets):
    """Expose only bounded identifiers and shape, never provider message/image data."""
    root = result if isinstance(result, dict) else {}
    data = root.get('data')
    entries = [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
    errors = [item['error'] for item in [root, *entries] if item.get('error') is not None]
    details = {'response_shape': {
        'json_object': isinstance(result, dict),
        'data_count': len(data) if isinstance(data, list) else None,
        'has_b64': any(isinstance(item.get('b64_json'), str) and bool(item['b64_json']) for item in entries),
        'has_url': any(isinstance(item.get('url'), str) and bool(item['url']) for item in entries),
        'has_error': bool(errors),
    }}
    safe_request = (_safe_identifier(root.get('request_id'), secrets) or
                    _safe_identifier(request_id, secrets))
    if safe_request:
        details['request_id'] = safe_request
    for error in errors:
        code = _safe_identifier(error.get('code'), secrets) if isinstance(error, dict) else None
        if code:
            details['provider_error_code'] = code
            break
    return details


def _generate(reference, prompt, settings, key, model):
    secrets = [key, *(os.environ.get(name, '') for name in ('ARK_API_KEY', 'SEEDREAM_API_KEY'))]
    buffer = io.BytesIO()
    reference.save(buffer, format='PNG')
    payload = {'model': model, 'prompt': prompt, 'size': '1200x800',
               'output_format': 'png', 'response_format': 'b64_json', 'watermark': False,
               'image': ['data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')]}
    request = Request(GENERATIONS_URL, data=json.dumps(payload).encode('utf-8'),
                      headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
    request_id = None
    try:
        with _open_request(request, settings['timeout']) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            headers = getattr(response, 'headers', {})
            request_id = headers.get('x-request-id') if headers is not None else None
    except BackgroundError:
        raise
    except HTTPError as exc:
        status = exc.code if isinstance(exc.code, int) else None
        result = None
        try:
            headers = getattr(exc, 'headers', {})
            request_id = headers.get('x-request-id') if headers is not None else None
            raw = exc.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) <= MAX_RESPONSE_BYTES:
                result = json.loads(raw)
        except (OSError, ValueError, TypeError, UnicodeError, HTTPException):
            pass
        finally:
            exc.close()
        details = _response_details(result, request_id, secrets)
        details['http_status'] = status
        code = 'BACKGROUND_PROVIDER_ERROR' if details['response_shape']['has_error'] else 'BACKGROUND_API_FAILED'
        raise BackgroundError(code, 'The Seedream background request failed.',
                              API_FAILURE_REMEDY, details) from None
    except (TimeoutError, URLError, OSError, ValueError, HTTPException):
        raise BackgroundError('BACKGROUND_API_FAILED', 'The Seedream background request did not complete.',
                              API_FAILURE_REMEDY) from None
    details = _response_details(None, request_id, secrets)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise BackgroundError('BACKGROUND_API_RESPONSE', 'The Seedream response exceeded the supported size.',
                              API_FAILURE_REMEDY, details)
    try:
        result = json.loads(raw)
    except (ValueError, TypeError, UnicodeError):
        raise BackgroundError('BACKGROUND_API_RESPONSE', 'Seedream did not return a valid JSON response.',
                              API_FAILURE_REMEDY, details) from None
    details = _response_details(result, request_id, secrets)
    if details['response_shape']['has_error']:
        raise BackgroundError('BACKGROUND_PROVIDER_ERROR', 'Seedream reported a background generation error.',
                              API_FAILURE_REMEDY, details)
    try:
        data = result.get('data') if isinstance(result, dict) else None
        if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
            raise ValueError()
        encoded = data[0].get('b64_json')
        if not isinstance(encoded, str) or len(encoded) > (MAX_IMAGE_BYTES * 4 // 3 + 4):
            raise ValueError()
        decoded = base64.b64decode(encoded, validate=True)
        if not decoded or len(decoded) > MAX_IMAGE_BYTES:
            raise ValueError()
    except (ValueError, TypeError, binascii.Error, UnicodeError):
        # Server error bodies can echo credentials. Never include their text.
        raise BackgroundError('BACKGROUND_API_RESPONSE', 'Seedream did not return one valid base64 image.',
                              API_FAILURE_REMEDY, details) from None
    reported_model = result.get('model')
    # A supplied model must identify Pro before metadata is redacted. Invalid
    # formatting or echoed credentials must not bypass this check or be logged.
    if (reported_model is not None and reported_model != 'doubao-seedream-5-0-pro'
            and not _valid_model(reported_model)):
        raise BackgroundError('BACKGROUND_MODEL_MISMATCH', 'Ark returned a model outside Seedream 5.0 Pro.',
                              API_FAILURE_REMEDY, details)
    try:
        image = _load_background(io.BytesIO(decoded))
    except BackgroundError as exc:
        raise BackgroundError(exc.code, exc.message, API_FAILURE_REMEDY,
                              {**exc.details, **details}) from None
    metadata = {}
    actual_model = _safe_identifier(result.get('model'), secrets)
    actual_request = _safe_identifier(result.get('request_id', request_id), secrets)
    if actual_model:
        metadata['model'] = actual_model
    if actual_request:
        metadata['request_id'] = actual_request
    return image, metadata


def fill_missing(photo: Image.Image, coverage: Image.Image, work_dir: Path,
                 settings: dict | None = None) -> tuple[Image.Image, dict]:
    """Return a filled panel; coverage=255 always retains the original pixel."""
    if not isinstance(photo, Image.Image) or photo.mode != 'RGB' or photo.size != PANEL:
        raise BackgroundError('BACKGROUND_INPUT', 'The background reference must be a 1200x800 RGB image.')
    if not isinstance(coverage, Image.Image) or coverage.mode != 'L' or coverage.size != PANEL:
        raise BackgroundError('BACKGROUND_INPUT', 'Source coverage must be a 1200x800 L mask.')
    histogram = coverage.histogram()
    if sum(histogram[1:255]):
        raise BackgroundError('BACKGROUND_INPUT', 'Source coverage must contain only 0 and 255.')
    config = validate_config({} if settings is None else settings, Path.cwd())
    missing = histogram[0]
    metadata = {'required': bool(missing), 'method': 'not-required',
                'image_generation_used': False, 'filled_pixels': missing,
                'retained_pixels': histogram[255]}
    if not missing:
        return photo.copy(), metadata
    if not histogram[255]:
        raise BackgroundError('BACKGROUND_INPUT', 'The background reference must retain part of the original photograph.')
    work = Path(work_dir).expanduser().resolve()
    output_path = work / 'background-filled.png'
    raw_path = work / 'background-generated.png'
    try:
        work.mkdir(parents=True, exist_ok=True)
        # This module never treats a previous run's output as a successful fill.
        if output_path.exists() and config.get('image') != output_path:
            output_path.unlink()
        if raw_path.exists() and 'image' not in config:
            raw_path.unlink()
    except OSError:
        raise BackgroundError('BACKGROUND_OUTPUT', 'The background work directory is not writable.') from None
    if 'image' in config:
        background = _load_background(config['image'])
        metadata['method'] = 'offline-image'
        metadata['external_background_used'] = True
        metadata['external_model_verified'] = False
    else:
        reference, request_details = _write_request(photo, coverage, work, config.get('scene'))
        key = os.environ.get('ARK_API_KEY') or os.environ.get('SEEDREAM_API_KEY')
        model = config.get('model') or os.environ.get('REDTHREAD_SEEDREAM_MODEL') or os.environ.get('ARK_IMAGE_MODEL')
        if not key or not model:
            raise BackgroundError('NEED_BACKGROUND', 'The background completion request is ready for the host image tool.',
                                  'Generate the saved request with Seedream 5.0 Pro, save the result, set background.image, and rerun the same render command.',
                                  request_details)
        if not _valid_model(model):
            raise BackgroundError('INVALID_CONFIG', 'The configured model must identify the Seedream 5.0 Pro deployment.',
                                  'Set REDTHREAD_SEEDREAM_MODEL or ARK_IMAGE_MODEL to its ep- deployment ID.')
        background, response_metadata = _generate(reference, request_details['prompt'], config, key, model)
        metadata.update(response_metadata)
        metadata.update({'method': 'seedream', 'image_generation_used': True})
    try:
        # Replay always starts with the normalized, uncorrected model image.
        background.save(raw_path)
        # The generator cannot contribute any RGB pixels to the retained area.
        output = Image.composite(photo, background, coverage)
        output, correction = _correct_boundary(photo, output, coverage)
        pnginfo = PngInfo()
        if correction['applied']:
            pnginfo.add_text(CORRECTION_MARKER, CORRECTION_VERSION)
        output.save(output_path, pnginfo=pnginfo)
    except OSError:
        raise BackgroundError('BACKGROUND_OUTPUT', 'Could not save the completed background panel.',
                              API_FAILURE_REMEDY if metadata['image_generation_used'] else
                              'Use a writable work directory.') from None
    metadata.update({'provider': 'seedream', 'requested_model': MODEL_FAMILY, 'output_path': str(output_path),
                     'raw_background_path': str(raw_path), 'seam_correction': correction})
    return output, metadata
