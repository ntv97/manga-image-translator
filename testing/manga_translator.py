import asyncio
import base64
import io

import cv2
from aiohttp.web_middlewares import middleware
from omegaconf import OmegaConf
import langcodes
import requests
import os
import re
import torch
import time
import logging
import numpy as np
from PIL import Image
from typing import List, Tuple, Union
from aiohttp import web
from marshmallow import Schema, fields, ValidationError

from manga_translator.utils.threading import Throttler

from .args import DEFAULT_ARGS, translator_chain
from .utils import (
    BASE_PATH,
    LANGUAGE_ORIENTATION_PRESETS,
    ModelWrapper,
    Context,
    PriorityLock,
    load_image,
    dump_image,
    replace_prefix,
    visualize_textblocks,
    add_file_logger,
    remove_file_logger,
    is_valuable_text,
    rgb2hex,
    hex2rgb,
    get_color_name,
    natural_sort,
    sort_regions,
)

from .detection import DETECTORS, dispatch as dispatch_detection, prepare as prepare_detection
from .upscaling import dispatch as dispatch_upscaling, prepare as prepare_upscaling, UPSCALERS
from .ocr import OCRS, dispatch as dispatch_ocr, prepare as prepare_ocr
from .textline_merge import dispatch as dispatch_textline_merge
from .mask_refinement import dispatch as dispatch_mask_refinement
from .inpainting import INPAINTERS, dispatch as dispatch_inpainting, prepare as prepare_inpainting
from .translators import (
    TRANSLATORS,
    VALID_LANGUAGES,
    LanguageUnsupportedException,
    TranslatorChain,
    dispatch as dispatch_translation,
    prepare as prepare_translation,
)
from .colorization import dispatch as dispatch_colorization, prepare as prepare_colorization
from .rendering import dispatch as dispatch_rendering, dispatch_eng_render
#from .save import save_result

# Will be overwritten by __main__.py if module is being run directly (with python -m)
logger = logging.getLogger('manga_translator')


def set_main_logger(l):
    global logger
    logger = l


class TranslationInterrupt(Exception):
    """
    Can be raised from within a progress hook to prematurely terminate
    the translation.
    """
    pass


class MangaTranslator():

    def __init__(self, params: dict = None):
        self._progress_hooks = []
        self._add_logger_hook()

        params = params or {}
        self.parse_init_params(params)
        self.result_sub_folder = ''

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

    def parse_init_params(self, params: dict):
        self.verbose = params.get('verbose', False)
        self.ignore_errors = params.get('ignore_errors', False)
        # check mps for apple silicon or cuda for nvidia
        device = 'mps' if torch.backends.mps.is_available() else 'cuda'
        self.device = device if params.get('use_gpu', False) else 'cpu'
        self._gpu_limited_memory = params.get('use_gpu_limited', False)
        if self._gpu_limited_memory and not self.using_gpu:
            self.device = device
        if self.using_gpu and ( not torch.cuda.is_available() and not torch.backends.mps.is_available()):
            raise Exception(
                'CUDA or Metal compatible device could not be found in torch whilst --use-gpu args was set.\n' \
                'Is the correct pytorch version installed? (See https://pytorch.org/)')
        if params.get('model_dir'):
            ModelWrapper._MODEL_DIR = params.get('model_dir')
        self.kernel_size=int(params.get('kernel_size'))
        os.environ['INPAINTING_PRECISION'] = params.get('inpainting_precision', 'fp32')

    @property
    def using_gpu(self):
        return self.device.startswith('cuda') or self.device == 'mps'

    def _preprocess_params(self, ctx: Context):
        # params auto completion
        # TODO: Move args into ctx.args and only calculate once, or just copy into ctx
        for arg in DEFAULT_ARGS:
            ctx.setdefault(arg, DEFAULT_ARGS[arg])

        if 'direction' not in ctx:
            if ctx.force_horizontal:
                ctx.direction = 'h'
            elif ctx.force_vertical:
                ctx.direction = 'v'
            else:
                ctx.direction = 'auto'
        if 'alignment' not in ctx:
            if ctx.align_left:
                ctx.alignment = 'left'
            elif ctx.align_center:
                ctx.alignment = 'center'
            elif ctx.align_right:
                ctx.alignment = 'right'
            else:
                ctx.alignment = 'auto'
        if ctx.prep_manual:
            ctx.renderer = 'none'
        ctx.setdefault('renderer', 'manga2eng' if ctx.manga2eng else 'default')

        if ctx.selective_translation is not None:
            ctx.selective_translation.target_lang = ctx.target_lang
            ctx.translator = ctx.selective_translation
        elif ctx.translator_chain is not None:
            ctx.target_lang = ctx.translator_chain.langs[-1]
            ctx.translator = ctx.translator_chain
        else:
            ctx.translator = TranslatorChain(f'{ctx.translator}:{ctx.target_lang}')
        if ctx.gpt_config:
            ctx.gpt_config = OmegaConf.load(ctx.gpt_config)

        if ctx.filter_text:
            ctx.filter_text = re.compile(ctx.filter_text)

        if ctx.font_color:
            colors = ctx.font_color.split(':')
            try:
                ctx.font_color_fg = hex2rgb(colors[0])
                ctx.font_color_bg = hex2rgb(colors[1]) if len(colors) > 1 else None
            except:
                raise Exception(f'Invalid --font-color value: {ctx.font_color}. Use a hex value such as FF0000')

    async def _run_colorizer(self, ctx: Context):
        return await dispatch_colorization(ctx.colorizer, device=self.device, image=ctx.input, **ctx)

    async def _run_upscaling(self, ctx: Context):
        return (await dispatch_upscaling(ctx.upscaler, [ctx.img_colorized], ctx.upscale_ratio, self.device))[0]

    async def _run_detection(self, ctx: Context):
        return await dispatch_detection(ctx.detector, ctx.img_rgb, ctx.detection_size, ctx.text_threshold,
                                        ctx.box_threshold,
                                        ctx.unclip_ratio, ctx.det_invert, ctx.det_gamma_correct, ctx.det_rotate,
                                        ctx.det_auto_rotate,
                                        self.device, self.verbose)

    async def _run_ocr(self, ctx: Context):
        textlines = await dispatch_ocr(ctx.ocr, ctx.img_rgb, ctx.textlines, ctx, self.device, self.verbose)

        new_textlines = []
        for textline in textlines:
            if textline.text.strip():
                if ctx.font_color_fg:
                    textline.fg_r, textline.fg_g, textline.fg_b = ctx.font_color_fg
                if ctx.font_color_bg:
                    textline.bg_r, textline.bg_g, textline.bg_b = ctx.font_color_bg
                new_textlines.append(textline)
        return new_textlines

    async def _run_textline_merge(self, ctx: Context):
        text_regions = await dispatch_textline_merge(ctx.textlines, ctx.img_rgb.shape[1], ctx.img_rgb.shape[0],
                                                     verbose=self.verbose)
        new_text_regions = []
        for region in text_regions:
            if len(region.text) >= ctx.min_text_length \
                    and not is_valuable_text(region.text) \
                    or (not ctx.no_text_lang_skip and langcodes.tag_distance(region.source_lang, ctx.target_lang) == 0):
                if region.text.strip():
                    logger.info(f'Filtered out: {region.text}')
            else:
                if ctx.font_color_fg or ctx.font_color_bg:
                    if ctx.font_color_bg:
                        region.adjust_bg_color = False
                new_text_regions.append(region)
        text_regions = new_text_regions

        # Sort ctd (comic text detector) regions left to right. Otherwise right to left.
        # Sorting will improve text translation quality.
        text_regions = sort_regions(text_regions, right_to_left=True if ctx.detector != 'ctd' else False)
        return text_regions

    async def _run_text_translation(self, ctx: Context):
        translated_sentences = \
            await dispatch_translation(ctx.translator,
                                       [region.text for region in ctx.text_regions],
                                       ctx.use_mtpe,
                                       ctx, 'cpu' if self._gpu_limited_memory else self.device)

        for region, translation in zip(ctx.text_regions, translated_sentences):
            if ctx.uppercase:
                translation = translation.upper()
            elif ctx.lowercase:
                translation = translation.upper()
            region.translation = translation
            region.target_lang = ctx.target_lang
            region._alignment = ctx.alignment
            region._direction = ctx.direction

        # Filter out regions by their translations
        new_text_regions = []
        for region in ctx.text_regions:
            # TODO: Maybe print reasons for filtering
            if not ctx.translator == 'none' and (region.translation.isnumeric() \
                    or ctx.filter_text and re.search(ctx.filter_text, region.translation)
                    or not ctx.translator == 'original' and region.text.lower().strip() == region.translation.lower().strip()):
                if region.translation.strip():
                    logger.info(f'Filtered out: {region.translation}')
            else:
                new_text_regions.append(region)
        return new_text_regions

    async def _run_mask_refinement(self, ctx: Context):
        return await dispatch_mask_refinement(ctx.text_regions, ctx.img_rgb, ctx.mask_raw, 'fit_text',
                                              ctx.mask_dilation_offset, ctx.ignore_bubble, self.verbose,self.kernel_size)

    async def _run_inpainting(self, ctx: Context):
        return await dispatch_inpainting(ctx.inpainter, ctx.img_rgb, ctx.mask, ctx.inpainting_size, self.device,
                                         self.verbose)

    async def _run_text_rendering(self, ctx: Context):
        if ctx.renderer == 'none':
            output = ctx.img_inpainted
        # manga2eng currently only supports horizontal left to right rendering
        elif ctx.renderer == 'manga2eng' and ctx.text_regions and LANGUAGE_ORIENTATION_PRESETS.get(
                ctx.text_regions[0].target_lang) == 'h':
            output = await dispatch_eng_render(ctx.img_inpainted, ctx.img_rgb, ctx.text_regions, ctx.font_path, ctx.line_spacing)
        else:
            output = await dispatch_rendering(ctx.img_inpainted, ctx.text_regions, ctx.font_path, ctx.font_size,
                                              ctx.font_size_offset,
                                              ctx.font_size_minimum, not ctx.no_hyphenation, ctx.render_mask, ctx.line_spacing)
        return output

    def _result_path(self, path: str) -> str:
        """
        Returns path to result folder where intermediate images are saved when using verbose flag
        or web mode input/result images are cached.
        """
        return "/home/nhivo/Image"
        #return os.path.join(BASE_PATH, 'result', self.result_sub_folder, path)

    def add_progress_hook(self, ph):
        self._progress_hooks.append(ph)

    async def _report_progress(self, state: str, finished: bool = False):
        for ph in self._progress_hooks:
            await ph(state, finished)

    def _add_logger_hook(self):
        # TODO: Pass ctx to logger hook
        LOG_MESSAGES = {
            'upscaling': 'Running upscaling',
            'detection': 'Running text detection',
            'ocr': 'Running ocr',
            'mask-generation': 'Running mask refinement',
            'translating': 'Running text translation',
            'rendering': 'Running rendering',
            'colorizing': 'Running colorization',
            'downscaling': 'Running downscaling',
        }
        LOG_MESSAGES_SKIP = {
            'skip-no-regions': 'No text regions! - Skipping',
            'skip-no-text': 'No text regions with text! - Skipping',
            'error-translating': 'Text translator returned empty queries',
            'cancelled': 'Image translation cancelled',
        }
        LOG_MESSAGES_ERROR = {
            # 'error-lang':           'Target language not supported by chosen translator',
        }

        async def ph(state, finished):
            if state in LOG_MESSAGES:
                logger.info(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_SKIP:
                logger.warn(LOG_MESSAGES_SKIP[state])
            elif state in LOG_MESSAGES_ERROR:
                logger.error(LOG_MESSAGES_ERROR[state])

        self.add_progress_hook(ph)


class MangaTranslatorWeb(MangaTranslator):
    """
    Translator client that executes tasks on behalf of the webserver in web_main.py.
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
        #self.host = params.get('host', '127.0.0.1')
        #if self.host == '0.0.0.0':
        #    self.host = '127.0.0.1'
        #self.port = params.get('port', 5003)
        self.nonce = params.get('nonce', '')
        self.ignore_errors = params.get('ignore_errors', True)
        self._task_id = None
        self._params = None

    async def trydetect(self, translation_params: dict = None):
        params = translation_params
        params = params or {}
        ctx = Context(**params)
        self._preprocess_params(ctx)
        temp = Image.open("/home/nhivo/Image/input.jpg")
        img = np.array(temp.convert('RGB'))
        ctx.img_rgb = ctx.input = img
        ctx.result = None
        logger.info('....Detection Attemption...')
        ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(ctx)

        await self.bboxes(ctx)
        await self.ocr(ctx)
        await self.textlinemerge(ctx)
        await self.translation(ctx)
        await self.maskrefinement(ctx)
        await self.inpainting(ctx)
        await self.rendering(ctx)
        cv2.imwrite("/home/nhivo/Image/result.png", cv2.cvtColor(ctx.img_rendered, cv2.COLOR_RGB2BGR))

    async def bboxes(self, ctx):
        img_bbox_raw = np.copy(ctx.img_rgb)
        for txtln in ctx.textlines:
            cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
        cv2.imwrite("/home/nhivo/Image/bboxes_unfiltered.png", cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))
        #self.img_bbox_raw = img_bbox_raw

    async def ocr(self, ctx):
        ctx.textlines = await self._run_ocr(ctx)
        if not ctx.textlines:
            ctx.result = ctx.img_rgb

    async def textlinemerge(self, ctx):
        ctx.text_regions = await self._run_textline_merge(ctx)
        bboxes = visualize_textblocks(cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB), ctx.text_regions)
        cv2.imwrite("/home/nhivo/Image/bboxes.png", bboxes)

    async def translation(self, ctx):
        ctx.text_regions = await self._run_text_translation(ctx)

    async def maskrefinement(self, ctx):
        ctx.mask = await self._run_mask_refinement(ctx)

    async def inpainting(self, ctx):
        inpaint_input_img = await dispatch_inpainting('none', ctx.img_rgb, ctx.mask, ctx.inpainting_size,
                                                          self.using_gpu, self.verbose)
        cv2.imwrite("/home/nhivo/Image/inpaint_input.png", cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("/home/nhivo/Image/mask_final.png", ctx.mask)
        ctx.img_inpainted = await self._run_inpainting(ctx)
        ctx.gimp_mask = np.dstack((cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), ctx.mask))
        cv2.imwrite("/home/nhivo/Image/inpainted.png", cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR))

    async def rendering(self, ctx):
        ctx.img_rendered = await self._run_text_rendering(ctx)

