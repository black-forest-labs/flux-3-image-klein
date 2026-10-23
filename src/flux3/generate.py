"""Prompt (+ references, + boxes) -> image: safety checks, prompt upsampling, reference preparation, sampling."""

import hashlib
import json

from PIL import Image

from flux3 import bbox
from flux3.pipeline import (MAX_REFERENCES, MODELS, WEIGHTS, Pipeline, default_size, prepare_multi_reference,
                      prepare_reference)
from flux3.safety import Safety, exif


class Refused(Exception):
    """Blocked by a safety check."""


class Models:
    """Everything loaded once. The upsampler is loaded first: vLLM reserves its share of the GPU up front."""

    def __init__(self, dits: dict[str, str], pup: str | None, pup_memory: float, offload: bool = False):
        from flux3.pup import Upsampler

        self.upsampler = Upsampler(pup, pup_memory) if pup else None
        shared = WEIGHTS / MODELS[next(iter(dits))]["dit"]  # every weights entry ships the text encoder and autoencoder
        self.pipeline = Pipeline(dits, str(shared / "text_encoder"), str(shared / "ae.safetensors"), offload=offload)
        self._safety = None

    @property
    def safety(self) -> Safety:
        if self._safety is None:
            self._safety = Safety(
                self.pipeline.text_encoder,
                str(WEIGHTS / "nsfw-classifier"),
                self.upsampler,
            )
        return self._safety


def generate(models: Models, model: str, prompt: str, images: list[Image.Image], boxes: list[dict] | None = None,
             width: int | None = None, height: int | None = None, steps: int | None = None,
             guidance: float | None = None, seed: int | None = None, pup: bool = True,
             safety: bool = False) -> tuple[Image.Image, dict]:
    """The image and its record. With ``safety``, raises Refused when the prompt, a reference or the output
    is flagged."""
    if len(images) > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} references")
    defaults = MODELS[model]
    if defaults["guidance"] is None and guidance not in (None, 1.0):
        raise ValueError(f"{model} samples without CFG; drop the guidance")
    steps = steps or defaults["steps"]
    guidance = defaults["guidance"] if guidance is None else guidance

    if safety:
        screened = " ".join([prompt] + [box.get("desc", "") for box in boxes or []])
        if models.safety.prompt_flagged(screened):
            raise Refused("The prompt may name a copyrighted character, a brand or a public figure.")
        for i, image in enumerate(images):
            if reason := models.safety.image_flagged(image):
                raise Refused(f"Reference image {i + 1} may show {reason}.")

    # A finished upsampler answer (its box table fully cited as <id>s) goes to the DiT as it is.
    finished = not boxes and bbox.is_finished(prompt)
    use_pup = pup and models.upsampler is not None and not finished
    caption = None
    if use_pup:
        try:
            caption = models.upsampler(prompt, images, boxes)
        except ValueError as error:  # e.g. a reference beyond the upsampler's aspect limit: use the prompt as typed
            print(f"upsampling skipped: {error}", flush=True)
            use_pup = False
    if caption is None:
        caption = f"{prompt.rstrip()} {json.dumps(bbox.to_canvas(boxes), ensure_ascii=False)}" if boxes else prompt

    if len(images) > 1:
        references, crop = [prepare_multi_reference(image) for image in images], None
    else:
        prepared = [prepare_reference(image) for image in images]
        references = [image for image, _ in prepared]
        crop = prepared[0][1] if prepared and not (width or height) else None
    default_width, default_height = default_size(references)
    width, height = width or default_width, height or default_height
    seed = seed if seed is not None else int(hashlib.sha256(caption.encode()).hexdigest(), 16) % 2**32

    image = models.pipeline.generate(model, caption, references, width, height, steps, guidance, seed)
    if crop is not None:
        image = image.crop(crop)
    if safety and (reason := models.safety.image_flagged(image)):
        raise Refused(f"The generated image may show {reason}, so it was not returned. Try another prompt or seed.")
    record = {"model": model, "prompt": caption, "input_prompt": prompt, "boxes": boxes, "width": image.width,
              "height": image.height, "steps": steps, "guidance": guidance, "seed": seed, "upsampled": use_pup}
    return image, record
