"""Linie de bază generativă (BLIP): produce un titlu nou, nu alege dintr-o listă.

Folosită pentru comparația „regăsire vs. generare” din evaluare și din interfață.
Modelul se încarcă la prima utilizare (≈1 GB).
"""

from __future__ import annotations

import threading

from PIL import Image

BLIP_MODEL = "Salesforce/blip-image-captioning-base"


class BlipCaptioner:
    def __init__(self, model_name: str = BLIP_MODEL, device: str | None = None):
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
        from transformers.utils import logging as hf_logging

        from .encoders.base import pick_device

        hf_logging.set_verbosity_error()
        self._torch = torch
        self.model_name = model_name
        self.device = device or pick_device()
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device).eval()

    def caption(self, image: Image.Image, max_new_tokens: int = 30, num_beams: int = 3) -> str:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
        return self.processor.decode(ids[0], skip_special_tokens=True).strip()


class LazyCaptioner:
    """Încarcă BLIP doar când e cerut prima oară."""

    def __init__(self):
        self._inst: BlipCaptioner | None = None
        self._lock = threading.Lock()

    def caption(self, image: Image.Image) -> str:
        with self._lock:
            if self._inst is None:
                self._inst = BlipCaptioner()
            return self._inst.caption(image)
