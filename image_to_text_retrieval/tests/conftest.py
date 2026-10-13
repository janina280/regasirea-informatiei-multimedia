import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from i2t.encoders import ModelRegistry, ModelSpec  # noqa: E402
from i2t.encoders.base import Encoder, normalize  # noqa: E402

COLORS = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1)}


class FakeEncoder(Encoder):
    """Imaginile sunt vectorizate după culoarea medie, textele după cuvântul-culoare conținut."""

    model_name = "fake-color"

    def __init__(self, key="fake"):
        self.key = key
        self.calls = 0

    def encode_texts(self, texts, batch_size=64):
        self.calls += len(texts)
        out = []
        for t in texts:
            v = np.array([0.05, 0.05, 0.05], dtype=np.float32)
            for w, c in COLORS.items():
                if w in t.lower():
                    v = np.array(c, dtype=np.float32) + 0.01
            out.append(v)
        return normalize(np.stack(out))

    def encode_images(self, images, batch_size=16):
        arrs = [np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0) for im in images]
        return normalize(np.stack(arrs))


class FakeCaptioner:
    def caption(self, image):
        return "a generated caption"


@pytest.fixture
def fake_encoder():
    return FakeEncoder()


@pytest.fixture
def registry():
    enc = {"fake": FakeEncoder("fake"), "fake2": FakeEncoder("fake2")}
    specs = [ModelSpec(k, k.upper(), "test", "en", (lambda e=e: e)) for k, e in enc.items()]
    reg = ModelRegistry(specs, default="fake")
    reg.encoders = enc
    return reg
