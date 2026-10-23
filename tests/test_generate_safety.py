import ast
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "flux3"
    / "generate.py"
)


class TestGenerateSafety(unittest.TestCase):

    def setUp(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

        nodes = [
            node for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in ("Models", "generate")
        ]

        self.safety = Mock()
        self.safety.prompt_flagged.return_value = False
        self.safety.image_flagged.return_value = ""

        self.safety_constructor = Mock(return_value=self.safety)

        self.pipeline = Mock()
        self.pipeline.text_encoder = object()
        self.pipeline.generate.return_value = types.SimpleNamespace(
            width=64,
            height=64,
        )

        namespace = {
            "Safety": self.safety_constructor,
            "Pipeline": Mock(return_value=self.pipeline),
            "WEIGHTS": Path("weights"),
            "MODELS": {
                "distilled": {
                    "dit": "flux-3-image-klein",
                    "steps": 16,
                    "guidance": None,
                }
            },
            "MAX_REFERENCES": 10,
            "Image": types.SimpleNamespace(Image=object),
            "bbox": types.SimpleNamespace(
                is_finished=lambda prompt: False
            ),
            "default_size": lambda references: (64, 64),
            "Refused": RuntimeError,
            "hashlib": hashlib,
            "json": json,
        }

        module = ast.Module(body=nodes, type_ignores=[])
        exec(compile(module, str(SOURCE), "exec"), namespace)

        fake_flux3 = types.ModuleType("flux3")
        fake_flux3.__path__ = []

        fake_pup = types.ModuleType("flux3.pup")
        fake_pup.Upsampler = Mock()

        with patch.dict(sys.modules, {
            "flux3": fake_flux3,
            "flux3.pup": fake_pup,
        }):
            self.models = namespace["Models"](
                {"distilled": "dummy_checkpoint"},
                pup=None,
                pup_memory=0.35,
            )

        self.generate = namespace["generate"]

    def run_generation(self, safety):
        return self.generate(
            self.models,
            "distilled",
            "A red bicycle",
            images=[],
            seed=42,
            pup=False,
            safety=safety,
        )

    def test_disabled_safety_does_not_initialize_classifier(self):
        self.run_generation(safety=False)

        self.safety_constructor.assert_not_called()
        self.safety.prompt_flagged.assert_not_called()
        self.safety.image_flagged.assert_not_called()

    def test_enabled_safety_initializes_once_and_is_reused(self):
        self.run_generation(safety=True)

        self.assertEqual(self.safety_constructor.call_count, 1)
        self.assertEqual(self.safety.prompt_flagged.call_count, 1)
        self.assertEqual(self.safety.image_flagged.call_count, 1)

        self.run_generation(safety=True)

        self.assertEqual(self.safety_constructor.call_count, 1)
        self.assertEqual(self.safety.prompt_flagged.call_count, 2)
        self.assertEqual(self.safety.image_flagged.call_count, 2)


if __name__ == "__main__":
    unittest.main()
