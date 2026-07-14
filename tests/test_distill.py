import unittest

from distill import validate_structured_prompt


class ValidateStructuredPromptTests(unittest.TestCase):
    def test_requires_all_six_block_headers(self):
        valid = """SUBJECT — A courier waits.
LOCATION — A rain-soaked station.
ACTION — The courier starts running.
SHOT 1 (0:00–0:15) — He crosses the platform.
CAMERA — Low tracking shot.
STYLE — Dominant blue 60% / Secondary gray 30% / Accent red 10%.
CONSTRAINTS — 16:9. NO slow-motion.
"""
        missing_style = valid.replace("STYLE —", "LOOK —")
        mentioned_in_prose = valid.replace(
            "STYLE — Dominant", "The STYLE block would use dominant"
        )

        self.assertTrue(validate_structured_prompt(valid))
        self.assertFalse(validate_structured_prompt(missing_style))
        self.assertFalse(validate_structured_prompt(mentioned_in_prose))


if __name__ == "__main__":
    unittest.main()
