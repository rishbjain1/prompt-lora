import tempfile
import unittest
from collections import Counter
from pathlib import Path

import data_prep


class DataPrepTests(unittest.TestCase):
    def test_extracts_explicit_and_derived_briefs(self):
        source = """# Vault

## Scenes

### Rain walk

**Tool:** Example | **Use case:** Moody city walk at night.

> A courier crosses a rain-soaked street.
>
> Neon reflections ripple underfoot.

### Window portrait

> An older sailor watches dawn through a salt-streaked window.
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vault.md"
            path.write_text(source, encoding="utf-8")
            pairs, _ = data_prep.extract_vault(path)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["brief"], "Moody city walk at night.")
        self.assertEqual(pairs[0]["brief_source"], "explicit")
        self.assertEqual(pairs[1]["brief_source"], "derived")
        self.assertIn("older sailor", pairs[1]["brief"].lower())

    def test_privacy_skips_topics_and_drops_named_person_lines(self):
        hits = Counter()
        self.assertIsNone(
            data_prep.apply_privacy_filter("A cinematic Teleparty office scene.", hits)
        )
        cleaned = data_prep.apply_privacy_filter(
            "Keep this visual line.\nHelmut Newton frames the portrait.\nKeep this too.",
            hits,
        )

        self.assertEqual(cleaned, "Keep this visual line.\nKeep this too.")
        self.assertEqual(hits["topic:teleparty"], 1)
        self.assertEqual(hits["person:helmut newton"], 1)

    def test_private_explicit_brief_falls_back_to_derived_brief(self):
        hits = Counter()
        pair = data_prep.build_pair(
            "Desert car",
            "A driver crosses a bright salt flat.",
            "Helmut Newton-inspired motorsport editorial.",
            hits,
        )

        self.assertIsNotNone(pair)
        self.assertEqual(pair["brief_source"], "derived")
        self.assertIn("driver", pair["brief"].lower())

    def test_split_is_seeded_and_disjoint(self):
        pairs = [
            {"brief": str(index), "prompt": "p", "brief_source": "derived"}
            for index in range(20)
        ]
        train_a, val_a = data_prep.split_pairs(pairs, seed=42)
        train_b, val_b = data_prep.split_pairs(pairs, seed=42)

        self.assertEqual((train_a, val_a), (train_b, val_b))
        self.assertEqual((len(train_a), len(val_a)), (18, 2))
        self.assertFalse(
            {row["brief"] for row in train_a} & {row["brief"] for row in val_a}
        )


if __name__ == "__main__":
    unittest.main()
