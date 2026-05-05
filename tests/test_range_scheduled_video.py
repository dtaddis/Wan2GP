from __future__ import annotations

import importlib
import shutil
import sys
import types
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins"))
sys.modules.setdefault("gradio", types.SimpleNamespace(Error=RuntimeError))
prompts = importlib.import_module("wan2gp-process-full-video.prompt_schedule")
output_paths = importlib.import_module("wan2gp-process-full-video.output_paths")


def write_manifest(tmp_path: Path, text: str) -> str:
    path = tmp_path / "hotb_001_opening.csv"
    path.write_text(text, encoding="utf-8")
    return str(path)


class RangeScheduledVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp_path = Path(__file__).resolve().parent / "_tmp_range_scheduled_video"
        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)
        self.tmp_path.mkdir(parents=True)

    def tearDown(self):
        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)

    def test_manifest_infers_ranges_and_accepts_time_formats(self):
        manifest = write_manifest(
            self.tmp_path,
            'end,positive\n'
            '20,"Logo, with comma"\n'
            '00:40,"Moorland"\n'
            '00:01:00.50,"Interior"\n',
        )
        schedule = prompts.parse_range_prompt_manifest(manifest)
        self.assertEqual(
            [(item.start_seconds, item.end_seconds, item.prompt) for item in schedule],
            [(0.0, 20.0, "Logo, with comma"), (20.0, 40.0, "Moorland"), (40.0, 60.5, "Interior")],
        )

    def test_manifest_rejects_non_increasing_end_times(self):
        manifest = write_manifest(self.tmp_path, 'end,positive\n00:20,"A"\n00:20,"B"\n')
        with self.assertRaises(Exception):
            prompts.parse_range_prompt_manifest(manifest)

    def test_manifest_rejects_non_csv_files_before_decoding(self):
        video_path = self.tmp_path / "not_a_manifest.mp4"
        video_path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\xba")
        with self.assertRaises(Exception) as caught:
            prompts.parse_range_prompt_manifest(str(video_path))
        self.assertIn(".csv", str(caught.exception))

    def test_range_prompt_resolution_combines_overlapping_rows(self):
        manifest = write_manifest(self.tmp_path, 'end,positive\n00:20,"Logo"\n00:40,"Moorland"\n')
        schedule = prompts.parse_range_prompt_manifest(manifest)
        prompt = prompts.resolve_range_prompt_for_chunk(schedule, 18.0, 24.0, "Colorize video.", "")
        self.assertEqual(prompt, "Colorize video.\n\nLogo\n\nMoorland")

    def test_manifest_output_path_uses_manifest_stem_and_source_range(self):
        source = self.tmp_path / "hound.mkv"
        source.write_text("", encoding="utf-8")
        output = output_paths.build_manifest_requested_output_path(str(source), "", "hotb_001_opening", "720p", 600.0, 1200.0)
        self.assertEqual(output.name, "hotb_001_opening_720p_10m00s_20m00s.mkv")

    def test_manifest_collision_variant_uses_v_suffix(self):
        output = self.tmp_path / "hotb_001_opening_720p_10m00s_20m00s.mp4"
        output.write_text("", encoding="utf-8")
        variant = output_paths.make_versioned_output_variant(output)
        self.assertEqual(Path(variant).name, "hotb_001_opening_720p_10m00s_20m00s_v002.mp4")


if __name__ == "__main__":
    unittest.main()
