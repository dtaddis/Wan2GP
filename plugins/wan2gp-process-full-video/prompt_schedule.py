from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

TIMED_PROMPT_EXAMPLE = "00:00\nA calm cinematic opening shot.\n\n00:30\nThe mood becomes tense and dramatic."
TIMED_PROMPT_TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?$")


@dataclass(frozen=True)
class RangePrompt:
    start_seconds: float
    end_seconds: float
    prompt: str


def parse_time_input(value, *, label: str, allow_empty: bool) -> float | None:
    if value is None:
        return None if allow_empty else 0.0
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise gr.Error(f"{label} must be a finite time value.")
        return max(0.0, float(value))
    text = str(value).strip()
    if len(text) == 0:
        return None if allow_empty else 0.0
    if ":" not in text:
        try:
            seconds = float(text)
            if not math.isfinite(seconds):
                raise ValueError
            return max(0.0, seconds)
        except ValueError as exc:
            raise gr.Error(f"{label} must be a number of seconds, MM:SS(.xx), or HH:MM:SS(.xx).") from exc
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise gr.Error(f"{label} must be a number of seconds, MM:SS(.xx), or HH:MM:SS(.xx).")
    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            if not math.isfinite(seconds):
                raise ValueError
            return max(0.0, minutes * 60.0 + seconds)
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        if not math.isfinite(seconds):
            raise ValueError
        return max(0.0, hours * 3600.0 + minutes * 60.0 + seconds)
    except ValueError as exc:
        raise gr.Error(f"{label} must be a number of seconds, MM:SS(.xx), or HH:MM:SS(.xx).") from exc


def parse_prompt_schedule(prompt_text: str) -> list[tuple[float, str]]:
    text = str(prompt_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) == 0:
        return [(0.0, "")]
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if len(block.strip()) > 0]
    first_line = text.split("\n", 1)[0].strip()
    if len(blocks) <= 1 and not TIMED_PROMPT_TIMESTAMP_RE.fullmatch(first_line):
        return [(0.0, text)]
    schedule: list[tuple[float, str]] = []
    for block in blocks:
        lines = block.split("\n")
        timestamp_line = lines[0].strip()
        if not TIMED_PROMPT_TIMESTAMP_RE.fullmatch(timestamp_line):
            raise gr.Error(
                "Timed prompts must be separated by blank lines, and each block must start with a timestamp like MM:SS(.xx) or HH:MM:SS(.xx).\n\n"
                f"Example:\n{TIMED_PROMPT_EXAMPLE}"
            )
        prompt_body = "\n".join(lines[1:]).strip()
        if len(prompt_body) == 0:
            raise gr.Error(
                "Each timed prompt block must contain prompt text after its timestamp.\n\n"
                f"Example:\n{TIMED_PROMPT_EXAMPLE}"
            )
        schedule.append((float(parse_time_input(timestamp_line, label="Timed prompt timestamp", allow_empty=False) or 0.0), prompt_body))
    return sorted(schedule, key=lambda item: item[0])


def resolve_prompt_for_chunk(prompt_schedule: list[tuple[float, str]], chunk_start_seconds: float, default_prompt: str) -> str:
    prompt_text = str(default_prompt or "")
    for start_seconds, scheduled_prompt in prompt_schedule:
        if float(start_seconds) <= float(chunk_start_seconds) + 1e-9:
            prompt_text = scheduled_prompt
        else:
            break
    return prompt_text


def parse_range_prompt_manifest(manifest_path: str) -> list[RangePrompt]:
    path_text = str(manifest_path or "").strip().strip('"')
    if len(path_text) == 0:
        raise gr.Error("CSV Manifest Path is required.")
    path = Path(path_text)
    if not path.is_file():
        raise gr.Error(f"CSV Manifest not found: {path}")
    if path.suffix.lower() != ".csv":
        raise gr.Error(f"CSV Manifest must be a .csv file: {path}")
    rows = None
    last_decode_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            with path.open("r", encoding=encoding, newline="") as reader:
                rows = list(csv.DictReader(reader))
            break
        except UnicodeDecodeError as exc:
            last_decode_error = exc
            continue
        except OSError as exc:
            raise gr.Error(f"Unable to read CSV Manifest: {path}") from exc
        except csv.Error as exc:
            raise gr.Error(f"Invalid CSV Manifest: {exc}") from exc
    if rows is None:
        message = str(last_decode_error or "unknown text encoding")
        raise gr.Error(f"Unable to read CSV Manifest as UTF-8, UTF-16, or Windows-1252 text: {path}\n{message}")
    if len(rows) == 0:
        raise gr.Error("CSV Manifest must contain at least one prompt row.")
    fieldnames = {str(name or "").strip().casefold() for name in (rows[0].keys() if rows else [])}
    if "end" not in fieldnames or "positive" not in fieldnames:
        raise gr.Error('CSV Manifest must contain "end" and "positive" columns.')

    schedule: list[RangePrompt] = []
    previous_end = 0.0
    for row_index, row in enumerate(rows, start=2):
        normalized = {str(key or "").strip().casefold(): value for key, value in row.items()}
        end_value = normalized.get("end")
        prompt_text = str(normalized.get("positive") or "").strip()
        if len(prompt_text) == 0:
            raise gr.Error(f"CSV Manifest row {row_index} must contain a positive prompt.")
        end_seconds = parse_time_input(end_value, label=f"CSV Manifest row {row_index} end", allow_empty=False)
        end_seconds = float(end_seconds or 0.0)
        if end_seconds <= previous_end + 1e-9:
            raise gr.Error(f"CSV Manifest row {row_index} end time must be greater than the previous end time.")
        schedule.append(RangePrompt(start_seconds=previous_end, end_seconds=end_seconds, prompt=prompt_text))
        previous_end = end_seconds
    return schedule


def resolve_range_prompt_for_chunk(range_schedule: list[RangePrompt], chunk_start_seconds: float, chunk_end_seconds: float, positive_prefix: str = "", default_prompt: str = "") -> str:
    chunk_start = max(0.0, float(chunk_start_seconds))
    chunk_end = max(chunk_start, float(chunk_end_seconds))
    active_prompts = [
        item.prompt
        for item in range_schedule
        if item.start_seconds < chunk_end - 1e-9 and item.end_seconds > chunk_start + 1e-9
    ]
    parts = [str(positive_prefix or "").strip(), *active_prompts]
    prompt_text = "\n\n".join(part for part in parts if len(part) > 0).strip()
    if len(prompt_text) > 0:
        return prompt_text
    return str(default_prompt or "")
