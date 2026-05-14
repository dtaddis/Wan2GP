from __future__ import annotations

import html
import importlib
from pathlib import Path

import gradio as gr

frame_planning = importlib.import_module("wan2gp-process-full-video.frame_planning")
output_paths = importlib.import_module("wan2gp-process-full-video.output_paths")
process_catalog = importlib.import_module("wan2gp-process-full-video.process_catalog")
process_library = importlib.import_module("wan2gp-process-full-video.process_library")
process_runner = importlib.import_module("wan2gp-process-full-video.process_runner")
status_ui = importlib.import_module("wan2gp-process-full-video.status_ui")
video_buffers = importlib.import_module("wan2gp-process-full-video.video_buffers")
common = importlib.import_module("wan2gp-process-full-video.common")
prompts = importlib.import_module("wan2gp-process-full-video.prompt_schedule")
from shared.utils.utils import get_video_info_details


DEFAULT_NEGATIVE_PROMPT = (
    "black and white, grayscale, monochrome, sepia, brown wash, hand-tinted, old archive colorization, "
    "WWII colorization, oversaturated Technicolor, neon colors, cartoon, plastic skin, color bleeding, "
    "muddy shadows, changed text, warped lettering, modern objects, red skin, pink skin, flushed skin, "
    "sunburn, magenta faces, orange faces, ruddy faces, crimson cheeks, tomato red faces, blotchy red skin, "
    "oversaturated skin, waxy skin, flickering color, pulsing color, breathing saturation, temporal color shift"
)


class _CompatProcessContext:
    def __init__(self, **kwargs):
        defaults = {
            "range_prompt_schedule": [],
            "positive_prefix": "",
            "negative_prompt": "",
        }
        defaults.update(kwargs)
        self.__dict__.update(defaults)


process_runner.ProcessContext = _CompatProcessContext


def _choices_for_model(library, model_type: str, user_refs: list[str]):
    choices = library.normal_process_choices(model_type, user_refs)
    return choices, choices[0][1] if choices else ""


def _process_strength_slider_bounds(value: float) -> tuple[float, float]:
    lower = value if value < 0.0 else 0.0
    upper = value if value > 3.0 else 3.0
    return lower, upper


def _process_strength_value(library, process_name: str, main_state: dict | None, user_refs: list[str]) -> float:
    process_definition = library.process_definition(process_name, main_state, user_refs)
    settings = process_definition.get("settings", {}) if isinstance(process_definition, dict) else {}
    if isinstance(process_definition, dict) and process_definition.get("source") == "user":
        user_default = library.user_lora_strength_override_default(process_definition)
        if user_default is not None:
            return user_default
    return common.get_default_process_strength(settings)


def _format_schedule_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    whole = int(seconds)
    frac = seconds - whole
    minutes, sec = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if frac > 0:
        sec_text = f"{sec + frac:06.3f}".rstrip("0").rstrip(".")
    else:
        sec_text = f"{sec:02d}"
    return f"{hours:02d}:{minutes:02d}:{sec_text}" if hours > 0 else f"{minutes:02d}:{sec_text}"


def _build_chunk_prompt_schedule_text(self, manifest_path: str, positive_prefix: str, default_prompt: str, source_path: str, model_type: str, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds) -> str:
    range_schedule = prompts.parse_range_prompt_manifest(manifest_path)
    metadata = get_video_info_details(source_path)
    start_value = prompts.parse_time_input(start_seconds, label="Start", allow_empty=False)
    end_value = prompts.parse_time_input(end_seconds, label="End", allow_empty=True)
    start_frame, end_frame_exclusive, fps_float, total_source_frames = video_buffers.compute_selected_frame_range(metadata, start_value, end_value)
    processing_fps = video_buffers.get_processing_fps(fps_float)
    rules = frame_planning.get_frame_plan_rules(model_type, self.get_model_def)
    chunk_frames = frame_planning.normalize_chunk_frames(common.require_float(chunk_size_seconds, "Chunk Size", minimum=0.1), processing_fps, frame_step=rules.frame_step, minimum_requested_frames=rules.minimum_requested_frames)
    overlap_frames = frame_planning.normalize_overlap_frames(common.require_int(sliding_window_overlap, "Sliding Window Overlap", minimum=1), frame_step=rules.frame_step)
    plans = frame_planning.build_chunk_plan(
        start_frame,
        end_frame_exclusive,
        total_source_frames,
        chunk_frames,
        frame_step=rules.frame_step,
        minimum_requested_frames=rules.minimum_requested_frames,
        overlap_frames=overlap_frames,
    )
    entries: list[str] = []
    actual_done = 0
    for plan in plans:
        unique_frames = max(1, plan.requested_frames - plan.overlap_frames)
        chunk_start = float(actual_done) / float(fps_float)
        chunk_end = float(actual_done + unique_frames) / float(fps_float)
        prompt_text = prompts.resolve_range_prompt_for_chunk(range_schedule, chunk_start, chunk_end, positive_prefix, default_prompt)
        entries.append(f"{_format_schedule_time(chunk_start)}\n{prompt_text}")
        actual_done += unique_frames
    return "\n\n".join(entries)


def _manifest_output_path(source_path: str, output_path: str, manifest_path: str, output_resolution: str, start_seconds, end_seconds) -> str:
    output_text = str(output_path or "").strip()
    if len(output_text) > 0 and not output_text.endswith(("\\", "/")) and not Path(output_text).is_dir():
        return output_text
    manifest_stem = Path(str(manifest_path or "").strip().strip('"')).stem
    start_value = prompts.parse_time_input(start_seconds, label="Start", allow_empty=False)
    end_value = prompts.parse_time_input(end_seconds, label="End", allow_empty=True)
    if hasattr(output_paths, "build_manifest_requested_output_path"):
        return str(output_paths.build_manifest_requested_output_path(source_path, output_text, manifest_stem, output_resolution, start_value, end_value))
    target_dir = Path(output_text) if len(output_text) > 0 else Path(source_path).parent
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in manifest_stem).strip("_") or "process"
    suffix = Path(source_path).suffix or ".mp4"
    return str(target_dir / f"{safe_stem}_{str(output_resolution or 'res').strip() or 'res'}{suffix}")


def create_config_ui(self, api_session):
    if process_catalog.PROCESS_DEFINITIONS_ERROR is not None:
        with gr.Blocks() as plugin_blocks:
            gr.Markdown(f"Process settings configuration error: {html.escape(process_catalog.PROCESS_DEFINITIONS_ERROR)}")
        return plugin_blocks

    library = process_library.ProcessLibrary(
        get_model_def=self.get_model_def,
        get_lora_dir=self.get_lora_dir,
        get_base_model_type=self.get_base_model_type,
    )
    saved_ui_settings = process_catalog.load_saved_process_full_video_settings()
    user_refs = process_catalog.get_saved_user_settings_refs(saved_ui_settings)
    model_type_choices = [
        choice
        for choice in library.model_type_choices(user_refs)
        if str(choice[1] or "").strip() != "__add_user_settings__"
    ]
    default_model_type = process_catalog.DEFAULT_MODEL_TYPE
    valid_model_values = {value for _label, value in model_type_choices}
    if default_model_type not in valid_model_values and model_type_choices:
        default_model_type = model_type_choices[0][1]
    process_choices, default_process_name = _choices_for_model(library, default_model_type, user_refs)
    source_audio_track_choices = [("Auto", "")] + [(f"Audio Track {track_no}", str(track_no)) for track_no in range(1, 10)]
    output_resolution_choices = [("720p", "720p"), ("1080p", "1080p"), ("900p", "900p"), ("540p", "540p"), ("480p", "480p"), ("384p", "384p"), ("320p", "320p"), ("256p", "256p")]
    default_process_strength = _process_strength_value(library, default_process_name, self.state.value if hasattr(self.state, "value") else None, user_refs)
    process_strength_min, process_strength_max = _process_strength_slider_bounds(default_process_strength)

    active_job = {"job": None, "running": False, "cancel_requested": False, "write_state": None}
    preview_state = {"image": None}
    ui_skip = object()

    def refresh_preview(_refresh_id):
        return preview_state["image"]

    def _button_update(label: str, enabled: bool | None):
        return gr.skip() if enabled is None else gr.update(value=label, interactive=enabled)

    def _ui_update(status=ui_skip, output=ui_skip, preview_refresh=ui_skip, *, start_enabled: bool | None = None, abort_enabled: bool | None = None):
        status_update = gr.skip() if status is ui_skip else status
        output_update = gr.skip() if output is ui_skip else status_ui.render_output_file_html(output)
        preview_update = gr.skip() if preview_refresh is ui_skip else preview_refresh
        return status_update, output_update, preview_update, _button_update("Start Process", start_enabled), _button_update("Stop", abort_enabled)

    def _info_exit(message: str, *, output=ui_skip, total_chunks: int = 1, completed_chunks: int = 0, current_chunk: int = 1, continued: bool = False):
        gr.Info(str(message or "").strip())
        return _ui_update(status_ui.render_chunk_status_html(total_chunks, completed_chunks, current_chunk, "Info", str(message or "").strip(), continued=continued), output, ui_skip, start_enabled=True, abort_enabled=False)

    def _reset_live_chunk_status(state: dict) -> None:
        gen = state.get("gen") if isinstance(state, dict) else None
        if not isinstance(gen, dict):
            return
        gen["status"] = ""
        gen["status_display"] = False
        gen["progress_args"] = None
        gen["progress_phase"] = None
        gen["progress_status"] = ""
        gen["preview"] = None

    runner = process_runner.ProcessRunner(
        plugin=self,
        api_session=api_session,
        library=library,
        get_model_def=self.get_model_def,
        active_job=active_job,
        preview_state=preview_state,
        ui_skip=ui_skip,
        ui_update=_ui_update,
        info_exit=_info_exit,
        reset_live_chunk_status=_reset_live_chunk_status,
    )

    def _time_hint_update(value):
        text = str(value or "").strip()
        if len(text) > 0 and ":" not in text:
            return "Plain numbers are seconds. Use 2:00 for two minutes."
        return ""

    def change_model(next_model_type):
        choices, selected = _choices_for_model(library, next_model_type, user_refs)
        strength = _process_strength_value(library, selected, self.state.value if hasattr(self.state, "value") else None, user_refs)
        minimum, maximum = _process_strength_slider_bounds(strength)
        return gr.update(choices=choices, value=selected), gr.update(value=strength, minimum=minimum, maximum=maximum)

    def change_process(next_process_name, main_state):
        strength = _process_strength_value(library, next_process_name, main_state, user_refs)
        minimum, maximum = _process_strength_slider_bounds(strength)
        return gr.update(value=strength, minimum=minimum, maximum=maximum)

    def preview_manifest(manifest_path):
        try:
            schedule = prompts.parse_range_prompt_manifest(manifest_path)
        except gr.Error as exc:
            return common.get_error_message(exc) or "Invalid CSV Manifest."
        manifest_stem = Path(str(manifest_path or "").strip().strip('"')).stem
        last_end = schedule[-1].end_seconds if schedule else 0.0
        return f"{manifest_stem}: {len(schedule)} range(s), ending at {last_end:.2f}s."

    def start_range_process(state, process_name, user_refs, source_path, process_strength, output_path, _prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds, manifest_path, positive_prefix, negative_prompt):
        process_definition = library.process_definition(process_name, state, user_refs)
        process_settings = process_definition.get("settings", {}) if isinstance(process_definition, dict) else {}
        model_type = str(process_settings.get("model_type") or "")
        default_prompt = str(process_settings.get("prompt") or "")
        try:
            prompt_text = _build_chunk_prompt_schedule_text(
                self,
                manifest_path,
                positive_prefix,
                default_prompt,
                source_path,
                model_type,
                chunk_size_seconds,
                sliding_window_overlap,
                start_seconds,
                end_seconds,
            )
            resolved_output_path = _manifest_output_path(source_path, output_path, manifest_path, output_resolution, start_seconds, end_seconds)
        except gr.Error as exc:
            yield _info_exit(common.get_error_message(exc) or "Invalid range-scheduled settings.")
            return
        except Exception as exc:
            yield _info_exit(f"Invalid range-scheduled settings: {exc}")
            return

        previous_negative = process_settings.get("negative_prompt")
        has_previous_negative = "negative_prompt" in process_settings
        if len(str(negative_prompt or "").strip()) > 0:
            process_settings["negative_prompt"] = str(negative_prompt or "").strip()
        try:
            yield from runner.start_process(
                state,
                process_name,
                user_refs,
                source_path,
                process_strength,
                resolved_output_path,
                prompt_text,
                continue_enabled,
                source_audio_track,
                output_resolution,
                target_ratio,
                chunk_size_seconds,
                sliding_window_overlap,
                start_seconds,
                end_seconds,
            )
        finally:
            if has_previous_negative:
                process_settings["negative_prompt"] = previous_negative
            else:
                process_settings.pop("negative_prompt", None)

    def stop_process():
        active_job["cancel_requested"] = True
        write_state = active_job.get("write_state")
        if write_state is not None:
            write_state.stopped = True
        job = active_job.get("job")
        if job is not None and not job.done:
            job.cancel()
            common.plugin_info("Stopping current range-scheduled processing job...")
            return gr.update(value="Start Process", interactive=False), gr.update(value="Stop", interactive=False)
        if active_job.get("running"):
            return gr.update(value="Start Process", interactive=False), gr.update(value="Stop", interactive=False)
        return gr.update(value="Start Process", interactive=True), gr.update(value="Stop", interactive=False)

    with gr.Column():
        with gr.Row():
            gr.Markdown("Process a full video with an end-time CSV prompt manifest. Automatic output names are based on the manifest filename.")
        with gr.Row():
            process_model_type = gr.Dropdown(model_type_choices, value=default_model_type, label="Model", scale=1)
            process_name = gr.Dropdown(process_choices, value=default_process_name, label="Process", scale=3)
        with gr.Row():
            source_path = gr.Textbox(label="Source Video Path File", value="", scale=3)
        with gr.Row():
            manifest_path = gr.Textbox(label="CSV Manifest Path", value="", scale=3)
            manifest_preview = gr.Textbox(label="Manifest", value="", interactive=False, scale=1)
        with gr.Row():
            output_path = gr.Textbox(label="Output File Path or Folder (empty = source folder)", value="", scale=3)
            continue_enabled = gr.Checkbox(label="Continue", value=True, elem_classes="cbx_bottom", scale=1)
        with gr.Row():
            output_resolution = gr.Dropdown(output_resolution_choices, value="720p", label="Output Resolution")
            process_strength = gr.Slider(label="Process Strength (LoRA Multiplier)", minimum=process_strength_min, maximum=process_strength_max, step=0.01, value=default_process_strength)
            chunk_size_seconds = gr.Number(label="Chunk Size (seconds)", value=20.0, precision=2)
            sliding_window_overlap = gr.Slider(label="Sliding Window Overlap", minimum=1, maximum=161, step=8, value=81)
        with gr.Row():
            start_seconds = gr.Textbox(label="Source Start (seconds or MM:SS; 2:00 = two minutes)", value="", placeholder="Usually blank or the segment start in the full movie")
            end_seconds = gr.Textbox(label="Source End (seconds or MM:SS; 2:00 = two minutes)", value="", placeholder="Optional. Plain 2 means 2 seconds.")
            source_audio_track = gr.Dropdown(source_audio_track_choices, value="", label="Source Audio Track")
        time_hint = gr.Markdown(value="", visible=False)
        positive_prefix = gr.Textbox(label="Global Positive Prefix", value="", lines=2)
        negative_prompt = gr.Textbox(label="Global Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=3)
        with gr.Row():
            start_btn = gr.Button("Start Process")
            abort_btn = gr.Button("Stop", interactive=False)
        status_html = gr.HTML(value=status_ui.render_chunk_status_html(0, 0, 0, "Idle", "Waiting to start..."))
        preview_image = gr.Image(label="Last Frame Preview", type="pil")
        output_file = gr.HTML(value=status_ui.render_output_file_html(""))
        preview_refresh = gr.Textbox(value="", visible=False)
        tab_refresh_trigger = gr.Textbox(value="", visible=False)

    self.on_tab_outputs = [tab_refresh_trigger]
    user_refs_state = gr.State(user_refs)
    process_model_type.change(fn=change_model, inputs=[process_model_type], outputs=[process_name, process_strength], queue=False, show_progress="hidden")
    process_name.change(fn=change_process, inputs=[process_name, self.state], outputs=[process_strength], queue=False, show_progress="hidden")
    manifest_path.change(fn=preview_manifest, inputs=[manifest_path], outputs=[manifest_preview], queue=False, show_progress="hidden")
    start_seconds.change(fn=lambda value: gr.update(value=_time_hint_update(value), visible=len(_time_hint_update(value)) > 0), inputs=[start_seconds], outputs=[time_hint], queue=False, show_progress="hidden")
    end_seconds.change(fn=lambda value: gr.update(value=_time_hint_update(value), visible=len(_time_hint_update(value)) > 0), inputs=[end_seconds], outputs=[time_hint], queue=False, show_progress="hidden")
    start_btn.click(
        fn=start_range_process,
        inputs=[
            self.state,
            process_name,
            user_refs_state,
            source_path,
            process_strength,
            output_path,
            gr.State(""),
            continue_enabled,
            source_audio_track,
            output_resolution,
            gr.State(""),
            chunk_size_seconds,
            sliding_window_overlap,
            start_seconds,
            end_seconds,
            manifest_path,
            positive_prefix,
            negative_prompt,
        ],
        outputs=[status_html, output_file, preview_refresh, start_btn, abort_btn],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )
    preview_refresh.change(fn=refresh_preview, inputs=[preview_refresh], outputs=[preview_image], queue=False, show_progress="hidden")
    abort_btn.click(fn=stop_process, outputs=[start_btn, abort_btn], queue=False, show_progress="hidden")
    return None
