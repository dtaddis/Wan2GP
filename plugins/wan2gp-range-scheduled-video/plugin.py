from __future__ import annotations

import importlib
import time

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin


PlugIn_Name = "Range-Scheduled Video Processor"
PlugIn_Id = "RangeScheduledVideoProcessor"


class RangeScheduledVideoPlugin(WAN2GPPlugin):
    def setup_ui(self):
        self.request_global("get_model_def")
        self.request_global("get_lora_dir")
        self.request_global("get_base_model_type")
        self.request_global("server_config")
        self.request_component("state")
        self.request_component("lset_name")
        self.request_component("refresh_form_trigger")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_config_ui)

    def on_tab_select(self, state: dict) -> str:
        return str(time.time_ns())

    def create_config_ui(self, api_session):
        ui = importlib.import_module("wan2gp-range-scheduled-video.plugin_ui")
        return ui.create_config_ui(self, api_session)
