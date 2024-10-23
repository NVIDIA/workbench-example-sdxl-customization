# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### This module contains the chatui gui for having a conversation. ###

import functools
from typing import Any, Dict, List, Tuple, Union

import gradio as gr
import shutil
import os
import subprocess
import time
import sys
import torch
import logging
import gc

from chatui import assets, chat_client
from chatui.utils import logger
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline

PATH = "/"
TITLE = "SDXL Image Generation"
OUTPUT_TOKENS = 250
MAX_DOCS = 5
GENERATED_IMG_DIR = "/project/data/generated_images"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

### Load in CSS here for components that need custom styling. ###

_LOCAL_CSS = """
#contextbox {
    overflow-y: scroll !important;
    max-height: 400px;
}

#params .tabs {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}
#params .tabitem[style="display: block;"] {
    flex-grow: 1;
    display: flex !important;
}
#params .gap {
    flex-grow: 1;
}
#params .form {
    flex-grow: 1 !important;
}
#params .form > :last-child{
    flex-grow: 1;
}
#accordion {
}
#rag-inputs .svelte-1gfkn6j {
    color: #76b900;
}
#rag-inputs .svelte-s1r2yt {
    color: #76b900;
}
"""

INSTRUCTIONS = """
<br /> Welcome to the SDXL Image Generation app! To get started with model inference,

&nbsp;&nbsp;1. Select a model from the dropdown

&nbsp;&nbsp;2. Input an image generation prompt into the textbox and press ENTER

&nbsp;&nbsp;3. Generated images are auto-saved to the project under ``data/generated_images``

\n\n <b>Important:</b> Make sure you disconnect any kernels that may still be utilizing the GPU!
"""

sys.stdout = logger.Logger("/project/code/output.log")

print("--- MODELS: Loading Model " + BASE_MODEL + " ---")
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
print("--- MODELS: Configuring Pipe ---")
pipe.to("cuda")
print("--- MODELS: Model is ready for inference ---")

def build_page(client: chat_client.ChatClient) -> gr.Blocks:
    """
    Build the gradio page to be mounted in the frame.
    
    Parameters: 
        client (chat_client.ChatClient): The chat client running the application. 
    
    Returns:
        page (gr.Blocks): A Gradio page.
    """
    kui_theme, kui_styles = assets.load_theme("kaizen")

    # Get a list of models
    entries = os.listdir("/project/models")
    models = [entry for entry in entries if os.path.isdir(os.path.join("/project/models", entry)) and entry[0] != '.']
    models.insert(0, BASE_MODEL)

    # Prep base model
    logging.basicConfig(level=logging.INFO)
    global pipe

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles + _LOCAL_CSS) as page:
        gr.Markdown(f"# {TITLE}")

        """ Keep state of which model pipe to use. """

        current_pipe = gr.State({"pipe": pipe})

        """ Build the Chat Application. """
        
        with gr.Row(equal_height=True):

            # Left Column will display the chatbot
            with gr.Column(scale=15, min_width=350):
                
                # Main chatbot panel. 
                with gr.Row(equal_height=True):
                    with gr.Column(min_width=350):
                        chatbot = gr.Chatbot(show_label=False, height=575)

                # Message box for user input
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3, min_width=450):
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="Enter your image prompt and press ENTER",
                            container=False,
                            interactive=True,
                        )

                    with gr.Column(scale=1, min_width=150):
                        clear = gr.ClearButton([msg, chatbot], value="Clear history")
            
            # Hidden column to be rendered when the user collapses all settings.
            with gr.Column(scale=1, min_width=100, visible=False) as hidden_settings_column:
                show_settings = gr.Button(value="< Expand", size="sm")
            
            # Right column to display all relevant settings
            with gr.Column(scale=10, min_width=350) as settings_column:
                with gr.Tabs(selected=0) as settings_tabs:
                    with gr.TabItem("Settings", id=0) as model_settings:
                        
                        gr.Markdown(INSTRUCTIONS)
                        
                        model = gr.Dropdown(models, 
                                            label="Select a model", 
                                            elem_id="rag-inputs", 
                                            value=BASE_MODEL)
                        
                        logs = gr.Textbox(label="Console", 
                                          elem_id="rag-inputs", 
                                          lines=12, 
                                          max_lines=12, 
                                          interactive=False)
                        
                    with gr.TabItem("Hide All Settings", id=1) as hide_all_settings:
                        gr.Markdown("")

        def _toggle_hide_all_settings():
            print("--- SETTINGS: Hiding Settings ---")
            return {
                settings_column: gr.update(visible=False),
                hidden_settings_column: gr.update(visible=True),
            }

        def _toggle_show_all_settings():
            print("--- SETTINGS: Expanding Settings ---")
            return {
                settings_column: gr.update(visible=True),
                settings_tabs: gr.update(selected=0),
                hidden_settings_column: gr.update(visible=False),
            }

        hide_all_settings.select(_toggle_hide_all_settings, None, [settings_column, hidden_settings_column])
        show_settings.click(_toggle_show_all_settings, None, [settings_column, settings_tabs, hidden_settings_column])

        def clear_imgs():
            print("--- IMAGES: Clearing Images... ---")
            for file in os.listdir(GENERATED_IMG_DIR):
                if not file.endswith(".png"):
                    continue
                print("--- SETTINGS: Deleting '" + file + "' ---")
                os.remove(os.path.join(GENERATED_IMG_DIR, file))
    
        clear.click(clear_imgs, [], [])

        def load_model(model: str):
            pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            if model == BASE_MODEL:
                print("--- MODELS: Loading Model " + model + " ---")
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    BASE_MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                )
                print("--- MODELS: Configuring Pipe ---")
                pipe.to("cuda")
            else:
                print("--- MODELS: Loading Model: " + model + " ---")
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    BASE_MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                )
                print("--- MODELS: Configuring Pipe ---")
                pipe.to("cuda")
                print("--- MODELS: Loading LoRA Weights for: " + model + " ---")
                pipe.load_lora_weights("/project/models/" + model)
            print("--- MODELS: Model is ready for inference ---")
            return {
                current_pipe: {"pipe": pipe}, 
                msg: gr.update(visible=True),
            }
    
        model.change(load_model, [model], [current_pipe, msg])

        page.load(logger.read_logs, None, logs, every=1)

        """ This helper function builds out the submission function call when a user submits a query. """
        
        _my_build_stream = functools.partial(_stream_predict, client)
        msg.submit(
            _my_build_stream, [msg, 
                               chatbot,
                               current_pipe], [msg, chatbot]
        )

    page.queue()
    return page

def create_img_dir():
    if not os.path.exists(GENERATED_IMG_DIR):
        print("--- IMAGES: Creating Image Directory ---")
        os.makedirs(GENERATED_IMG_DIR)

def get_image_count():
    """Count all .png files in the given directory."""
    count = 0
    for filename in os.listdir(GENERATED_IMG_DIR):
        if filename.endswith('.png'):
            count += 1
    return count

def gen_new_img_name():
    print("--- IMAGES: Created Image 'generated_image-" + str(get_image_count()) + ".png' ---")
    return GENERATED_IMG_DIR + "/generated_image-" + str(get_image_count()) + ".png"

def generate_image(pipe, prompt):
    create_img_dir()
    print("--- IMAGES: Image is generating... ---")
    image = pipe(prompt=prompt).images[0]
    name = gen_new_img_name()
    image.save(name)
    return name

""" This helper function executes and generates a response to the user query. """

def _stream_predict(
    client: chat_client.ChatClient,
    prompt: str,
    chat_history: List[Tuple[str, str]],
    pipe, 
) -> Any:
    
    try:
        yield "", chat_history + [[prompt, "Generating your image..."]]
        filename = generate_image(pipe["pipe"], prompt)
        yield "", chat_history + [[prompt, (filename,)]]
    except Exception as e: 
        yield "", chat_history + [[prompt, "*** ERR: Unable to process query. ***\n\nException: " + str(e)]]
