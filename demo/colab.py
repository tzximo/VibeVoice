# Original Code: https://github.com/microsoft/VibeVoice/blob/main/demo/gradio_demo.py
"""
VibeVoice Gradio Demo 
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Iterator
from datetime import datetime
import threading
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import os
import traceback
import shutil
import re  # Added for timestamp feature
import uuid # Added for timestamp feature

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference, VibeVoiceGenerationOutput
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed

# New dependency for silence removal feature
# Please install with: pip install pydub
try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# !pip install tqdm

import os, requests, urllib.request, urllib.error
from tqdm.auto import tqdm

def download_file(url, download_file_path, redownload=False):
    """Download a single file with urllib + tqdm progress bar."""

    base_path = os.path.dirname(download_file_path)
    os.makedirs(base_path, exist_ok=True)

    # skip logic
    if os.path.exists(download_file_path):
        if redownload:
            os.remove(download_file_path)
            tqdm.write(f"♻️ Redownloading: {os.path.basename(download_file_path)}")
        elif os.path.getsize(download_file_path) > 0:
            tqdm.write(f"✔️ Skipped (already exists): {os.path.basename(download_file_path)}")
            return True

    try:
        request = urllib.request.urlopen(url)
        total = int(request.headers.get('Content-Length', 0))
    except urllib.error.URLError as e:
        print(f"❌ Error: Unable to open URL: {url}")
        print(f"Reason: {e.reason}")
        return False

    with tqdm(total=total, desc=os.path.basename(download_file_path), unit='B', unit_scale=True, unit_divisor=1024) as progress:
        try:
            urllib.request.urlretrieve(
                url, 
                download_file_path,
                reporthook=lambda count, block_size, total_size: progress.update(block_size)
            )
        except urllib.error.URLError as e:
            print(f"❌ Error: Failed to download {url}")
            print(f"Reason: {e.reason}")
            return False

    tqdm.write(f"⬇️ Downloaded: {os.path.basename(download_file_path)}")
    return True


def download_model(repo_id, download_folder="./", redownload=False):
    # Download all files from a HuggingFace repo without HF_TOKEN using urllib.
    """
    In Google Colab, downloading models from Hugging Face can be unnecessarily frustrating.  
    Even when a model is completely open-source and does NOT require a license agreement or token,  
    Colab often forces you to provide a Hugging Face token anyway.  

    That means you have to:  
    1. Go to Hugging Face,  
    2. Generate a Access Tokens, Enter Password,  
    3. Paste it into Colab’s secret keys,  
    4. Restart the runtime,  

    all just to download a model that should be publicly accessible.  
    It’s a waste of time and breaks the flow of experimentation.  

    This function avoids that hassle by directly fetching the file list via the Hugging Face TOKEN  
    and downloading the files with `aria2c`, no token required (unless the repo truly requires a license).  
    """  
    # normalize empty string as current dir
    if not download_folder.strip():
        download_folder = "."
    url = f"https://huggingface.co/api/models/{repo_id}"
    download_dir = os.path.abspath(f"{download_folder.rstrip('/')}/{repo_id.split('/')[-1]}")
    os.makedirs(download_dir, exist_ok=True)

    print(f"📂 Download directory: {download_dir}")

    response = requests.get(url)
    if response.status_code != 200:
        print("❌ Error:", response.status_code, response.text)
        return None

    data = response.json()
    siblings = data.get("siblings", [])
    files = [f["rfilename"] for f in siblings]

    print(f"📦 Found {len(files)} files in repo '{repo_id}'. Checking cache ...")

    for file in tqdm(files, desc="Processing files", unit="file"):
        file_url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
        file_path = os.path.join(download_dir, file)
        download_file(file_url, file_path, redownload=redownload)

    return download_dir







# NEW FEATURE: Function to generate unique filenames for output
def generate_file_name(text):
    """Generates a unique, clean filename based on the script's first line."""
    output_dir = "./podcast_audio"
    os.makedirs(output_dir, exist_ok=True)
    # Clean the text to get a base for the filename
    cleaned = re.sub(r"^\s*speaker\s*\d+\s*:\s*", "", text, flags=re.IGNORECASE)
    short = cleaned[:30].strip()
    short = re.sub(r'[^a-zA-Z0-9\s]', '', short)
    short = short.lower().strip().replace(" ", "_")
    if not short:
        short = "podcast_output"
    # Add a unique identifier
    unique_name = f"{short}_{uuid.uuid4().hex[:6]}"

    return os.path.join(output_dir, unique_name)


class VibeVoiceDemo:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5):
        """Initialize the VibeVoice demo with model loading."""
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.load_model()
        self.setup_voice_presets()
        self.load_example_scripts()  # Load example scripts

    def load_model(self):
        """Load the VibeVoice model and processor."""
        print(f"Loading processor & model from {self.model_path}")
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map='cuda',
        )
        self.model.eval()
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        if hasattr(self.model.model, 'language_model'):
            print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}, creating it.")
            os.makedirs(voices_dir, exist_ok=True)
        self.voice_presets = {}
        audio_files = [f for f in os.listdir(voices_dir)
                    if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')) and os.path.isfile(os.path.join(voices_dir, f))]
        for audio_file in audio_files:
            name = os.path.splitext(audio_file)[0]
            full_path = os.path.join(voices_dir, audio_file)
            self.voice_presets[name] = full_path
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {name: path for name, path in self.voice_presets.items() if os.path.exists(path)}
        if not self.available_voices:
            print("Warning: No voice presets found.")
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")

    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])

    def trim_silence_from_numpy(self, audio_np: np.ndarray, sample_rate: int, silence_thresh: int = -45, min_silence_len: int = 100, keep_silence: int = 50) -> np.ndarray:
        """Removes silence from a NumPy audio array using pydub."""
        if not PYDUB_AVAILABLE:
            raise ImportError("pydub is not installed. Please run `pip install pydub` to use this feature.")
        
        audio_int16 = (audio_np * 32767).astype(np.int16)
        sound = AudioSegment(
            data=audio_int16.tobytes(),
            sample_width=audio_int16.dtype.itemsize,
            frame_rate=sample_rate,
            channels=1
        )
        audio_chunks = split_on_silence(
            sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=keep_silence
        )
        if not audio_chunks:
            return np.array([0.0], dtype=np.float32)
        
        combined = sum(audio_chunks)
        samples = np.array(combined.get_array_of_samples())
        trimmed_audio_np = samples.astype(np.float32) / 32767.0
        return trimmed_audio_np

    def generate_podcast_with_timestamps(self,
                                 num_speakers: int,
                                 script: str,
                                 speaker_1: str = None,
                                 speaker_2: str = None,
                                 speaker_3: str = None,
                                 speaker_4: str = None,
                                 cfg_scale: float = 1.3,
                                 remove_silence: bool = False,
                                 progress=gr.Progress()):
        try:
            self.stop_generation = False
            self.is_generating = True

            # --- Input Validation and Setup ---
            if not script.strip(): raise gr.Error("Error: Please provide a script.")
            if remove_silence and not PYDUB_AVAILABLE:
                raise gr.Error("Error: 'Trim Silence' is enabled, but pydub is not installed. Please run `pip install pydub`.")
            script = script.replace("’", "'")
            if not 1 <= num_speakers <= 4: raise gr.Error("Error: Number of speakers must be between 1 and 4.")

            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            for i, speaker in enumerate(selected_speakers):
                if not speaker or speaker not in self.available_voices:
                    raise gr.Error(f"Error: Please select a valid speaker for Speaker {i+1}.")

            voice_samples = [self.read_audio(self.available_voices[name]) for name in selected_speakers]
            if any(len(vs) == 0 for vs in voice_samples): raise gr.Error("Error: Failed to load one or more audio files.")

            lines = script.strip().split('\n')
            formatted_script_lines = []
            for line in lines:
                line = line.strip()
                if not line: continue
                if re.match(r'Speaker\s*\d+:', line, re.IGNORECASE):
                    formatted_script_lines.append(line)
                else:
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")

            if not formatted_script_lines: raise gr.Error("Error: Script is empty after formatting.")

            # --- Prepare for Generation ---
            log = f"🎙️ Starting generation for {len(formatted_script_lines)} lines...\n"
            timestamps = {}
            current_time = 0.0
            sample_rate = 24000
            total_lines = len(formatted_script_lines)

            base_filename = generate_file_name(formatted_script_lines[0])
            final_audio_path = base_filename + ".wav"
            final_json_path = base_filename + ".json"

            # --- Open file and write chunks sequentially (MEMORY EFFICIENT) ---
            with sf.SoundFile(final_audio_path, 'w', samplerate=sample_rate, channels=1, subtype='PCM_16') as audio_file:
                for i, line in enumerate(formatted_script_lines):
                    if self.stop_generation:
                        log += "\n🛑 Generation stopped by user."
                        break

                    progress(i / total_lines, desc=f"Generating line {i+1}/{total_lines}")

                    match = re.match(r'Speaker\s*(\d+):\s*(.*)', line, re.IGNORECASE)
                    if not match: continue

                    speaker_idx = int(match.group(1)) - 1
                    text_content = match.group(2).strip()

                    if speaker_idx < 0 or speaker_idx >= len(voice_samples):
                        log += f"\n⚠️ Warning: Speaker ID {speaker_idx + 1} is out of bounds. Skipping line."
                        continue

                    inputs = self.processor(
                        text=[line], voice_samples=[voice_samples], padding=True, return_tensors="pt"
                    )

                    output_waveform: VibeVoiceGenerationOutput = self.model.generate(
                        **inputs, max_new_tokens=None, cfg_scale=cfg_scale, tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False}, verbose=False, refresh_negative=True
                    )

                    audio_np = output_waveform.speech_outputs[0].cpu().float().numpy().squeeze()

                    # NEW FEATURE: Remove silence if enabled
                    if remove_silence:
                        original_duration = len(audio_np) / sample_rate
                        audio_np = self.trim_silence_from_numpy(audio_np, sample_rate)
                        new_duration = len(audio_np) / sample_rate
                        # log += f"✅ Line {i+1}: Trimmed from {original_duration:.2f}s to {new_duration:.2f}s\n"

                    duration = len(audio_np) / sample_rate
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    audio_file.write(audio_int16)

                    timestamps[str(i + 1)] = {
                        "text": text_content, "speaker_id": speaker_idx,
                        "start": current_time, "end": current_time + duration
                    }
                    current_time += duration
                    # log += f"✅ Line {i+1}: '{text_content[:30]}...' ({duration:.2f}s)\n"

            # --- Finalize and Save JSON ---
            progress(1.0, desc="Saving timestamp file...")
            with open(final_json_path, "w") as f:
                json.dump(timestamps, f, indent=2)

            log += f"\n✨ Generation successful!\n🎵 Audio: {final_audio_path}\n📄 Timestamps: {final_json_path}\n"
            self.is_generating = False

            return final_audio_path, final_audio_path, final_json_path, log, gr.update(visible=True), gr.update(visible=False)

        except Exception as e:
            self.is_generating = False
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            traceback.print_exc()
            return None, None, None, error_msg, gr.update(visible=True), gr.update(visible=False)

    def stop_audio_generation(self):
        if self.is_generating:
            self.stop_generation = True
            print("🛑 Audio generation stop requested")

    def load_example_scripts(self):
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []
        if not os.path.exists(examples_dir): return
        txt_files = sorted([f for f in os.listdir(examples_dir) if f.lower().endswith('.txt')])
        for txt_file in txt_files:
            try:
                with open(os.path.join(examples_dir, txt_file), 'r', encoding='utf-8') as f:
                    script = f.read().strip()
                if script: self.example_scripts.append([self._get_num_speakers_from_script(script), script])
            except Exception as e:
                print(f"Error loading example {txt_file}: {e}")

    def _get_num_speakers_from_script(self, script: str) -> int:
        speakers = set(re.findall(r'^Speaker\s+(\d+)\s*:', script, re.MULTILINE | re.IGNORECASE))
        return max(int(s) for s in speakers) if speakers else 1

def create_demo_interface(demo_instance: VibeVoiceDemo):
    with gr.Blocks(
        title="VibeVoice - AI Podcast Generator",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple", neutral_hue="slate")
    ) as interface:

        gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ Vibe Podcasting</h1>
            <p style="font-size: 1.2em; color: #555;">Generate Long-form Multi-speaker AI Podcasts with VibeVoice and Timestamps</p>
        </div>
        """)

        with gr.Row():
            # Left column - Settings
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🎛️ Podcast Settings")
                    num_speakers = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Number of Speakers")

                    gr.Markdown("### 🎭 Speaker Selection")
                    speaker_selections = []
                    available_voices = list(demo_instance.available_voices.keys())
                    defaults = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']
                    for i in range(4):
                        val = defaults[i] if i < len(defaults) and defaults[i] in available_voices else None
                        speaker = gr.Dropdown(choices=available_voices, value=val, label=f"Speaker {i+1}", visible=(i < 2))
                        speaker_selections.append(speaker)

                    with gr.Accordion("🎤 Upload Custom Voices", open=False):
                        upload_audio = gr.File(label="Upload Voice Samples", file_count="multiple", file_types=["audio"])
                        process_upload_btn = gr.Button("Add Uploaded Voices to List")

                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        cfg_scale = gr.Slider(minimum=1.0, maximum=2.0, value=1.3, step=0.05, label="CFG Scale")
                        # NEW FEATURE: Silence removal checkbox
                        remove_silence_checkbox = gr.Checkbox(label="Trim Silence from Segments", value=False,)

            # Right column - Generation
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 📝 Script Input")
                    script_input = gr.Textbox(label="Conversation Script", placeholder="Enter script here...", lines=10)

                    with gr.Row():
                        random_example_btn = gr.Button("🎲 Random Example", scale=1)
                        generate_btn = gr.Button("🚀 Generate Podcast & Timestamps", variant="primary", scale=2)

                    stop_btn = gr.Button("🛑 Stop Generation", variant="stop", visible=False)

                    gr.Markdown("### 🎵 **Generated Output**")
                    audio_output = gr.Audio(label="Play Generated Podcast", type="filepath")
                    with gr.Accordion("📦 Download Files", open=False):
                      download_file = gr.File(label="Download Audio File (.wav)")
                      json_file_output = gr.File(label="Download Timestamps (.json)")
                      log_output = gr.Textbox(label="Generation Log", lines=8, interactive=False)

        with gr.Accordion("💡 **Usage Tips & Examples**", open=True):
            gr.Markdown("""
            - **Silence Trimming:** The 'Trim Silence' option in Advanced Settings can create a more tightly edited podcast. It requires the `pydub` library (`pip install pydub`).
            - **Process:** Audio is generated line-by-line and written directly to a file. The UI updates only when generation is complete.
            - **Playback & Download:** You can play the audio in the interface and get a separate link to download the file.
            - **Output:** You get a playable audio file and a downloadable timestamp `.json` file.
            """)
            gr.Examples(examples=demo_instance.example_scripts, inputs=[num_speakers, script_input], label="Try these example scripts:")

        # --- Backend Functions ---
        def process_and_refresh_voices(uploaded_files):
            if not uploaded_files: return [gr.update() for _ in speaker_selections] + [None]
            voices_dir = os.path.join(os.path.dirname(__file__), "voices")
            for f in uploaded_files: shutil.copy(f.name, os.path.join(voices_dir, os.path.basename(f.name)))
            demo_instance.setup_voice_presets()
            new_choices = list(demo_instance.available_voices.keys())
            return [gr.update(choices=new_choices) for _ in speaker_selections] + [None]

        def update_speaker_visibility(num):
            return [gr.update(visible=(i < num)) for i in range(4)]

        def handle_generate_click():
            return gr.update(visible=False), gr.update(visible=True)

        num_speakers.change(fn=update_speaker_visibility, inputs=num_speakers, outputs=speaker_selections)
        process_upload_btn.click(fn=process_and_refresh_voices, inputs=upload_audio, outputs=speaker_selections + [upload_audio])

        gen_event = generate_btn.click(
            fn=handle_generate_click,
            outputs=[generate_btn, stop_btn]
        ).then(
            fn=demo_instance.generate_podcast_with_timestamps,
            inputs=[num_speakers, script_input] + speaker_selections + [cfg_scale, remove_silence_checkbox],
            outputs=[audio_output, download_file, json_file_output, log_output, generate_btn, stop_btn],
        )

        stop_btn.click(fn=demo_instance.stop_audio_generation, cancels=[gen_event])

        def load_random_example():
            import random
            return random.choice(demo_instance.example_scripts) if demo_instance.example_scripts else (2, "Speaker 0: No examples loaded.")

        random_example_btn.click(fn=load_random_example, outputs=[num_speakers, script_input])

    return interface

def main():
    """Main function to run the demo without command-line arguments."""

    model_repo = "microsoft/VibeVoice-1.5B"
    # big_model="WestZhang/VibeVoice-Large-pt"
    model_folder=download_model(model_repo, download_folder="./", redownload=False)
    inference_steps = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    print("🎙️ Initializing VibeVoice Demo with Timestamp Support...")

    if not PYDUB_AVAILABLE:
        print("\n⚠️  Warning: `pydub` is not installed. The 'Trim Silence' feature will be disabled.")
        print("   To enable it, please run: pip install pydub\n")


    demo_instance = VibeVoiceDemo(
        model_path=model_folder,
        device=device,
        inference_steps=inference_steps
    )

    interface = create_demo_interface(demo_instance)

    print(f"🚀 Launching demo...")
    try:
        interface.queue().launch(
            share=True,
            debug=True
        )
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
