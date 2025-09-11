import os
from pathlib import Path
import typing as t
import tempfile
import bentoml
from bentoml.io import JSON, File
import torchaudio as ta
import torch

# Define a runner class for Chatterbox
class ChatterboxRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = False
    
    def __init__(self):
        from chatterbox.tts import ChatterboxTTS
        # Initialize with GPU if available, fallback to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ChatterboxTTS.from_pretrained(device=device)
    
    @bentoml.Runnable.method(batchable=False)
    def synthesize(
        self, 
        text: str, 
        output_path: str, 
        audio_prompt_path: str = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,  # Fixed parameter name
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0
    ) -> str:
        """
        Synthesize speech using Chatterbox TTS with correct parameters
        """
        # Generate audio with Chatterbox using correct parameters
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            wav = self.model.generate(
                text, 
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p
            )
        else:
            wav = self.model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p
            )
        
        # Save audio file using torchaudio
        ta.save(output_path, wav, self.model.sr)
        return output_path

# Create the actual BentoML runner
chatterbox_runner = bentoml.Runner(ChatterboxRunnable, name="chatterbox_runner")

# Define the service and register the runner
svc = bentoml.Service(name="chatterbox", runners=[chatterbox_runner])

# Basic API endpoint
@svc.api(input=JSON(), output=File(mime_type="audio/wav"))
async def synthesize(input_data: dict) -> t.IO[t.Any]:
    text = input_data.get("text", "Hello, this is Chatterbox TTS.")
    exaggeration = input_data.get("exaggeration", 0.5)
    cfg_weight = input_data.get("cfg_weight", 0.5)  # Fixed parameter name
    temperature = input_data.get("temperature", 0.8)
    repetition_penalty = input_data.get("repetition_penalty", 1.2)
    min_p = input_data.get("min_p", 0.05)
    top_p = input_data.get("top_p", 1.0)
    
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, "output.wav")
    
    # Handle optional audio prompt for voice cloning
    audio_prompt_path = None
    if "audio_prompt_base64" in input_data:
        import base64
        # Decode base64 audio prompt and save to temp file
        audio_data = base64.b64decode(input_data["audio_prompt_base64"])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile.write(audio_data)
            audio_prompt_path = tmpfile.name
    
    await chatterbox_runner.synthesize.async_run(
        text, 
        output_path, 
        audio_prompt_path,
        exaggeration,
        cfg_weight,
        temperature,
        repetition_penalty,
        min_p,
        top_p
    )
    
    # Clean up temp file if it was created
    if audio_prompt_path and os.path.exists(audio_prompt_path):
        os.unlink(audio_prompt_path)
    
    return open(output_path, 'rb')

# Expressive endpoint with optimized defaults
@svc.api(input=JSON(), output=File(mime_type="audio/wav"))
async def synthesize_expressive(input_data: dict) -> t.IO[t.Any]:
    """
    Endpoint optimized for expressive/dramatic speech
    """
    text = input_data.get("text", "This is dramatic expressive speech!")
    exaggeration = input_data.get("exaggeration", 0.7)  # Higher default
    cfg_weight = input_data.get("cfg_weight", 0.3)  # Lower for more expressive speech
    temperature = input_data.get("temperature", 0.9)  # Higher for more variation
    
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, "expressive_output.wav")
    
    await chatterbox_runner.synthesize.async_run(
        text, 
        output_path, 
        None,
        exaggeration,
        cfg_weight,
        temperature,
        1.2,  # repetition_penalty
        0.05, # min_p
        1.0   # top_p
    )
    
    return open(output_path, 'rb')