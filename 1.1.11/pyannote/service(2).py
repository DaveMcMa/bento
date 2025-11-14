import os
from pathlib import Path
import typing as t
import tempfile
import bentoml
from bentoml.io import JSON, File
import torch

# Define a runner class for Pyannote Speaker Diarization
class PyannoteRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = False
    
    def __init__(self):
        from pyannote.audio import Pipeline
        
        # Hardcoded HuggingFace token
        hf_token = "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"
        
        # Load the speaker diarization pipeline
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))
    
    @bentoml.Runnable.method(batchable=False)
    def diarize(self, audio_file_path: str) -> dict:
        """
        Perform speaker diarization on audio file
        Returns structured data with speaker segments
        """
        try:
            # Run diarization
            diarization = self.pipeline(audio_file_path)
            
            # Convert to structured format
            segments = []
            speakers_found = set()
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_label = f"SPEAKER_{speaker}"
                speakers_found.add(speaker_label)
                
                segment = {
                    "start_time": float(turn.start),
                    "end_time": float(turn.end),
                    "duration": float(turn.end - turn.start),
                    "speaker": speaker_label
                }
                segments.append(segment)
            
            # Calculate speaker statistics
            speaker_stats = {}
            total_duration = sum(seg["duration"] for seg in segments)
            
            for speaker in speakers_found:
                speaker_duration = sum(
                    seg["duration"] for seg in segments 
                    if seg["speaker"] == speaker
                )
                speaker_stats[speaker] = {
                    "total_duration": round(speaker_duration, 2),
                    "percentage": round((speaker_duration / total_duration) * 100, 1) if total_duration > 0 else 0,
                    "segment_count": len([seg for seg in segments if seg["speaker"] == speaker])
                }
            
            return {
                "success": True,
                "segments": segments,
                "speaker_statistics": speaker_stats,
                "total_speakers": len(speakers_found),
                "total_duration": round(total_duration, 2),
                "speakers_found": list(speakers_found)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "segments": [],
                "speaker_statistics": {},
                "total_speakers": 0
            }

# Create the actual BentoML runner
pyannote_runner = bentoml.Runner(PyannoteRunnable, name="pyannote_runner")

# Define the service and register the runner
svc = bentoml.Service(name="pyannote_diarization", runners=[pyannote_runner])

@svc.api(input=File(), output=JSON())
async def diarize_audio(audio_file: t.IO[t.Any]) -> dict:
    """
    Main endpoint: Upload audio file and get speaker diarization results
    """
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_file.read())
        temp_audio_path = tmp_file.name
    
    try:
        # Run diarization
        result = await pyannote_runner.diarize.async_run(temp_audio_path)
        return result
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

@svc.api(input=JSON(), output=JSON())
async def diarize_from_path(input_data: dict) -> dict:
    """
    Alternative endpoint: Provide local file path for diarization
    Useful for batch processing or when file is already on server
    """
    audio_path = input_data.get("audio_path")
    
    if not audio_path or not os.path.exists(audio_path):
        return {
            "success": False,
            "error": "Invalid or missing audio_path in request"
        }
    
    # Run diarization on provided path
    result = await pyannote_runner.diarize.async_run(audio_path)
    return result

@svc.api(input=File(), output=JSON())
async def quick_speaker_count(audio_file: t.IO[t.Any]) -> dict:
    """
    Quick endpoint: Just return number of speakers detected
    """
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_file.read())
        temp_audio_path = tmp_file.name
    
    try:
        # Run diarization
        result = await pyannote_runner.diarize.async_run(temp_audio_path)
        
        # Return simplified response
        return {
            "success": result["success"],
            "total_speakers": result.get("total_speakers", 0),
            "total_duration": result.get("total_duration", 0),
            "speakers_found": result.get("speakers_found", [])
        }
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)