import os
from pyannote.audio import Pipeline
from typing import Any, Dict, List
import torchaudio
from pydub import AudioSegment
import uuid
import pandas as pd
import numpy as np
import json
# ComfyUI imports (assuming the necessary base classes exist)
from nodes import NODE_CLASS_MAPPINGS  # Import for registering custom nodes
# from .utils import save_temp_audio_file  # Utility for handling audio uploads (if needed)

class SpeakerDiarizationNode:
    def __init__(self):
        # Node metadata
        self.name = "Speaker Diarization"
        self.category = "Audio Processing"
        self.description = "Performs speaker diarization on an audio file."

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "audio": ("AUDIO",),  # Assuming ComfyUI supports AUDIO type
                "hf_token": ("STRING",),   # Hugging Face token input
            }
        }

    RETURN_TYPES = ("speaker_segments",)  # Returning a list of speaker segments
    RETURN_NAMES = ("speaker_segments",)

    FUNCTION = "diarize_audio"

    def diarize_audio(self, audio: dict, hf_token: str) -> List[Dict[str, Any]]:

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        
        audio_save_path = os.path.join(os.getcwd() , "input" , f"{uuid.uuid1()}.wav")
        
        torchaudio.save(audio_save_path, audio['waveform'].squeeze(
            0), audio["sample_rate"])
        
        diarization = pipeline(audio_save_path)
        
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print( f"start : {turn.start}, end: {turn.end}, speaker: {speaker}")
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        
        return (speaker_segments,)  

class WhisperDiarizationNode:
    
    # Node metadata
    def __init__(self) -> None:
        self.name = "Speech To Speaker"
        self.category = "Audio Processing"
        self.description = "Add Speaker to whisper segments"
        
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "whisper_segments": ("whisper_alignment",), 
                "speaker_segments": ("speaker_segments",),  
            }
        }
        
    RETURN_TYPES = ("whisper_alignment",)  
    RETURN_NAMES = ("segments_alignment",)

    FUNCTION = "convert"
    CATEGORY = "pyannote"
    
    def convert(self,whisper_segments: dict, speaker_segments: dict):
        diarize_df = pd.DataFrame(speaker_segments)
        print("Diarization DataFrame columns:", diarize_df.columns) 
    
        enriched_segments = []
        
        for seg in whisper_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            
            # Calculate overlap between the current segment and all diarization segments
            diarize_df["intersection"] = np.minimum(diarize_df["end"], seg_end) - np.maximum(diarize_df["start"], seg_start)
            diarize_df["union"] = np.maximum(diarize_df["end"], seg_end) - np.minimum(diarize_df["start"], seg_start)
            
            # Keep only positive overlaps
            dia_tmp = diarize_df[diarize_df["intersection"] > 0]
            
            if not dia_tmp.empty:
                # Find the speaker with the maximum overlap                
                if "speaker" in dia_tmp.columns:
                    speaker = dia_tmp.groupby("speaker")["intersection"].sum().idxmax()
                    seg["speaker"] = speaker
                else:
                    print("Speaker column missing.")
                    seg["speaker"] = "Unknown"
            else:
                # Assign 'Unknown' if no overlap is found
                seg["speaker"] = "Unknown"
            
            enriched_segments.append(seg)
        
        return (enriched_segments,)



NODE_CLASS_MAPPINGS = { 
    "Speaker Diarization" : SpeakerDiarizationNode,
    "Whisper Segments to Speaker": WhisperDiarizationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "Speaker Diarization" : "Speaker Diarization", 
     "Whisper Segments to Speaker": "Whisper Segments to Speaker"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
