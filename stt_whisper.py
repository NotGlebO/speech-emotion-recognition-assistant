from dataclasses import dataclass
import numpy as np
from faster_whisper import WhisperModel


@dataclass
class WhisperConfig:
    model_size: str = "small"     # tiny/small/medium
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"          # "en" for English
    beam_size: int = 5


class WhisperSTT:
    def __init__(self, cfg: WhisperConfig):
        self.cfg = cfg
        print("Loading Whisper model...")
        self.model = WhisperModel(self.cfg.model_size, device=self.cfg.device, compute_type=self.cfg.compute_type)
        print("Whisper ready.")

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size
        )
        text = " ".join(s.text.strip() for s in segments if s.text and s.text.strip())
        return text.strip()