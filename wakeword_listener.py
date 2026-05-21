import json
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
from vosk import Model as VoskModel, KaldiRecognizer


@dataclass
class WakewordConfig:
    sample_rate: int = 16000
    chunk_sec: float = 0.2

    wake_words: List[str] = None  # ["bob", "reset", "exit"]

    silence_stop_sec: float = 3.0
    calibration_sec: float = 2.0
    thresh_mult: float = 2.5
    min_threshold: float = 0.008
    max_record_sec: float = 20.0

    pre_roll_sec: float = 1.0

    vosk_model_path: str = r"./vosk-model-small-en-us-0.15"
    debug_rms: bool = False


def _rms(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def _to_int16_bytes(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


class WakewordListener:
  
    def __init__(self, config: WakewordConfig):
        self.cfg = config
        if not self.cfg.wake_words:
            self.cfg.wake_words = ["bob", "reset", "exit"]

        self.chunk = int(self.cfg.sample_rate * self.cfg.chunk_sec)
        self.pre_roll_chunks = int(self.cfg.pre_roll_sec / self.cfg.chunk_sec)

        self.vosk_model = VoskModel(self.cfg.vosk_model_path)
        self.rec = KaldiRecognizer(self.vosk_model, self.cfg.sample_rate)

        self.state = "IDLE"
        self.pre_roll = deque(maxlen=self.pre_roll_chunks)

        self.recorded: List[np.ndarray] = []
        self.silence_sec = 0.0
        self.record_sec = 0.0
        self.silence_threshold = self.cfg.min_threshold

        self.stream: Optional[sd.InputStream] = None

    def __enter__(self):
        self.stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk,
        )
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _read_chunk(self) -> np.ndarray:
        assert self.stream is not None
        chunk, _ = self.stream.read(self.chunk)
        return chunk[:, 0].copy()

    def _detect_trigger(self, partial_text: str) -> Optional[str]:
        t = partial_text.lower()
        for w in self.cfg.wake_words:
            if w.lower() in t:
                return w.lower()
        return None

    def wait_for_event(self) -> Tuple[str, Optional[np.ndarray]]:
     
        assert self.stream is not None

        self.state = "IDLE"
        self.pre_roll.clear()
        self.recorded = []
        self.silence_sec = 0.0
        self.record_sec = 0.0
        self.silence_threshold = self.cfg.min_threshold
        self.rec.Reset()

        while True:
            chunk = self._read_chunk()
            self.pre_roll.append(chunk)

            if self.state == "IDLE":
                self.rec.AcceptWaveform(_to_int16_bytes(chunk))
                partial = json.loads(self.rec.PartialResult()).get("partial", "")
                trigger = self._detect_trigger(partial)

                if trigger:
                    # for reset/exit: no recording, just return event
                    if trigger in {"reset", "exit"}:
                        return trigger, None

                    # trigger == bob -> start recording with calibration
                    print("✅ Wake word detected. Recording...")

                    print(f"🔧 Calibrating for {self.cfg.calibration_sec}s... Please stay quiet.")
                    n_chunks = int(self.cfg.calibration_sec / self.cfg.chunk_sec)
                    noise_vals = []
                    for _ in range(n_chunks):
                        c = self._read_chunk()
                        self.pre_roll.append(c)
                        noise_vals.append(_rms(c))

                    noise_rms = float(np.median(noise_vals)) if noise_vals else 0.0
                    self.silence_threshold = max(self.cfg.min_threshold, noise_rms * self.cfg.thresh_mult)
                    print(f"🔧 noise_rms={noise_rms:.4f} -> silence_threshold={self.silence_threshold:.4f}")

                    self.state = "RECORDING"
                    self.recorded = list(self.pre_roll)
                    self.silence_sec = 0.0
                    self.record_sec = 0.0
                    continue

            else:  # RECORDING
                self.recorded.append(chunk)
                self.record_sec += self.cfg.chunk_sec

                cur_rms = _rms(chunk)
                if cur_rms < self.silence_threshold:
                    self.silence_sec += self.cfg.chunk_sec
                else:
                    self.silence_sec = 0.0

                if self.cfg.debug_rms:
                    print(
                        f"rms={cur_rms:.4f} thr={self.silence_threshold:.4f} "
                        f"silence={self.silence_sec:.1f}s rec={self.record_sec:.1f}s",
                        end="\r",
                        flush=True
                    )

                if self.silence_sec >= self.cfg.silence_stop_sec or self.record_sec >= self.cfg.max_record_sec:
                    if self.cfg.debug_rms:
                        print()
                    audio = np.concatenate(self.recorded).astype(np.float32)
                    self.state = "IDLE"
                    self.rec.Reset()
                    return "bob", audio