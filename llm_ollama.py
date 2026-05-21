from dataclasses import dataclass
from typing import List, Dict, Optional
import os
import json
import time
import socket
import shutil
import subprocess
import requests


@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434/api/chat"
    health_url: str = "http://localhost:11434/api/tags"
    model: str = "llama3.1:8b"
    system_prompt: str = (
        "Your name is Bob. "
        "You must always identify yourself as Bob and never use any other name. "
        "You are a helpful local voice assistant. "
        "Always respond in English. "
        "Be clear, concise, and helpful. "
        "You may receive speech emotion recognition data together with the user's text. "
        "Use the detected speech emotion as the main emotional signal, but also consider the meaning of the user's text. "
        "If the top two detected emotions are close, consider both and let the user's text help resolve the ambiguity. "
        "Adapt your tone to the user's emotional state, but do not mention emotion probabilities unless the user asks. "
    )
    timeout_connect: int = 5
    timeout_read: int = 300
    stream: bool = True

    auto_start_ollama: bool = True
    ollama_start_timeout: int = 25
    ollama_exe: Optional[str] = None


class OllamaChat:
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.cfg.system_prompt}
        ]
        self.session = requests.Session()
        self._ollama_process: Optional[subprocess.Popen] = None

    def reset(self):
        self.messages = [{"role": "system", "content": self.cfg.system_prompt}]

    def _is_port_open(self, host: str = "127.0.0.1", port: int = 11434, timeout: float = 1.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _find_ollama_exe(self) -> Optional[str]:
        if self.cfg.ollama_exe and os.path.exists(self.cfg.ollama_exe):
            return self.cfg.ollama_exe

        exe_from_path = shutil.which("ollama")
        if exe_from_path:
            return exe_from_path

        candidates = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\Ollama.exe"),
            os.path.expandvars(r"%ProgramFiles%\Ollama\Ollama.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Ollama\Ollama.exe"),
        ]

        for path in candidates:
            if path and os.path.exists(path):
                return path

        return None

    def _start_ollama(self) -> None:
        exe = self._find_ollama_exe()
        if not exe:
            raise RuntimeError(
                "Ollama was not found. Install Ollama, add it to PATH, "
                "or provide the full executable path in OllamaConfig(ollama_exe=...)."
            )

        print("Ollama is not running. Attempting to start it...")

        try:
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW

            self._ollama_process = subprocess.Popen(
                [exe, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except Exception:
            try:
                if os.name == "nt":
                    os.startfile(exe)  # type: ignore[attr-defined]
                else:
                    self._ollama_process = subprocess.Popen(
                        [exe],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            except Exception as e:
                raise RuntimeError(f"Failed to start Ollama: {e}") from e

        start_time = time.time()
        while time.time() - start_time < self.cfg.ollama_start_timeout:
            if self._is_port_open():
                print("Ollama is running and ready.")
                return
            time.sleep(1.0)

        raise RuntimeError(
            "Ollama was started, but the API did not become available in time. "
            "Check whether the Ollama server and model are starting correctly."
        )

    def ensure_server_running(self) -> None:
        if self._is_port_open():
            return

        if not self.cfg.auto_start_ollama:
            raise RuntimeError("Ollama server is not running.")

        self._start_ollama()

    def ask(self, user_text: str) -> str:
        self.ensure_server_running()

        self.messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.cfg.model,
            "messages": self.messages,
            "stream": self.cfg.stream,
        }

        if not self.cfg.stream:
            r = self.session.post(
                self.cfg.url,
                json=payload,
                timeout=(self.cfg.timeout_connect, self.cfg.timeout_read),
            )
            r.raise_for_status()
            answer = r.json()["message"]["content"].strip()
            self.messages.append({"role": "assistant", "content": answer})
            return answer

        r = self.session.post(
            self.cfg.url,
            json=payload,
            stream=True,
            timeout=(self.cfg.timeout_connect, self.cfg.timeout_read),
        )
        r.raise_for_status()

        full = []
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception:
                continue

            if "message" in data and "content" in data["message"]:
                chunk = data["message"]["content"]
                if chunk:
                    print(chunk, end="", flush=True)
                    full.append(chunk)

            if data.get("done", False):
                break

        print()
        answer = "".join(full).strip()
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def ask_with_emotion_context(self, user_text: str, emotion_context: str) -> str:
        """Send user text together with hidden emotion context to the LLM.

        The assistant sees both the transcribed text and the speech emotion
        recognition output, but it should answer naturally to the user.
        """
        prompt = (
            "User message:\n"
            f"{user_text}\n\n"
            "Internal emotion context from the speech emotion recognition module:\n"
            f"{emotion_context}\n\n"
            "Response instruction:\n"
            "Answer the user's message naturally. Use the emotion context only to adapt tone and empathy. "
            "Do not expose the internal analysis or probabilities unless the user explicitly asks for them."
        )
        return self.ask(prompt)

