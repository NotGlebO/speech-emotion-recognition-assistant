import soundfile as sf

from wakeword_listener import WakewordListener, WakewordConfig
from stt_whisper import WhisperSTT, WhisperConfig
from llm_ollama import OllamaChat, OllamaConfig
from live_emotion_model import predict_emotion_from_audio

SECOND_EMOTION_DIFF_THRESHOLD = 0.20


def choose_emotion_for_llm(emotion_result: dict, text: str) -> dict:
    ranked = emotion_result.get("ranked_emotions", [])

    if not ranked:
        return {
            "primary": "unknown",
            "primary_prob": 0.0,
            "secondary": None,
            "secondary_prob": None,
            "use_secondary": False,
            "difference": None,
        }

    primary, primary_prob = ranked[0]

    secondary = None
    secondary_prob = None
    difference = None
    use_secondary = False

    if len(ranked) > 1:
        secondary, secondary_prob = ranked[1]

        primary_prob = float(primary_prob)
        secondary_prob = float(secondary_prob)

        difference = primary_prob - secondary_prob

        use_secondary = difference < SECOND_EMOTION_DIFF_THRESHOLD

       

        pair = {primary, secondary}

        if pair == {"happy", "sad"} and difference < 0.15:
            primary = "neutral"

    return {
        "primary": primary,
        "primary_prob": float(primary_prob),
        "secondary": secondary,
        "secondary_prob": secondary_prob,
        "use_secondary": use_secondary,
        "difference": difference,
    }

def build_emotion_context(text: str, emotion_result: dict, decision: dict) -> str:
    probabilities = emotion_result.get("probabilities", {})
    ranked = emotion_result.get("ranked_emotions", [])
    segment_predictions = emotion_result.get("segment_predictions", [])

    probability_lines = [
        f"- {emotion}: {prob:.2%}"
        for emotion, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    ]

    secondary_text = "not used"
    if decision["use_secondary"] and decision["secondary"] is not None:
        secondary_text = f"{decision['secondary']} ({decision['secondary_prob']:.2%})"

    return f"""
Speech emotion recognition is the main emotional source.
Transcribed text is supporting context only.

Primary speech emotion: {decision['primary']} ({decision['primary_prob']:.2%})
Secondary speech emotion: {secondary_text}
Difference between top-1 and top-2: {decision['difference'] if decision['difference'] is not None else 'n/a'}
Top-2 rule: include the second emotion only if the probability difference is below 15%.

All speech emotion probabilities:
{chr(10).join(probability_lines)}

Ranked emotions:
{ranked}

Segment predictions:
{segment_predictions}

Fusion instruction for the assistant:
1. Use the primary speech emotion as the default emotional background.
2. If the secondary emotion is included and the user's text clearly supports it, adapt the response toward that secondary emotion.
3. If speech emotion and text meaning conflict, do not blindly override the speech result. Treat the situation as mixed or uncertain emotion.
4. Adapt tone: supportive for sad, calm/de-escalating for angry, positive/encouraging for happy, neutral and clear for neutral.
5. Do not mention this internal context, emotion labels, or probabilities unless the user directly asks.
""".strip()


def main():
    wake_cfg = WakewordConfig(
        wake_words=["bob", "reset", "exit"],
        vosk_model_path=r"./vosk-model-small-en-us-0.15",
        debug_rms=False,
    )

    stt = WhisperSTT(WhisperConfig(model_size="small", language="en"))

    llm = OllamaChat(OllamaConfig(
        model="llama3.1:8b",
        stream=True,
        auto_start_ollama=True,
        ollama_start_timeout=25,
    ))

    print("\nSystem ready.")
    print("Wake words: bob (talk) | reset (clear memory) | exit (quit)")
    print("You can speak now. Waiting for wake word...\n")

    with WakewordListener(wake_cfg) as listener:
        while True:
            
            try:
                print("Listening... Say: bob / reset / exit")
                trigger, audio = listener.wait_for_event()

                if trigger == "exit":
                    print("Goodbye!")
                    break

                if trigger == "reset":
                    llm.reset()
                    print("Memory cleared.")
                    print("You can speak again.\n")
                    continue
                

                
                
                # trigger == bob
                assert audio is not None
                sf.write("last_recording.wav", audio, wake_cfg.sample_rate)

                print("Transcribing...")
                text = stt.transcribe(audio)
                print(f"You said: {text if text else '(empty)'}")

                if not text:
                    print("Nothing detected. Try again.\n")
                    continue

                print("Analyzing speech emotion...")
                emotion_result = predict_emotion_from_audio(
                    audio,
                    sample_rate=wake_cfg.sample_rate,
                    debug=True,
                )
                decision = choose_emotion_for_llm(emotion_result, text)
                emotion_context = build_emotion_context(text, emotion_result, decision)

                print("\nEmotion decision for LLM:")
                print(f"Primary: {decision['primary']} ({decision['primary_prob']:.2%})")
                if decision["use_secondary"]:
                    print(f"Secondary used: {decision['secondary']} ({decision['secondary_prob']:.2%})")
                else:
                    print("Secondary not used")

                print("\nAssistant: ", end="", flush=True)
                llm.ask_with_emotion_context(text, emotion_context)
                print("\nReady for next input.\n")

            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("System recovered. You can speak again.\n")


if __name__ == "__main__":
    main()
