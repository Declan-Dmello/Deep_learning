import whisper


def transcribe_audio(audio_path: str, model_size: str = "base"):

    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    print(f"Transcribing audio file: {audio_path}")
    result = model.transcribe(audio_path)

    print("\nTranscription:")
    return result["text"]


if __name__ == "__main__":
    # 📂 Change this path to your audio file
    audio_file_path = r"C:\Users\decla\PycharmProjects\pythonProject2\data\harvard\harvard.wav"

    # 🧠 Choose model size: "tiny", "base", "small", "medium", "large"
    model_choice = "base"

    text = transcribe_audio(audio_file_path, model_choice)
    print(text)
