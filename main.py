import json
import os

from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
from typing import Dict, List

import speech_recognition as sr

from Pair import Pair
from Phrase import Phrase
from Transcription import Transcription


SIMILARITY_THRESHOLD = 0.6 # настолько предложение должна быть похожа на фразу
SIGNATURE_THRESHOLD = 2 # вот на столько фраз должно быть похоже предложение
MARKERS_THRESHOLD = 3 # вот столько подозрительных предложений должно быть

SEMANTIC_ANALYZER = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# SEMANTIC_ANALYZER.save('paraphrase-multilingual-MiniLM-L12-v2')  # for the first time only -- to save locally


# def convert_audio(audio_path: str) -> str:
#     filename, extension = os.path.basename(audio_path).split(".")
#     base_path = os.path.join("resources", "audio")
#     if extension == "wav":
#         audio = AudioSegment.from_wav(audio_path)
#         audio = AudioSegment.fromm
#         audio.export(os.path.join(base_path, filename + ".wav"), format="wav")
#     if extension == "amr":
#         audio = AudioSegment.from_file(audio_path, format="amr")
#         audio.export(os.path.join(base_path, filename + ".wav"), format="wav")
#     return os.path.join(base_path, filename + ".wav")


def transcribe(audio_path: str) -> str:
    if audio_path.endswith(".mp3"):
        audio = AudioSegment.from_mp3(audio_path)
        audio_path = "resources/temp.wav"
        audio.export(audio_path, format="wav")

    with sr.AudioFile(audio_path) as source:
        recognizer = sr.Recognizer()
        # recognizer.energy_threshold = 4000
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    text = recognizer.recognize_google(audio, language="ru-RU")
    # print(text)
    return text.lower().strip()

def load_signatures() -> Dict[str, List[str]]:
    signatures = dict()
    with open("resources/signatures.json", encoding="utf-8") as file:
        data = json.load(file)
    for signature, sentences in data.items():
        signatures[signature] = sentences
    return signatures


def test() -> str:
    with open("resources/test.txt", encoding="utf-8") as file:
        return file.read()


def analyze(audio_path: str) -> Dict[str, str]:
    markers = dict()
    signatures = load_signatures()
    transcription = transcribe(audio_path)
    # transcription = test()
    chat_sentences = Transcription(transcription).sentences()
    for chat_sentence in chat_sentences:
        for signature, signature_sentences, in signatures.items():
            signature_markers = dict()
            for signature_sentence in signature_sentences:
                similarity = Pair(Phrase(chat_sentence, SEMANTIC_ANALYZER),
                                  Phrase(signature_sentence, SEMANTIC_ANALYZER),
                                  SEMANTIC_ANALYZER
                ).similarity()
                print(chat_sentence)
                print("\t", signature_sentence, similarity)
                if similarity >= SIMILARITY_THRESHOLD:
                    signature_markers[chat_sentence] = signature
            if len(signature_markers) > SIGNATURE_THRESHOLD:
                markers.update(signature_markers)
    if len(markers) >= MARKERS_THRESHOLD:
        markers["suspicious"] = "true"
    else:
        markers["suspicious"] = "false"
    return markers


def main():
    data = analyze("audio.mp3")
    # print()
    print(data)


if __name__ == '__main__':
    main()
