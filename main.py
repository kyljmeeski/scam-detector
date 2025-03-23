import json
import subprocess
import wave

from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
from typing import Dict
# from vosk import KaldiRecognizer, Model, SetLogLevel

import speech_recognition as sr

from Pair import Pair
from Phrase import Phrase
from Transcription import Transcription

SEMANTIC_ANALYZER = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# MODEL.save('paraphrase-multilingual-MiniLM-L12-v2')  # for the first time only -- to save locally


def transcribe(audio_path: str) -> str:
    with sr.AudioFile(audio_path) as source:
        recognizer = sr.Recognizer()
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    text = recognizer.recognize_google(audio, language="ru-RU")
    return text.lower().strip()


def analyze(audio_path: str) -> Dict[str, str]:
    signatures = [
        Phrase('Ваш счёт заблокирован, срочно подтвердите данные.', SEMANTIC_ANALYZER),
        Phrase('Ваш родственник попал в беду, нужно перевести деньги.', SEMANTIC_ANALYZER),
        Phrase('Оформляется подозрительный платёж, подтвердите код из SMS.', SEMANTIC_ANALYZER),
    ]

    transcription = Transcription(transcribe(audio_path))
    for sentence in transcription.sentences():
        print(sentence)
        for signature in signatures:
            similarity = Pair(Phrase(sentence, SEMANTIC_ANALYZER), signature, SEMANTIC_ANALYZER).similarity()
            print("\t", signature, ":", similarity)
    return dict()


def main():
    data = analyze("resources/audio/granddad.wav")
    print(data)
    # signatures = [
    #     Phrase('Ваш счёт заблокирован, срочно подтвердите данные.', SEMANTIC_ANALYZER),
    #     Phrase('Ваш родственник попал в беду, нужно перевести деньги.', SEMANTIC_ANALYZER),
    #     Phrase('Оформляется подозрительный платёж, подтвердите код из SMS.', SEMANTIC_ANALYZER),
    # ]
    # sentences = Transcription('алло внучек это дедушка у меня тут небольшая неприятность вышла да вот ехал в магазин'
    #               'поскользнулся упал телефон чуть не разбил хорошо что прохожие помогли даже довезли до дома'
    #               'да не все нормально только карточку банковскую кажется потерял хотел тебя попросить не мог'
    #               'бы ты мне немного скинуть на еду покая новую не сделаю').sentences()
    # for sentence in sentences:
    #     for signature in signatures:
    #         similarity = Pair(Phrase(sentence, SEMANTIC_ANALYZER), signature, SEMANTIC_ANALYZER).similarity()
    #         print(sentence, signature, similarity)
    #     print("=======================================================================================================")


if __name__ == '__main__':
    main()
