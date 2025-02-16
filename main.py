from sentence_transformers import SentenceTransformer

from Pair import Pair
from Phrase import Phrase
from Transcription import Transcription

MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# MODEL.save('paraphrase-multilingual-MiniLM-L12-v2')  # for the first time only -- to save locally


def main():
    signatures = [
        Phrase('Ваш счёт заблокирован, срочно подтвердите данные.', MODEL),
        Phrase('Ваш родственник попал в беду, нужно перевести деньги.', MODEL),
        Phrase('Оформляется подозрительный платёж, подтвердите код из SMS.', MODEL),
    ]
    sentences = Transcription('алло внучек это дедушка у меня тут небольшая неприятность вышла да вот ехал в магазин'
                  'поскользнулся упал телефон чуть не разбил хорошо что прохожие помогли даже довезли до дома'
                  'да не все нормально только карточку банковскую кажется потерял хотел тебя попросить не мог'
                  'бы ты мне немного скинуть на еду покая новую не сделаю').sentences()
    for sentence in sentences:
        for signature in signatures:
            similarity = Pair(Phrase(sentence, MODEL), signature, MODEL).similarity()
            print(sentence, signature, similarity)
        print("=======================================================================================================")


if __name__ == '__main__':
    main()
