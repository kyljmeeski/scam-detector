from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

first = model.encode('К утру асфальт ещё будет хранить следы вечернего дождя').reshape(1, -1)
second = model.encode('Ночные капли дождя оставят на асфальте свою прохладу').reshape(1, -1)

print(model.similarity(first, second).item())


