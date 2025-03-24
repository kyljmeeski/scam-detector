# from sentence_transformers import SentenceTransformer
#
# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
#
# first = model.encode('К утру асфальт ещё будет хранить следы вечернего дождя').reshape(1, -1)
# second = model.encode('Ночные капли дождя оставят на асфальте свою прохладу').reshape(1, -1)
#
# print(model.similarity(first, second).item())
#
#


from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

# Load a pre-trained model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')


def split_text_by_semantics(text, chunk_size=50, num_clusters=5):
    # Step 1: Split text into roughly equal-sized chunks (not by punctuation)
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Step 2: Generate embeddings for each chunk
    embeddings = model.encode(chunks)

    # Step 3: Cluster embeddings based on semantic similarity
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Step 4: Group chunks by cluster label to form logical units
    clustered_phrases = {}
    for label, chunk in zip(labels, chunks):
        if label not in clustered_phrases:
            clustered_phrases[label] = []
        clustered_phrases[label].append(chunk)

    # Combine chunks in each cluster to form final phrases
    phrases = [' '.join(clustered_phrases[label]) for label in clustered_phrases]
    return phrases


# Example usage
text = "This is a sample text where we need to extract semantically meaningful phrases without relying on punctuation marks"
phrases = split_text_by_semantics(text, chunk_size=10, num_clusters=3)
print(phrases)