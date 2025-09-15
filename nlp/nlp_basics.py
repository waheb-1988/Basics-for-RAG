
# Basic NLP concepts for RAG

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download resources
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 1. Tokenization
text = "Chatbots are amazing because they use Natural Language Processing."
tokens = word_tokenize(text)
print("Tokens:", tokens)

# 2. Stopwords removal
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
print("Filtered Tokens:", filtered_tokens)

# 3. Vectorization (Bag of Words)
docs = [
    "Chatbots use Natural Language Processing.",
    "RAG combines retrieval and generation.",
    "Embeddings capture semantic meaning."
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW Vectors:\n", X.toarray())

# 4. Similarity
similarity = cosine_similarity(X)
print("Document Similarity Matrix:\n", similarity)
