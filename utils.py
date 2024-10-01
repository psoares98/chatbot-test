import os
import openai
from dotenv import load_dotenv
import faiss
import numpy as np

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

def initialize_vector_store(dimension=1536):
    index = faiss.IndexFlatL2(dimension)
    return index

def add_to_vector_store(index, embedding):
    embedding_np = np.array(embedding).astype('float32')
    index.add(np.expand_dims(embedding_np, axis=0))

def search_vector_store(index, query_embedding, top_k=5):
    query_np = np.array(query_embedding).astype('float32')
    distances, indices = index.search(np.expand_dims(query_np, axis=0), top_k)
    return indices[0], distances[0]
