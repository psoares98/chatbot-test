from flask import Flask, request, jsonify
import os
from utils import get_embedding, initialize_vector_store, add_to_vector_store, search_vector_store
import openai

app = Flask(__name__)

# Inicializar a base de dados vetorial
vector_store = initialize_vector_store()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')

    if not user_input:
        return jsonify({'error':'Mensagem não fornecida'})

    #gera embedding
    user_embedding = get_embedding(user_input)

    indices, distances = search_vector_store(vector_store, user_embedding)

    context = ""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um assistente útil."},
            {"role": "user", "content": user_input},
            # Adicionar contexto se disponível
        ]
    )

    reply = response['choices'][0]['message']['content']

    add_to_vector_store(vector_store, user_embedding)

    return jsonify({'response': reply})

if __name__ == '__main__':
    app.run(debug=True)