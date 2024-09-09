from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import json
import torch
import os
import spacy
import redis

# Initialize SpaCy and Redis clients
nlp = spacy.load("en_core_web_sm")
redis_client = redis.Redis(host='localhost', port=6379, db=0)

app = Flask(__name__, template_folder='templates')

# Define the paths to the model and tokenizer
model_path = "/maahr/home/ameen/Mistral-7B-Instruct-v0.2/"
tokenizer_path = "/maahr/home/ameen/Mistral-7B-Instruct-v0.2/"
embedder_path = "all-MiniLM-L6-v2"  # or any other suitable pre-trained model

# Load the tokenizer and model for conversational AI
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure the pad_token is properly set
except Exception as e:
    print(f"Error loading the model or tokenizer: {e}")
    exit()

# Initialize the text generation pipeline
try:
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Error initializing the text generation pipeline: {e}")
    exit()

# Load the JSON file
json_path = "/maahr/home/ameen/ahb_bot/Ai/ahb_bot/ahb.json"  # Updated path to the uploaded file
try:
    with open(json_path, 'r') as file:
        knowledge_base = json.load(file)
except Exception as e:
    print(f"Error loading the JSON file: {e}")
    exit()

# Initialize the sentence transformer model for semantic search
try:
    embedder = SentenceTransformer(embedder_path)
except Exception as e:
    print(f"Error loading the sentence transformer model: {e}")
    exit()

# Precompute embeddings for the knowledge base
kb_inputs = []
for intent in knowledge_base['intents']:
    if 'patterns' in intent:
        kb_inputs.extend(intent['patterns'])
    else:
        print(f"Warning: 'patterns' key missing in intent: {intent}")

if not kb_inputs:
    print("Error: No patterns found in the knowledge base.")
    exit()

kb_embeddings = embedder.encode(kb_inputs, convert_to_tensor=True)

# Define the path to store chat histories
chat_histories_dir = '/maahr/home/ameen/ahb_bot/Ai/chats/'

# Ensure the directory exists
os.makedirs(chat_histories_dir, exist_ok=True)

def get_chat_histories():
    chat_histories = {}
    for file_name in os.listdir(chat_histories_dir):
        if file_name.endswith('.json'):
            chat_id = file_name.replace('.json', '')
            with open(os.path.join(chat_histories_dir, file_name), 'r') as file:
                chat_histories[chat_id] = json.load(file)
    return chat_histories

def save_chat_history(chat_id, messages):
    with open(os.path.join(chat_histories_dir, f'{chat_id}.json'), 'w') as file:
        json.dump(messages, file)

def delete_chat_history(chat_id):
    os.remove(os.path.join(chat_histories_dir, f'{chat_id}.json'))

def search_knowledge_base(query, knowledge_base, kb_embeddings, threshold=0.75):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, kb_embeddings)[0]
    top_result = torch.argmax(cos_scores).item()
    
    if cos_scores[top_result] >= threshold:
        for intent in knowledge_base['intents']:
            if kb_inputs[top_result] in intent.get('patterns', []):
                return intent['responses'][0]
    return None

common_responses = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! What can I help you with?",
    "how are you": "I'm just a chatbot, but I'm here to help! How can I assist you today?",
    "what is your name": "I'm Gemini, your AI assistant. How can I help you today?"
}

def extract_key_info(response):
    doc = nlp(response)
    key_info = {
        "dates": [],
        "authors": [],
        "policies": []
    }
    
    for ent in doc.ents:
        if ent.label_ == "DATE":
            key_info["dates"].append(ent.text)
        elif ent.label_ == "PERSON":
            key_info["authors"].append(ent.text)
        elif ent.label_ in {"ORG", "GPE", "PRODUCT"}:
            key_info["policies"].append(ent.text)

    return key_info

def store_important_responses(response, chat_id):
    key_info = extract_key_info(response)
    
    if key_info["dates"] or key_info["authors"] or key_info["policies"]:
        summary = " and ".join(key_info["policies"]) if key_info["policies"] else "policy"
        details = f"This policy was created on {', '.join(key_info['dates'])} and the author was {', '.join(key_info['authors'])}." if key_info["dates"] and key_info["authors"] else response

        important_info = {
            'summary': summary,
            'details': details
        }
        
        redis_client.set(f'important_info:{response}', json.dumps(important_info))

def provide_contextual_response(last_bot_message, user_input):
    if not last_bot_message:
        return None

    important_info = redis_client.get(f'important_info:{last_bot_message}')
    if important_info:
        important_info = json.loads(important_info)

    if important_info and any(word in user_input.lower() for word in ["what", "when", "who", "explain"]):
        return f"In my previous message, I mentioned: {important_info['summary']}. {important_info['details']}"

    return None


@app.route('/')
def index():
    chat_histories = get_chat_histories()
    return render_template('index.html', chat_histories=chat_histories)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    chat_id = request.json.get('chat_id')
    chat_histories = get_chat_histories()
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    chat_history = chat_histories[chat_id]
    
    # Retrieve last bot message from chat history
    last_bot_message = next((msg['message'] for msg in reversed(chat_history) if msg['sender'] == 'Gemini'), None)
    
    # Generate response based on user input and last bot message
    response = generate_response(user_input, knowledge_base, kb_embeddings, last_bot_message)
    
    # Append user input and bot response to chat history
    chat_history.append({'sender': 'You', 'message': user_input})
    chat_history.append({'sender': 'Gemini', 'message': response})
    
    # Save updated chat history
    save_chat_history(chat_id, chat_history)
    
    # Store important information from the response
    store_important_responses(response, chat_id)
    
    return jsonify({'response': response})
    
def generate_response(user_input, knowledge_base, kb_embeddings, last_bot_message=None):
    # Search the knowledge base first
    kb_response = search_knowledge_base(user_input, knowledge_base, kb_embeddings)
    if kb_response:
        return kb_response

    # Handle common responses
    for key, value in common_responses.items():
        if key.lower() in user_input.lower():
            return value

    # Check if there's a contextually aware response based on last_bot_message
    contextual_response = provide_contextual_response(last_bot_message, user_input)
    if contextual_response:
        return contextual_response

    # If no response is found in the knowledge base or context, return default message
    return "I’m sorry, I don’t have information on that."



@app.route('/get_chats', methods=['GET'])
def get_chats():
    chat_histories = get_chat_histories()
    return jsonify(chat_histories)

@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    chat_id = request.json.get('chat_id')
    delete_chat_history(chat_id)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
