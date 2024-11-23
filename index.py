from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests
import json

app = Flask(__name__)

# Configure CORS with credentials support
CORS(app, 
     resources={r"/*": {
         "origins": ["http://localhost:8080"],
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type"],
         "supports_credentials": True
     }})

OLLAMA_API_URL = "http://192.168.1.68:11434/api/chat"
OLLAMA_MODEL = "llama3.1:latest"

def get_llama_response(prompt):
    """
    Get response from Ollama's Llama 3.1 model using the chat endpoint
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a friendly chat companion. Respond in a casual, friendly manner."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            return f"Error: {response.text}"
        
        return response.json()['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
        return f"Error: {str(e)}"

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """
    Endpoint to handle chat requests
    """
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:8080')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        response.headers.add('Access-Control-Allow-Credentials', 'true')  # Added this header
        return response

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    
    if 'message' not in data:
        return jsonify({"error": "Message field is required"}), 400
    
    user_message = data['message']
    response = get_llama_response(user_message)
    
    # Create response with proper CORS headers
    resp = make_response(jsonify({
        "user_message": user_message,
        "bot_response": response
    }))
    resp.headers['Access-Control-Allow-Origin'] = 'http://localhost:8080'
    resp.headers['Access-Control-Allow-Credentials'] = 'true'
    return resp

@app.after_request
def after_request(response):
    """
    Add CORS headers to all responses
    """
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)