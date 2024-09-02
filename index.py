from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from app import run_conversation

app = Flask(__name__)
CORS(app) 

load_dotenv()

@app.route("/", methods=["GET"])
def hello():
    return "Hello World!"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        input_data = request.get_json()
        
        if not input_data or 'message' not in input_data:
            return jsonify({"error": "Invalid input. 'message' field is required."}), 400
        
        user_message = input_data['message']
        
        ai_response = run_conversation(user_message)
        
        return jsonify({"response": ai_response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)