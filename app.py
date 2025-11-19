from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import traceback
import os
import re

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global model, tokenizer
    
    print("Loading fine-tuned model...")
    MODEL_PATH = "./phi3-cricket-finetuned-v2"
    BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=bnb_config,
        attn_implementation="eager"
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    
    print(f"âœ“ Model loaded on {device}")

def is_cricket_related(question):
    """Check if question is cricket-related"""
    cricket_keywords = [
        'cricket', 'bat', 'ball', 'wicket', 'over', 'run', 'boundary',
        'bowler', 'batsman', 'fielder', 'stump', 'lbw', 'catch', 'innings',
        'test', 'odi', 't20', 'player', 'team', 'match', 'pitch', 'six', 'four'
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in cricket_keywords)


def generate_response(question, max_length=150, temperature=0.3):
    try:
        # Handle greetings and casual conversation with WORD BOUNDARIES
        question_lower = question.lower().strip()
        
        # Greetings - use word boundaries to avoid false matches
        greeting_pattern = r'\b(hi|hello|hey|greetings)\b'
        if re.search(greeting_pattern, question_lower):
            # But exclude if it's clearly a cricket question
            if not is_cricket_related(question):
                return "Hi! I am your cricket assistant. Ask me anything about cricket rules, terms, players, or matches!"
        
        # Good morning/afternoon/evening
        if re.search(r'\bgood (morning|afternoon|evening)\b', question_lower):
            return "Good day! I'm here to help you with cricket questions. What would you like to know about cricket?"
        
        # How are you - use word boundaries
        if re.search(r'\bhow are you\b|\bhow do you do\b|\bwhats up\b|\bwhat\'s up\b', question_lower):
            return "I'm doing great, thank you! I'm ready to answer your cricket questions. What would you like to know?"
        
        # Thanks
        if re.search(r'\bthank', question_lower) and not is_cricket_related(question):
            return "You're welcome! Feel free to ask more cricket questions anytime!"
        
        # Bye
        if re.search(r'\bbye\b|\bgoodbye\b|\bsee you\b', question_lower):
            return "Goodbye! Come back if you have more cricket questions. Have a great day!"
        
        # Check if cricket-related
        if not is_cricket_related(question):
            return "I'm sorry, but I'm specialized in cricket only. I can answer questions about cricket rules, terms, players, matches, and formats. Please ask me something related to cricket!"
        
        # === FIXED: Remove 'nan' from prompt ===
        prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,  # Slightly higher for more diverse responses
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
                repetition_penalty=1.2,  # Increased to reduce repetition
                num_beams=1
            )
        
        # Extract only generated tokens
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up response
        response = response.strip()
        
        # Remove end tokens if present
        if "<|end|>" in response:
            response = response.split("<|end|>")[0].strip()
        
        # Remove common artifacts
        cleanup_phrases = [
            "You are a cricket expert.",
            "Answer the following question about cricket concisely.",
            "<|user|>",
            "<|assistant|>",
            "nan",
            "<|end|>"
        ]
        
        for phrase in cleanup_phrases:
            response = response.replace(phrase, "").strip()
        
        # Remove question if it leaked into response
        if question.lower() in response.lower():
            parts = response.split(question, 1)
            if len(parts) > 1 and len(parts[1].strip()) > 0:
                response = parts[1].strip()
        
        # Remove leading punctuation
        response = response.lstrip(":. ")
        
        # Validate response quality
        if not response or len(response) < 10:
            response = "I understand your question is about cricket, but I need more context to answer accurately. Could you please rephrase or provide more details?"
        
        return response
    
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        print(traceback.format_exc())
        return "I apologize, but I encountered an error processing your question. Please try asking again."


def is_cricket_related(question):
    """Check if question is cricket-related - EXPANDED"""
    cricket_keywords = [
        'cricket', 'bat', 'ball', 'wicket', 'over', 'run', 'boundary',
        'bowler', 'batsman', 'fielder', 'stump', 'lbw', 'catch', 'innings',
        'test', 'odi', 't20', 'player', 'team', 'match', 'pitch', 'six', 'four',
        # Player names
        'kohli', 'dhoni', 'tendulkar', 'rohit', 'bumrah', 'pujara', 'dravid',
        'ganguly', 'sehwag', 'kapil', 'sachin', 'virat', 'ms', 'sharma',
        'harbhajan', 'kumble', 'raina', 'yuvraj', 'pathan', 'zaheer', 'ishant', 
        'jadeja', 'rahul', 'ishan', 'subhman', 'pant', 'gill', 'smriti', 'mandhana', 'deepa', 'shafali', 'harmanpreet',
        'jhulan', 'sundar', 'shami', 'chahal', 'kuldeep',
        # Venues
        'wankhede', 'eden', 'kotla', 'chinnaswamy', 'chepauk',
        # Awards and records
        'century', 'double century', 'triple century', 'hat-trick', 'world cup',
        'ipl', 'champions trophy', 'ashes', 'series'
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in cricket_keywords)


@app.route('/')
def home():
    """Serve the HTML UI"""
    if os.path.exists('index.html'):
        return send_file('index.html')
    else:
        return jsonify({
            "status": "online",
            "message": "Cricket Expert Chatbot API",
            "model": "Phi-3 Fine-tuned on Cricket Dataset",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "capabilities": [
                "Cricket rules and terminology",
                "General cricket knowledge",
                "Casual greetings and conversation",
                "Politely declines non-cricket topics"
            ],
            "note": "UI file (index.html) not found. Access API via /ask endpoint."
        })

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "Please provide a 'question' field"}), 400
        
        question = data['question']
        max_length = data.get('max_length', 150)
        temperature = data.get('temperature', 0.7)
        
        print(f"\n[Request] Question: {question}")
        
        answer = generate_response(question, max_length, temperature)
        
        print(f"[Response] Answer: {answer}\n")
        
        return jsonify({
            "question": question,
            "answer": answer,
            "model": "Phi-3-cricket-finetuned"
        })
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n[ERROR] {error_msg}")
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "device": str(device)
    })

if __name__ == '__main__':
    load_model()
    print("\n" + "=" * 50)
    print("Starting Flask API on http://localhost:5000")
    print("Open http://localhost:5000 in your browser!")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)