import os
import json
import uuid
from flask import Flask, render_template, request, jsonify, Response, session
from flask_cors import CORS
import openai
import anthropic
from google import genai
from google.genai import types
from dotenv import load_dotenv
from router import choose_model

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Check if API keys are loaded
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

# API Client Initializations
openai_client = openai.OpenAI(api_key=openai_key)
anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
genai_client = genai.Client(api_key=google_key)

# Store conversation histories (in a real app, use a database)
conversation_histories = {}

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and return streaming responses"""
    data = request.json
    prompt = data.get('message', '').strip()
    
    if not prompt:
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    # Get or create session ID
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    # Get conversation history
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    conversation_history = conversation_histories[session_id]
    
    # Use router to choose the best model
    router_result = choose_model(prompt)
    model_info = router_result["model_info"]
    instruction = router_result["instruction"]
    
    # Combine instruction with user prompt
    full_prompt = f"{instruction}\n\nUser: {prompt}"
    
    def generate_response():
        """Generator function for streaming responses"""
        try:
            # Send model selection info first
            model_selection_data = {
                'type': 'model_selection',
                'data': {
                    'provider': model_info['provider'],
                    'model': model_info['model'],
                    'category': router_result['category'],
                    'analysis': router_result['analysis']
                }
            }
            yield f"data: {json.dumps(model_selection_data)}\n\n"
            
            # Execute based on provider
            if model_info["provider"] == "openai":
                yield from stream_openai(full_prompt, conversation_history, model_info)
            elif model_info["provider"] == "claude":
                yield from stream_claude(full_prompt, conversation_history, model_info)
            elif model_info["provider"] == "gemini":
                yield from stream_gemini(full_prompt, conversation_history, model_info)
            
        except Exception as e:
            error_data = {
                'type': 'error',
                'data': {'message': str(e)}
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return Response(generate_response(), mimetype='text/plain')

def stream_openai(prompt, conversation_history, model_info):
    """Stream OpenAI responses"""
    try:
        # Build messages
        messages = []
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})
        
        api_params = {
            "model": model_info["model"],
            "messages": messages,
            "stream": True,
        }
        
        if "reasoning_effort" in model_info:
            api_params["reasoning_effort"] = model_info["reasoning_effort"]
        
        response = openai_client.chat.completions.create(**api_params)
        
        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                chunk_data = {
                    'type': 'content',
                    'data': {'chunk': content}
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
        
        # Save to history
        save_to_history(prompt, full_response, conversation_history)
        
        # Send completion signal
        completion_data = {'type': 'complete', 'data': {}}
        yield f"data: {json.dumps(completion_data)}\n\n"
        
    except Exception as e:
        error_data = {
            'type': 'error',
            'data': {'message': f"OpenAI Error: {str(e)}"}
        }
        yield f"data: {json.dumps(error_data)}\n\n"

def stream_claude(prompt, conversation_history, model_info):
    """Stream Claude responses"""
    try:
        # Build messages
        messages = []
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})
        
        api_params = {
            "model": model_info["model"],
            "max_tokens": 2048,
            "messages": messages,
        }
        
        if "temperature" in model_info:
            api_params["temperature"] = model_info["temperature"]
        
        full_response = ""
        with anthropic_client.messages.stream(**api_params) as stream:
            for text in stream.text_stream:
                full_response += text
                chunk_data = {
                    'type': 'content',
                    'data': {'chunk': text}
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
        
        # Save to history
        save_to_history(prompt, full_response, conversation_history)
        
        # Send completion signal
        completion_data = {'type': 'complete', 'data': {}}
        yield f"data: {json.dumps(completion_data)}\n\n"
        
    except Exception as e:
        error_data = {
            'type': 'error',
            'data': {'message': f"Anthropic Error: {str(e)}"}
        }
        yield f"data: {json.dumps(error_data)}\n\n"

def stream_gemini(prompt, conversation_history, model_info):
    """Stream Gemini responses"""
    try:
        # Build conversation
        contents = []
        for msg in conversation_history:
            contents.append(f"{msg['role']}: {msg['content']}")
        contents.append(f"user: {prompt}")
        full_conversation = "\n".join(contents)
        
        # Set up config
        config_params = {}
        if "temperature" in model_info:
            config_params["temperature"] = model_info["temperature"]
        
        thinking_config = None
        if "thinking_budget" in model_info:
            thinking_config = types.ThinkingConfig(thinking_budget=model_info["thinking_budget"])
        
        config = types.GenerateContentConfig(
            **config_params,
            thinking_config=thinking_config
        ) if config_params or thinking_config else None
        
        response = genai_client.models.generate_content(
            model=model_info["model"],
            contents=full_conversation,
            config=config
        )
        
        # Send response (Gemini doesn't stream in this API)
        chunk_data = {
            'type': 'content',
            'data': {'chunk': response.text}
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"
        
        # Save to history
        save_to_history(prompt, response.text, conversation_history)
        
        # Send completion signal
        completion_data = {'type': 'complete', 'data': {}}
        yield f"data: {json.dumps(completion_data)}\n\n"
        
    except Exception as e:
        error_data = {
            'type': 'error',
            'data': {'message': f"Google Error: {str(e)}"}
        }
        yield f"data: {json.dumps(error_data)}\n\n"

def save_to_history(user_prompt, ai_response, conversation_history):
    """Save the conversation exchange to history"""
    # Extract original user message from the full prompt
    user_message = user_prompt.split("\n\nUser: ")[-1] if "\n\nUser: " in user_prompt else user_prompt
    
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": ai_response})

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    session_id = session.get('session_id')
    if session_id and session_id in conversation_histories:
        conversation_histories[session_id] = []
    return jsonify({'status': 'success'})

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available model configurations"""
    with open('model_prompt.json', 'r') as f:
        config = json.load(f)
    return jsonify(config)

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration for editing"""
    try:
        with open('model_prompt.json', 'r') as f:
            config = json.load(f)
        return jsonify({'status': 'success', 'config': config})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration and save to file"""
    try:
        new_config = request.json
        
        # Validate the configuration structure
        if not isinstance(new_config, dict):
            return jsonify({'status': 'error', 'message': 'Invalid configuration format'}), 400
        
        # Basic validation for each category
        for category_name, category_data in new_config.items():
            if not isinstance(category_data, dict):
                return jsonify({'status': 'error', 'message': f'Invalid category format: {category_name}'}), 400
            
            required_fields = ['model', 'instructions']
            for field in required_fields:
                if field not in category_data:
                    return jsonify({'status': 'error', 'message': f'Missing {field} in category {category_name}'}), 400
            
            # Validate model structure
            model = category_data['model']
            if not isinstance(model, dict) or 'provider' not in model or 'model' not in model:
                return jsonify({'status': 'error', 'message': f'Invalid model format in category {category_name}'}), 400
            
            # Validate instructions
            instructions = category_data['instructions']
            if not isinstance(instructions, dict) or len(instructions) == 0:
                return jsonify({'status': 'error', 'message': f'Invalid instructions format in category {category_name}'}), 400
        
        # Backup current config
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f'model_prompt_backup_{timestamp}.json'
        
        with open('model_prompt.json', 'r') as f:
            backup_config = json.load(f)
        
        with open(backup_filename, 'w') as f:
            json.dump(backup_config, f, indent=2)
        
        # Save new config
        with open('model_prompt.json', 'w') as f:
            json.dump(new_config, f, indent=2)
        
        return jsonify({'status': 'success', 'message': f'Configuration updated successfully! Backup saved as {backup_filename}'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/config')
def config_page():
    """Serve the configuration interface"""
    return render_template('config.html')

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 