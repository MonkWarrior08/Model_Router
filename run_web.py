#!/usr/bin/env python3
"""
Simple startup script for the AI Model Router Web UI
"""

import os
import sys
from dotenv import load_dotenv

def check_environment():
    """Check if required environment variables are set"""
    load_dotenv()
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for o4-mini model',
        'ANTHROPIC_API_KEY': 'Anthropic API key for Claude models', 
        'GOOGLE_API_KEY': 'Google API key for Gemini models'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  {var}: {description}")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        print("\n".join(missing_vars))
        print("\nPlease create a .env file with these variables, or set them in your environment.")
        print("See env_template.txt for the template.")
        return False
    
    print("✅ All required environment variables are set!")
    return True

def main():
    print("🤖 AI Model Router - Web UI")
    print("=" * 40)
    
    if not check_environment():
        sys.exit(1)
    
    print("\n🚀 Starting web server...")
    print("📱 Open your browser to: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop the server")
    print("\n" + "=" * 40 + "\n")
    
    # Import and run the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Thanks for using AI Model Router!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("Make sure you've installed the requirements: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 