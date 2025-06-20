import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# We use a cheap and fast model for the routing itself.
# Gemini Flash 2.0 Flash Lite is a good choice for fast classification tasks.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load our benchmark data
with open('benchmarks.json', 'r') as f:
    BENCHMARKS = json.load(f)

TASK_CATEGORIES = list(BENCHMARKS.keys())

def choose_model(prompt: str) -> dict:
    """
    Uses an AI to classify the user's prompt and then selects the best model
    from the benchmark database.
    """
    print("🧠 Router: Analyzing prompt to select the best model...")

    try:
        # This is a specialized prompt telling the AI how to behave.
        system_prompt = f"""
        You are an expert task router. Your job is to analyze a user's prompt and
        classify it into one of the following predefined categories.
        Respond with ONLY the category name and nothing else.

        Available categories are: {', '.join(TASK_CATEGORIES)}
        """

        # Create the model instance
        model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')
        
        # Combine system and user prompt for Gemini
        full_prompt = f"{system_prompt}\n\nUser prompt: {prompt}"
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=50
            )
        )

        category = response.text.strip()

        if category in BENCHMARKS:
            chosen_model_info = BENCHMARKS[category]
            print(f"✅ Router: Task classified as '{category}'. Routing to {chosen_model_info['provider']} ({chosen_model_info['model']}) with temperature {chosen_model_info['temperature']}.")
            return chosen_model_info
        else:
            print(f"⚠️ Router: Could not classify. Defaulting to 'general_qa'.")
            return BENCHMARKS["general_qa"]

    except Exception as e:
        print(f"🚨 Router Error: {e}. Defaulting to 'general_qa'.")
        return BENCHMARKS["general_qa"]