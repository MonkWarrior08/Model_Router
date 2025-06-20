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

def categorize_task_and_instruction(prompt: str) -> dict:
    """
    Categorizes the user request into a task category and selects the appropriate instruction.
    """
    system_prompt = f"""
    You are an expert task classifier. Your job is to analyze a user's prompt and
    classify it into one of the following task categories.
    Respond with ONLY the category name and nothing else.

    Available categories are: {', '.join(TASK_CATEGORIES)}

    Examples:
    - "Write a poem about love" → creative_writing
    - "Build a web application" → professional_coding
    - "Solve this math equation" → advanced_reasoning
    - "Quick calculation" → fast_reasoning
    - "Explain quantum physics" → general_qa
    - "Summarize this article" → fast_simple_task
    - "Let's have a casual chat" → conversational_ai
    """

    try:
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
        
        # Validate the category
        if category in BENCHMARKS:
            return {"category": category, "valid": True}
        else:
            print(f"⚠️ Router: Could not classify category '{category}'. Defaulting to 'general_qa'.")
            return {"category": "general_qa", "valid": False}
            
    except Exception as e:
        print(f"🚨 Router Error: {e}. Defaulting to 'general_qa'.")
        return {"category": "general_qa", "valid": False}

def select_instruction_for_category(category: str, prompt: str) -> str:
    """
    Selects the appropriate instruction for the given category based on prompt content.
    """
    category_info = BENCHMARKS[category]
    instructions = category_info["instructions"]
    
    prompt_lower = prompt.lower()
    
    # Select instruction based on category and prompt content
    if category == "creative_writing":
        if any(word in prompt_lower for word in ["poem", "poetry", "verse", "rhyme", "sonnet", "haiku"]):
            return "poetry"
        elif any(word in prompt_lower for word in ["story", "narrative", "tale", "fiction", "novel", "plot"]):
            return "storytelling"
        elif any(word in prompt_lower for word in ["copy", "marketing", "advertisement", "persuasive", "sales"]):
            return "copywriting"
        elif any(word in prompt_lower for word in ["script", "screenplay", "dialogue", "scene", "film", "theater"]):
            return "scriptwriting"
        else:
            return "default"
    
    elif category == "professional_coding":
        if any(word in prompt_lower for word in ["web", "html", "css", "javascript", "frontend", "website"]):
            return "web_development"
        elif any(word in prompt_lower for word in ["data", "machine learning", "ml", "ai", "statistics", "analysis"]):
            return "data_science"
        elif any(word in prompt_lower for word in ["architecture", "system", "microservices", "cloud", "infrastructure"]):
            return "system_architecture"
        else:
            return "default"
    
    elif category == "advanced_reasoning":
        if any(word in prompt_lower for word in ["math", "mathematical", "equation", "calculation", "proof"]):
            return "mathematical"
        elif any(word in prompt_lower for word in ["science", "scientific", "experiment", "hypothesis", "research"]):
            return "scientific"
        elif any(word in prompt_lower for word in ["philosophy", "philosophical", "ethics", "logic", "argument"]):
            return "philosophical"
        else:
            return "default"
    
    elif category == "fast_reasoning":
        if any(word in prompt_lower for word in ["math", "calculation", "number", "equation"]):
            return "quick_math"
        elif any(word in prompt_lower for word in ["puzzle", "riddle", "logic", "brain teaser"]):
            return "logic_puzzles"
        elif any(word in prompt_lower for word in ["decision", "choose", "option", "recommend"]):
            return "decision_making"
        else:
            return "default"
    
    elif category == "general_qa":
        if any(word in prompt_lower for word in ["explain", "teach", "learn", "education", "tutorial"]):
            return "educational"
        elif any(word in prompt_lower for word in ["research", "find", "information", "study", "analysis"]):
            return "research"
        elif any(word in prompt_lower for word in ["advice", "help", "solution", "problem", "fix"]):
            return "practical_advice"
        else:
            return "default"
    
    elif category == "fast_simple_task":
        if any(word in prompt_lower for word in ["summarize", "summary", "condense", "brief"]):
            return "summarization"
        elif any(word in prompt_lower for word in ["translate", "language", "translation"]):
            return "translation"
        elif any(word in prompt_lower for word in ["format", "organize", "structure", "layout"]):
            return "formatting"
        else:
            return "default"
    
    elif category == "conversational_ai":
        if any(word in prompt_lower for word in ["chat", "casual", "small talk", "friendly"]):
            return "casual_chat"
        elif any(word in prompt_lower for word in ["support", "emotional", "therapeutic", "comfort"]):
            return "therapeutic"
        elif any(word in prompt_lower for word in ["entertain", "fun", "humor", "joke", "story"]):
            return "entertainment"
        else:
            return "default"
    
    else:
        return "default"

def choose_model(prompt: str) -> dict:
    """
    First categorizes the user request into a task category and selects an instruction,
    then selects the best model for that category.
    """
    print("🧠 Router: Analyzing prompt to select category and instruction...")

    # 1. Categorize the task
    categorization = categorize_task_and_instruction(prompt)
    category = categorization["category"]
    
    # 2. Select the appropriate instruction for this category
    instruction_type = select_instruction_for_category(category, prompt)
    
    # 3. Get the model info and instruction
    model_info = BENCHMARKS[category]
    selected_instruction = model_info["instructions"][instruction_type]
    
    print(f"✅ Router: Selected category: '{category}'")
    print(f"📝 Router: Selected instruction: '{instruction_type}'")
    
    return {
        "model_info": model_info,
        "instruction": selected_instruction,
        "category": category,
        "instruction_subtype": instruction_type
    }