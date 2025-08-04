import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_llm_recommendation(product_count, timestamp):
    prompt = f"""
You are a smart inventory advisor helping small business owners with real-time decisions.

A detection system has just counted {product_count} products at {timestamp}.
Based on this, give 3 clear, actionable and brief recommendations to the user.

Include:
1. Whether they need to reorder or not.
2. If they should adjust the minimum stock threshold.
3. A quick visual strategy if there is too much inventory.

Use a concise and professional tone.
"""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"
