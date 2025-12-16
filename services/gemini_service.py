import google.generativeai as genai
import os

# Configure Geminih
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def generate_rfm_insight(cluster_id, recency, frequency, monetary, total_customers, segment_label):
    """
    Generate actionable marketing insights using Gemini API based on RFM stats.
    """
    if not api_key:
        return {
            "strategy": "Gemini API Key missing.",
            "action": "Please configure GEMINI_API_KEY in .env",
            "email_template": "N/A"
        }

    try:
        # Dynamically find a supported model
        model_name = 'gemini-1.5-flash' # Default preference
        supported_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if supported_models:
            # Prefer flash if available, else take the first one
            if 'models/gemini-1.5-flash' in supported_models:
                model_name = 'gemini-1.5-flash'
            elif 'models/gemini-pro' in supported_models:
                model_name = 'gemini-pro'
            else:
                model_name = supported_models[0].replace('models/', '') # strip prefix if needed
                
        print(f"Using Gemini Model: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        As a Marketing Expert, analyze this customer segment:
        - Label: "{segment_label}"
        - Average Recency: {recency:.1f} days (Time since last purchase)
        - Average Frequency: {frequency:.1f} transactions
        - Average Monetary: ${monetary:,.2f}
        - Total Customers in Segment: {total_customers}

        Based on these specific stats, provide a targeted marketing strategy.
        
        Provide a JSON response with:
        1. "insight": 1-sentence behavioral analysis explaining WHY they have these specific stats.
        2. "strategy": 2-sentence marketing strategy tailored to improve these specific metrics.
        3. "action": 1 concrete action item (e.g., "Send email with subject...").
        
        Output valid JSON only.
        """

        response = model.generate_content(prompt)
        return response.text.replace("```json", "").replace("```", "").strip()

    except Exception as e:
        print(f"Gemini Error: {e}")
        return {
            "error": str(e)
        }
