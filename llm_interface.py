import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Set your Gemini API key as an environment variable 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
    raise ValueError("Please set your GEMINI_API_KEY environment variable")

genai.configure(api_key=GEMINI_API_KEY)

def generate_answer(context, question, max_tokens=512, temperature=0.2):
    """
    Uses Gemini Flash 1.5 to answer a question based on provided context.
    
    Args:
        context (str): Relevant text chunks concatenated as context.
        question (str): User's question.
        max_tokens (int): Maximum tokens in response.
        temperature (float): Response randomness.
    
    Returns:
        answer (str): Gemini-generated answer.
    """
    prompt = (
        "You are an expert AI assistant. Use ONLY the context below to answer the question.\n"
        "If you don't know the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature
        }
    )
    if hasattr(response, "text"):
        return response.text.strip()
    elif hasattr(response, "candidates") and response.candidates:
        return response.candidates[0].content.parts[0].text.strip()
    else:
        return "[No answer generated]"
