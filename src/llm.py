import ollama
from src.config import get_settings

SETTINGS = get_settings()

class LLM:
    async def generateResponse(prompt, file):
        
        # Use ollama.generate for vision tasks
        response = ollama.generate(
            model=SETTINGS.llm,
            prompt=prompt,
            images=[file]
        )
        return response