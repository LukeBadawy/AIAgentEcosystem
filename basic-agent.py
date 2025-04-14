import os
import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dotenv import load_dotenv

# --- 1. Define the Structured Output using Pydantic ---
# This model remains the same.
class TopicSummary(BaseModel):
    """Represents a brief summary and keywords for a given topic."""
    summary: str = Field(description="A concise 1-2 sentence summary of the topic.")
    keywords: list[str] = Field(description="A list of 3-5 relevant keywords associated with the topic.")

# --- 2. Load Environment Variables ---
# This remains the same.
load_dotenv()

# Check if the Google API key is loaded
if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY not found. Please set it in the .env file.")
    exit()

# --- 3. Configure and Instantiate the PydanticAI Agent ---
# <<< CHANGED: Updated the model identifier to use Gemini 2.5 Pro Preview
# Note: Model availability might depend on your Google account's access to preview features.
summarizer_agent = Agent(
    'google-gla:gemini-2.5-pro-preview-03-25', # Use Gemini 2.5 Pro Preview model
    result_type=TopicSummary   # Still expect the same Pydantic model output
)

# --- 4. Define the Main Execution Logic ---
# This function remains the same.
async def get_topic_summary(topic: str):
    """
    Uses the PydanticAI agent to get a structured summary for a topic.
    """
    print(f"\n--- Requesting summary for topic: '{topic}' using Gemini 2.5 Pro ---") # Added model name here
    try:
        result = await summarizer_agent.run(
            f"Provide a very brief summary and some keywords for the topic: {topic}"
        )

        print("\n--- Structured Result Received ---")
        print(f"Result: {result}")
        
        # Access the result data correctly
        if hasattr(result, 'data'):
            structured_result = result.data
            print(f"Summary: {structured_result.summary}")
            print(f"Keywords: {', '.join(structured_result.keywords)}")
            return structured_result
        else:
            print(f"Result does not have expected structure. Raw result: {result}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        # You might get specific errors if the model isn't accessible
        # or if there are temporary issues with preview models.
        return None

# --- 5. Run the Agent ---
# This function remains the same.
async def main():
    await get_topic_summary("Quantum Computing")
    await get_topic_summary("Photosynthesis")
    await get_topic_summary("Large Language Models")

if __name__ == "__main__":
    # Use asyncio.run() to execute the async main function
    asyncio.run(main())