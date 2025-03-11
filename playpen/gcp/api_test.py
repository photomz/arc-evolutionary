from google import genai
import os
from devtools import debug
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"], http_options={"api_version": "v1alpha"}
)


async def main() -> None:
    response = await client.aio.models.generate_content_stream(
        model="gemini-2.0-flash-thinking-exp",
        contents="Explain how RLHF works in simple terms.",
    )

    async for chunk in response:
        debug(chunk)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
