import asyncio
import base64
import io
import os
import re
import typing as T

import google.generativeai as genai
import PIL.Image
from anthropic import AsyncAnthropic, RateLimitError
from devtools import debug
from openai import AsyncAzureOpenAI, AsyncOpenAI

from src import logfire
from src.models import Model, ModelUsage

nvidia_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"]
)
groq_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"]
)
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]
)
azure_client = AsyncAzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-10-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def text_only_messages(messages: list[dict[str, T.Any]]) -> list[dict[str, T.Any]]:
    new_messages = []
    for message in messages:
        content_strs: list[str] = []
        if isinstance(message["content"], str):
            content_strs.append(message["content"])
        else:
            for content in message["content"]:
                if content["type"] == "text":
                    content_strs.append(content["text"])
        if content_strs:
            new_messages.append(
                {
                    "role": message["role"],
                    "content": "\n".join(content_strs),
                }
            )
    return new_messages


async def get_next_message(
    *, messages: list[dict[str, T.Any]], model: Model, temperature: float
) -> tuple[str, ModelUsage]:
    if int(os.environ.get("NO_WIFI", 0)) == 1:
        return "[[1, 2, 3], [4, 5, 6]]", ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            input_tokens=0,
            output_tokens=0,
        )
    if model in [Model.claude_3_5_sonnet, Model.claude_3_5_haiku]:
        anthropic_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        if messages[0]["role"] == "system":
            system_messages = messages[0]["content"]
            messages = messages[1:]
        else:
            system_messages = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                for content in message["content"]:
                    if content["type"] == "image_url":
                        content["type"] = "image"
                        content["source"] = {
                            "data": content["image_url"]["url"].replace(
                                "data:image/png;base64,", ""
                            ),
                            "media_type": "image/png",
                            "type": "base64",
                        }
                        del content["image_url"]

        retry_count = 0
        max_retries = 12
        while True:
            try:
                message = await anthropic_client.beta.prompt_caching.messages.create(
                    system=system_messages,
                    temperature=temperature,
                    max_tokens=8_192,
                    messages=messages,
                    model=model.value,
                    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
                    timeout=120,
                )
                break  # Success, exit the loop
            except RateLimitError:
                logfire.debug(
                    f"Rate limit error, retrying in 30 seconds ({retry_count}/{max_retries})..."
                )
                retry_count += 1
                if retry_count >= max_retries:
                    raise  # Re-raise the exception after max retries
                await asyncio.sleep(15)  # Wait for 30 seconds before retrying

        return message.content[-1].text, ModelUsage(
            cache_creation_input_tokens=message.usage.cache_creation_input_tokens,
            cache_read_input_tokens=message.usage.cache_read_input_tokens,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
    elif model in [Model.gpt_4o, Model.gpt_4o_mini]:
        openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        message = await openai_client.chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=10_000,
        )
        cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.nvidia_llama_3_1_nemotron_70b_instruct:
        message = await nvidia_client.chat.completions.create(
            model=model.value,
            messages=text_only_messages(messages),
            temperature=temperature,
            max_tokens=10_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.groq_llama_3_2_90b_vision:
        message = await groq_client.chat.completions.create(
            model=model.value,
            messages=text_only_messages(messages),
            temperature=temperature,
            max_tokens=8_192,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.openrouter_claude_3_5_sonnet:
        message = await openrouter_client.chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=10_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.openrouter_o1:
        message = await openrouter_client.chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=20_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.openrouter_o1_mini:
        message = await openrouter_client.chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=20_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.azure_gpt_4o:
        message = await azure_client.chat.completions.create(
            model=model.value.replace("azure-", ""),
            messages=messages,
            temperature=temperature,
            max_tokens=10_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.azure_gpt_4o_mini:
        message = await azure_client.chat.completions.create(
            model=model.value.replace("azure-", ""),
            messages=messages,
            temperature=temperature,
            max_tokens=10_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.gemini_1_5_pro:
        if messages[0]["role"] == "system":
            system_messages = messages[0]["content"]
            messages = messages[1:]
        else:
            system_messages = []
        model = genai.GenerativeModel(
            model.value, system_instruction=system_messages[0]["text"]
        )
        gemini_contents = []
        for message in messages:
            if message["role"] == "assistant":
                role = "model"
            else:
                role = message["role"]
            # debug(message["content"])
            if type(message["content"]) is str:
                parts = [genai.types.PartDict(text=message["content"])]
            else:
                parts = []
                for c in message["content"]:
                    if c["type"] == "text":
                        parts.append(genai.types.PartDict(text=c["text"]))
                    elif c["type"] == "image_url":
                        image = PIL.Image.open(
                            io.BytesIO(
                                base64.b64decode(
                                    c["image_url"]["url"].replace(
                                        "data:image/png;base64,", ""
                                    )
                                )
                            )
                        )
                        if image.mode == "RGBA":
                            image = image.convert("RGB")
                        parts.append(image)
            gemini_contents.append(genai.types.ContentDict(role=role, parts=parts))
        response = await model.generate_content_async(
            contents=gemini_contents,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=10_000,
            ),
        )
        return response.text, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )
    else:
        raise ValueError(f"Invalid model: {model}")


noop_code = """
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    raise NotImplementedError()
""".strip()


def clean_code(s: str) -> str:
    return s.replace("\t", " " * 4)


def parse_python_backticks(s: str) -> str:
    if s.count("```python") == 0:
        logfire.debug("NO CODE BLOCKS")
        out = s.partition("</reasoning>")[2]
        if out == "":
            return noop_code
        return clean_code(out)

    if s.count("```python") > 1:
        # print(f"MULTIPLE CODE BLOCKS\n=====\n\n{s}\n\n=====")
        for chunk in s.split("```python")[::-1]:
            if "def transform(" in chunk:
                s = "```python" + chunk
                break

    assert s.count("```python") == 1

    attempted_search = re.search(r"```python\n(.*)\n```", s, re.DOTALL | re.MULTILINE)
    if attempted_search is not None:
        return clean_code(attempted_search.group(1))

    attempted_search = re.search(r"```python\n(.*)\n`", s, re.DOTALL | re.MULTILINE)
    if attempted_search is not None:
        logfire.debug("PARSE ERROR CASE (1)")
        return clean_code(attempted_search.group(1))
    else:
        logfire.debug("PARSE ERROR CASE (2!)")

    return clean_code(s.partition("```python")[2])


def parse_2d_arrays_from_string(s: str) -> list[list[list[int]]]:
    # Regular expression pattern to match 2D arrays
    pattern = r"\[\s*(\[[^\[\]]*\](?:,\s*\[[^\[\]]*\])*\s*)\]"

    # Find all matches of the pattern in the output string
    matches = re.findall(pattern, s)

    # Process each match to create a list of 2D arrays
    arrays_list: list[list[list[int]]] = []

    for match in matches:
        # Find all inner arrays within the matched 2D array
        rows = re.findall(r"\[([^\]]*)\]", match)
        array_2d = []
        for row in rows:
            # Split the row by commas and convert to integers
            nums = [int(n.strip()) for n in row.split(",") if n.strip()]
            array_2d.append(nums)
        arrays_list.append(array_2d)

    return arrays_list
