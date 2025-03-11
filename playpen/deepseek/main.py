import os

from playpen.deepseek.custom_devtools import debug
from openai import AsyncOpenAI
from dataclasses import dataclass
from enum import Enum
from tqdm.asyncio import tqdm
from time import time

from src.llms import get_next_messages
from src.models import Model
import numpy as np
import pandas as pd


class ProviderName(str, Enum):
    Deepseek = "deepseek"
    Hyperbolic = "hyperbolic"


@dataclass
class Provider:
    name: ProviderName
    model_name: str
    api_key: str
    base_url: str


deepseek = Provider(
    name=ProviderName.Deepseek,
    model_name="deepseek-reasoner",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

hyperbolic = Provider(
    name=ProviderName.Hyperbolic,
    model_name="deepseek-ai/DeepSeek-R1",
    api_key=os.environ["HYPERBOLIC_API_KEY"],
    base_url="https://api.hyperbolic.xyz/v1",
)

model = hyperbolic


async def main() -> None:
    deepseek_client = AsyncOpenAI(
        api_key=model.api_key,
        base_url=model.base_url,
    )

    res = await deepseek_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "what is 2+2?"},
        ],
        model="deepseek/deepseek-r1",
    )

    m = await get_next_messages(
        model=Model.deep_seek_r1,
        temperature=0.95,
        n_times=1,
    )
    debug(m)


async def main_again() -> None:
    # Please install OpenAI SDK first: `pip3 install openai`

    from openai import OpenAI

    client = OpenAI(api_key=model.api_key, base_url=model.base_url)

    response = client.chat.completions.create(
        model=model.model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Prove Heine Borel to a five year old."},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )
    print("Sent request")
    completion = ""
    for chunk in response:
        debug(chunk)
        if len(chunk.choices):
            completion += chunk.choices[0].delta.content
        if u := chunk.usage:
            usage_format = {
                "cache_creation_input_tokens": (
                    u.prompt_cache_miss_tokens
                    if hasattr(u, "prompt_cache_miss_tokens")
                    else 0
                ),
                "cache_read_input_tokens": (
                    u.prompt_cache_hit_tokens
                    if hasattr(u, "prompt_cache_hit_tokens")
                    else 0
                ),
                "input_tokens": u.prompt_tokens,
                "output_tokens": u.completion_tokens,
                "reasoning_tokens": (
                    u.completion_tokens_details.reasoning_tokens
                    if u.completion_tokens_details
                    else None
                ),
            }
            debug(usage_format)
    debug(completion)


async def timed_run(
    client: AsyncOpenAI, id: int, prompt: str
) -> tuple[str, list[float]]:
    response = await client.chat.completions.create(
        model=model.model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    completion = ""
    token_times = []
    t_prev = time()
    async for chunk in (pbar := tqdm(response, desc=f"n={id}")):
        if len(chunk.choices):
            delta = chunk.choices[0].delta.content or ""
            t_now = time()
            t_delta = t_now - t_prev
            token_times += [t_delta]
            t_prev = t_now
            pbar.set_postfix(delta=delta)
            completion += delta
        if chunk.usage:
            total_tokens = chunk.usage.total_tokens

    print(f"{id} done")

    return completion, token_times, total_tokens


def init_latency_plot():
    import plotly.graph_objects as go

    # Create figure
    fig = go.Figure()

    # Initialize empty plots
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            name="Mean Latency",
            mode="lines+markers",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            name="±1 Std Dev",
            fill="tonexty",
            mode="none",
            fillcolor="rgba(0,0,255,0.2)",
        )
    )

    # Update layout
    fig.update_layout(
        title="DeepSeek Latency Analysis",
        xaxis_title="Concurrent Requests (k)",
        yaxis_title="Latency (seconds)",
        showlegend=True,
        height=600,
    )

    return fig


async def main_parallel(prompt="Hello", max_k=500) -> None:
    client = AsyncOpenAI(api_key=model.api_key, base_url=model.base_url)

    # Initialize plot
    fig = init_latency_plot()
    fig.show()

    # Initialize data storage
    ks = []
    means = []
    stds = []

    k = 1
    while k <= max_k:
        print(f"Simulated load: {k} requests")
        tasks = [timed_run(client, i, prompt) for i in range(k)]
        results = await asyncio.gather(*tasks)

        # Process results
        batch_times = []
        total_tokens = 0
        for _, timings, usage in results:
            batch_times += timings
            total_tokens += usage

        print(f"Cost: ${(total_tokens/1e6*2):.2f}")
        mean_latency = np.mean(batch_times)
        std_latency = np.std(batch_times)

        ks += [k]
        means += [mean_latency]
        stds += [std_latency]

        with fig.batch_update():
            fig.data[0].x = ks
            fig.data[0].y = means
            fig.data[1].x = ks + ks[::-1]
            fig.data[1].y = (np.array(means) + np.array(stds)).tolist() + (
                np.array(means) - np.array(stds)
            )[::-1].tolist()
        fig.show()

        debug(k, mean_latency, std_latency)

        k *= 2


if __name__ == "__main__":
    import asyncio

    asyncio.run(main_parallel(max_k=500))
