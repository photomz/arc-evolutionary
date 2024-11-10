from fastapi import APIRouter, FastAPI

from src.logic import (
    GRID,
    CacheData,
    Challenge,
    RootAttemptConfig,
    solve_challenge_background,
)

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/solve_challenge", response_model=tuple[list[GRID], list[GRID]])
async def solve_challenge(
    tree: list[RootAttemptConfig],
    challenge: Challenge,
    cache_data: CacheData,
    environ_data: dict[str, str],
) -> tuple[list[GRID], list[GRID]]:
    return await solve_challenge_background(
        tree=tree, challenge=challenge, cache_data=cache_data, environ_data=environ_data
    )


from pydantic import BaseModel

from src.run_python import GRID, PythonResult
from src.run_python import run_python_transform as transform


class TransformInput(BaseModel):
    code: str
    grid_lists: list[GRID]
    timeout: int
    raise_exception: bool


@app.post("/run_python_transform", response_model=list[PythonResult | None])
def run_python_transform(inputs: list[TransformInput]) -> list[PythonResult | None]:
    print(f"RUNNING PYTHON: {len(inputs)}")
    results: list[PythonResult | None] = []
    for input in inputs:
        try:
            results.append(
                transform(
                    code=input.code,
                    grid_lists=input.grid_lists,
                    timeout=input.timeout,
                    raise_exception=input.raise_exception,
                )
            )
        except Exception as e:
            print(f"ERROR RUNNING PYTHON: {e}")
            results.append(None)
    return results
