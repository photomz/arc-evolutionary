from fastapi import APIRouter, FastAPI

from src.logic import (
    GRID,
    CacheData,
    Challenge,
    RootAttemptConfig,
    solve_challenge_background,
    solve_challenge_server,
)
from src.run_python import PythonResult, TransformInput
from src.run_python import run_python_transforms as _transforms

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/solve_challenge", response_model=tuple[list[GRID], list[GRID]])
async def solve_challenge(
    tree: list[RootAttemptConfig],
    challenge: Challenge,
    env_vars: dict[str, str],
) -> tuple[list[GRID], list[GRID]]:
    return await solve_challenge_server(
        tree=tree, challenge=challenge, env_vars=env_vars
    )


@app.post("/run_python_transform", response_model=list[PythonResult | None])
async def run_python_transforms(
    inputs: list[TransformInput],
) -> list[PythonResult | None]:
    return await _transforms(inputs=inputs)
