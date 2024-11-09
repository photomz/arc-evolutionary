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
    tree: list[RootAttemptConfig], challenge: Challenge, cache_data: CacheData
) -> tuple[list[GRID], list[GRID]]:
    return await solve_challenge_background(
        tree=tree, challenge=challenge, cache_data=cache_data
    )
