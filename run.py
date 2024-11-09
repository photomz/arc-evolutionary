import asyncio
import json
import os
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from src import logfire
from src.data import build_challenges
from src.logic import (
    CacheData,
    random_string,
    solve_challenge,
    solve_challenge_background,
)
from src.models import GRID
from src.trees.prod import (
    RootAttemptConfig,
    big_claude_tree,
    fast_claude_tree,
    small_claude_tree,
)


class ChallengeSolution(BaseModel):
    attempt_1: GRID
    attempt_2: GRID


async def run_from_json(
    *,
    challenges_path: str,
    solutions_path: str,
    tree: list[RootAttemptConfig],
    limit: int | None,
    only_run_ids: set[str] = None,
) -> None:
    challenges = build_challenges(
        challenges_path=Path(challenges_path), solutions_path=None
    )
    if only_run_ids:
        challenges = {k: challenges[k] for k in only_run_ids}
    if limit:
        # only include the first 10 challenges
        challenges = {k: challenges[k] for k in list(challenges)[:limit]}

    solutions_d: dict[str, list[ChallengeSolution]] = {}
    for challenge_id, challenge in challenges.items():
        print(f"[{challenge_id}] starting challenge...")
        first_solutions, second_solutions = await solve_challenge_background(
            challenge=challenge,
            tree=tree,
            cache_data=CacheData(
                redis_dsn=os.environ["REDIS_DSN"],
                run_id=random_string(),
            ),
            environ_data={
                "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            },
            url=os.environ["SERVER_URL"],
        )
        solutions_d[challenge_id] = []
        for i in range(len(first_solutions)):
            solutions_d[challenge_id].append(
                ChallengeSolution(
                    attempt_1=first_solutions[i],
                    attempt_2=second_solutions[i],
                )
            )
        # just write after each challenge in case program crashes
        logfire.debug(
            "solution",
            challenge_id=challenge_id,
            challenge=challenge,
            solution_d=solutions_d[challenge_id],
        )
        open(solutions_path, "w").write(
            TypeAdapter(dict[str, list[ChallengeSolution]])
            .dump_json(solutions_d)
            .decode("utf-8")
        )


async def run() -> None:
    await run_from_json(
        # challenges_path="test_data/challenges.json",
        challenges_path="arc-prize-2024/arc-agi_evaluation_challenges.json",
        solutions_path="test_data/eval_solutions.json",
        tree=fast_claude_tree,
        limit=1,
        # only_run_ids={},
    )


def evaluate_solutions(attempts_solutions_path: str, truth_solutions_path: str) -> None:
    truth: dict[str, list[GRID]] = json.loads(open(truth_solutions_path).read())
    attempts: dict[str, list[ChallengeSolution]] = TypeAdapter(
        dict[str, list[ChallengeSolution]]
    ).validate_json(open(attempts_solutions_path).read())
    total_count = 0
    correct_count = 0
    for challenge_id, attempt_list in attempts.items():
        truth_grids: list[GRID] = truth[challenge_id]
        for i, truth_grid in enumerate(truth_grids):
            total_count = total_count + 1
            attempt_grids = attempt_list[i]
            if attempt_grids.attempt_1 == truth_grid:
                correct_count = correct_count + 1
            elif attempt_grids.attempt_2 == truth_grid:
                correct_count = correct_count + 1

    print("total count", total_count, "correct count", correct_count)


async def main() -> None:
    await run()
    evaluate_solutions(
        attempts_solutions_path="test_data/eval_solutions.json",
        truth_solutions_path="arc-prize-2024/arc-agi_evaluation_solutions.json",
    )


if __name__ == "__main__":
    asyncio.run(main())
