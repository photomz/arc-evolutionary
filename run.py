import asyncio
import json
from pathlib import Path

import logfire
from devtools import debug
from pydantic import BaseModel, TypeAdapter

from src.data import build_challenges
from src.logic import solve_challenge
from src.models import GRID
from src.trees.prod import RootAttemptConfig, big_claude_tree, small_claude_tree


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
        print(f"solving challenge {challenge_id}")
        first_solutions, second_solutions = await solve_challenge(
            challenge=challenge, tree=tree
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
            solutions_d=solutions_d,
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
        tree=small_claude_tree,
        limit=1,
        only_run_ids={"0934a4d8"},
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
