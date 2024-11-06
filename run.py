import asyncio
import json
from code.data import build_challenges
from code.logic import solve_challenge
from code.models import GRID
from code.trees.prod import RootAttemptConfig, big_claude_tree, small_claude_tree
from pathlib import Path

from pydantic import BaseModel, TypeAdapter


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
        _first_solution, _second_solution = await solve_challenge(
            challenge=challenge, tree=tree
        )
        # TODO fix, output should be list of solutions
        first_solutions: list[GRID] = [_first_solution]
        second_solutions: list[GRID] = [_second_solution]

        solutions_d[challenge_id] = []
        for i in range(len(first_solutions)):
            solutions_d[challenge_id].append(
                ChallengeSolution(
                    attempt_1=first_solutions[i],
                    attempt_2=second_solutions[i],
                )
            )
        # just write after each challenge in case program crashes

        open(solutions_path, "w").write(
            TypeAdapter(dict[str, list[ChallengeSolution]])
            .dump_json(solutions_d)
            .decode("utf-8")
        )


async def run() -> None:
    await run_from_json(
        # challenges_path="test_data/challenges.json",
        challenges_path="test_data/eval_challenges.json",
        solutions_path="test_data/eval_solutions.json",
        tree=small_claude_tree,
        limit=1,
        only_run_ids={"12997ef3"},
    )


if __name__ == "__main__":
    asyncio.run(run())
