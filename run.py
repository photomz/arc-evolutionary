import asyncio
import json
from code.data import build_challenges
from code.logic import solve_challenge
from code.trees.prod import big_claude_tree, small_claude_tree
from pathlib import Path

SOLUTION_TYPE = list[list[list[int]]]


async def run_from_json(
    *, challenges_path: str, solutions_path: str, limit: int | None
) -> None:
    challenges = build_challenges(
        challenges_path=Path(challenges_path), solutions_path=None
    )
    if limit:
        # only include the first 10 challenges
        challenges = {k: challenges[k] for k in list(challenges)[:limit]}

    solutions_d: dict[str, tuple[SOLUTION_TYPE, SOLUTION_TYPE]] = {}
    for challenge_id, challenge in challenges.items():
        first_solution, second_solution = await solve_challenge(
            challenge=challenge, tree=small_claude_tree
        )
        # TODO fix, output should be list of solutions
        solutions_d[challenge_id] = [first_solution], [second_solution]
        # just write after each challenge in case program crashes
        open(solutions_path, "w").write(json.dumps(solutions_d))


async def run() -> None:
    await run_from_json(
        challenges_path="test_data/challenges.json",
        solutions_path="test_data/solutions.json",
        limit=2,
    )


if __name__ == "__main__":
    asyncio.run(run())
