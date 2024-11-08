import os
import time
import typing as T
from copy import deepcopy

import logfire
import numpy as np
from tqdm.asyncio import tqdm_asyncio

from src import PLOT
from src.data import training_challenges
from src.models import (
    GRID,
    Attempt,
    AttemptEdge,
    Challenge,
    FixAttemptConfig,
    Prompt,
    RootAttemptConfig,
    prompt_map,
    random_string,
)
from src.prompts.examples import (
    GRID_CHANGE_PROMPT_EXAMPLE_1,
    GRID_SAME_PROMPT_EXAMPLE_1,
    example_1_grid_change_challenge_id,
    example_1_reasoning_grid_change,
    example_1_same_grid_challenge_id,
    example_1_same_grid_reasoning,
    example_2_challenge_id,
    example_2_reasoning_grid_same,
    example_3_challenge_id,
    example_3_reasoning_grid_same,
    example_7_grid_change_challenge_id,
    example_7_reasoning_grid_change_bad_colors,
)
from src.render_legacy import grid_to_base64_png_oai_content
from src.reps import array_to_str, grid_diffs_to_ascii, grid_to_ascii
from src.run_python import run_python_transform


class TqdmLogfire:
    """File-like class redirecting tqdm progress bar to given logging logger."""

    def __init__(self):
        pass

    def write(self, msg: str) -> None:
        logfire.debug(msg.lstrip("\r"))

    def flush(self) -> None:
        pass


def chunk_list(lst: list, n: int) -> list[list]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def content_blocks_from_matrix(
    *,
    matrix: GRID,
    _label: str,
    include_image: bool,
    use_ascii: bool,
    use_array: bool,
) -> list[dict[str, str]]:
    matrix = deepcopy(matrix)
    grid = np.array(matrix)
    x, y = grid.shape
    messages = [
        {"type": "text", "text": _label},
        {"type": "text", "text": f"Shape: {x} by {y}\n\n"},
    ]
    if include_image:
        messages.append(grid_to_base64_png_oai_content(grid=grid))
    if use_ascii:
        messages.append(
            {
                "type": "text",
                "text": f"ASCII representation:\n\n{grid_to_ascii(grid=grid, separator='|', spreadsheet_ascii=False)}\n\n",
            }
        )
    if use_array:
        messages.append({"type": "text", "text": array_to_str(grid=matrix)})
    return messages


def content_from_challenge(
    challenge: Challenge,
    include_diffs: bool,
    include_image: bool,
    use_ascii: bool,
    use_array: bool,
) -> list[dict[str, T.Any]]:
    content = []
    for i, train in enumerate(challenge.train):
        example_number = i + 1
        # add input blocks
        content.extend(
            content_blocks_from_matrix(
                matrix=train.input,
                _label=f"# Example {example_number}\n\n## Input {example_number}\n\n",
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
        )
        # add output blocks
        content.extend(
            content_blocks_from_matrix(
                matrix=train.output,
                _label=f"## Output {example_number}\n\n",
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
        )
        if not does_grid_change_shape(challenge=challenge) and include_diffs:
            content.append(
                {
                    "type": "text",
                    "text": f"## Color changes between the Input and Output ASCII representation:\n\n"
                    f"{grid_diffs_to_ascii(grid_input=np.array(train.input), grid_output=np.array(train.output), separator='|')}\n\n",
                }
            )

    # TODO for now, only do the first test... Will have to treat these multi tests as multiple examples later
    # assert len(challenge.test) == 1
    content.extend(
        content_blocks_from_matrix(
            matrix=challenge.test[0].input,
            _label="# Additional input\n\n",
            include_image=include_image,
            use_ascii=use_ascii,
            use_array=use_array,
        )
    )

    return content


def does_grid_change_shape(challenge: Challenge) -> bool:
    for train in challenge.train:
        if np.array(train.input).shape != np.array(train.output).shape:
            return True
    return False


def challenge_to_messages(
    *,
    challenge: Challenge,
    add_examples: bool,
    use_cache_control: bool = True,
    include_diffs: bool,
    prompt: Prompt,
    include_image: bool,
    use_ascii: bool,
    use_array: bool,
) -> list[dict[str, str]]:
    # first, is example same grid size?
    grid_change_shape = does_grid_change_shape(challenge)
    # debug(grid_change_shape)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt_map[prompt]}]}
    ]
    if add_examples:
        if grid_change_shape:
            # messages.extend(GRID_CHANGE_PROMPT_EXAMPLE_1)
            example_1_grid_change_prompt = content_from_challenge(
                challenge=training_challenges[example_1_grid_change_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
            example_7_grid_change_prompt = content_from_challenge(
                challenge=training_challenges[example_7_grid_change_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": example_1_grid_change_prompt,
                    },
                    {"role": "assistant", "content": example_1_reasoning_grid_change},
                ]
            )
        else:
            example_1_grid_same_prompt = content_from_challenge(
                challenge=training_challenges[example_1_same_grid_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )

            # ADDING OTHER EXAMPLE!
            example_2_grid_same_prompt = content_from_challenge(
                challenge=training_challenges[example_2_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
            example_3_grid_same_prompt = content_from_challenge(
                challenge=training_challenges[example_3_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": example_1_grid_same_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": example_1_same_grid_reasoning,
                    },
                ]
            )

        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Great work! Now I will give you another puzzle to solve just like that one.",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Great, please give me the next puzzle.",
                        }
                    ],
                },
            ]
        )
    if use_cache_control:
        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    content = content_from_challenge(
        challenge=challenge,
        include_diffs=include_diffs,
        include_image=include_image,
        use_ascii=use_ascii,
        use_array=use_array,
    )
    if use_cache_control:
        content[-1]["cache_control"] = {"type": "ephemeral"}
    messages.append({"role": "user", "content": content})
    return messages


def eval_attempts(
    attempts: list[Attempt], config: RootAttemptConfig | FixAttemptConfig, plot: bool
) -> None:
    if not attempts:
        return None

    for attempt in attempts:
        # debug(attempt.train_accuracy, attempt.test_accuracy)
        if plot:
            try:
                start = time.time()
                attempt.plot(ignore_fixing=True)
                took = time.time() - start
                if took > 0.5:
                    logfire.debug(f"TOOK {took} SECONDS TO PLOT")
            except Exception as e:
                logfire.debug(f"FAILED TO PLOT: {e}")

    # get total accuracies
    avg_train_accuracy = sum(attempt.train_accuracy for attempt in attempts) / len(
        attempts
    )
    avg_test_accuracy = sum(attempt.test_accuracy for attempt in attempts) / len(
        attempts
    )
    total_cost = sum(attempt.cost_cents for attempt in attempts)
    total_runs = len(attempts)
    total_correct = len(
        [a for a in attempts if a.test_accuracy == 1 and a.train_accuracy == 1]
    )
    debug_d = {
        "challenge_id": attempts[0].challenge.id,
        "total_runs": total_runs,
        "total_correct": total_correct,
        "avg_train_accuracy": avg_train_accuracy,
        "avg_test_accuracy": avg_test_accuracy,
        "total_cost": total_cost,
        "prompt_config": config.prompt_config,
        "llm_config": config.llm_config,
    }
    logfire.debug("eval", **debug_d)
    print(
        f"[{attempts[0].challenge.id}] finished processing node:",
        {
            "total_runs": total_runs,
            "avg_train_accuracy": avg_train_accuracy,
            "total_cost": total_cost,
        },
    )


def get_best_attempts(
    attempts: list[Attempt], k_top: int, unique_code: bool, unique_output: bool
) -> list[Attempt]:
    # first, order attempts by how many examples they got right
    # then, order by the diff in cells
    # use a better metric later
    example_correct: list[Attempt] = []
    example_wrong: list[Attempt] = []
    for a in attempts:
        if a.train_accuracy > 0:
            example_correct.append(a)
        else:
            example_wrong.append(a)
    sorted_correct = sorted(
        example_correct, key=lambda a: a.train_accuracy, reverse=True
    )
    sorted_wrong = sorted(
        example_wrong,
        key=lambda a: a.avg_cell_diff_percent,
        reverse=True,
    )
    all_sorted: list[Attempt] = [*sorted_correct, *sorted_wrong]

    if unique_code:
        has_seen_python: set[str] = set()
        unique_sorted = []
        for item in all_sorted:
            code_str = item.python_code_str
            if code_str not in has_seen_python:
                unique_sorted.append(item)
                has_seen_python.add(code_str)
        all_sorted = unique_sorted
    if unique_output:
        has_seen_grid: set[str] = set()
        unique_sorted = []
        for item in all_sorted:
            output_grid = str(item.test_attempt)
            if output_grid not in has_seen_grid:
                unique_sorted.append(item)
                has_seen_grid.add(output_grid)
        all_sorted = unique_sorted

    return all_sorted[:k_top]


def get_diverse_attempts(
    root_attempt: Attempt, sorted_attempts: list[Attempt], limit: int
) -> list[Attempt]:
    if root_attempt in sorted_attempts:
        sorted_attempts.remove(root_attempt)
    attempts_by_correct_examples: dict[int, list[Attempt]] = {}
    correct_examples_by_attempt: dict[Attempt, set[int]] = {}
    for a in [root_attempt, *sorted_attempts]:
        for i, train_example in enumerate(a.challenge.train):
            if a.train_attempts[i] == train_example.output:
                if i not in attempts_by_correct_examples:
                    attempts_by_correct_examples[i] = []
                attempts_by_correct_examples[i].append(a)
                if a not in correct_examples_by_attempt:
                    correct_examples_by_attempt[a] = set()
                correct_examples_by_attempt[a].add(i)
    # make sure you have at least one attempt for each correct example
    final_attempts: list[Attempt] = [root_attempt, *sorted_attempts][0:limit]
    count: dict[int, int] = {}
    for a in final_attempts:
        for ii in correct_examples_by_attempt.get(a, set()):
            count[ii] = 1
    # find the missing ones
    missing = attempts_by_correct_examples.keys() - count.keys()
    for miss in missing:
        final_attempts.append(attempts_by_correct_examples[miss][0])
    return final_attempts


async def run_fixes_tree(
    parent_attempts: list[Attempt],
    edges: list[AttemptEdge],
    warm_cache: bool,  # too complex rn w speed
) -> list[Attempt]:
    # DFS fixes
    all_attempts: list[Attempt] = []
    if not edges:
        return all_attempts
    for edge in edges:
        best_k = get_best_attempts(
            attempts=parent_attempts,
            k_top=edge.k_top_config.k_top,
            unique_code=edge.k_top_config.unique_code,
            unique_output=edge.k_top_config.unique_output,
        )
        for fix_attempt_config in edge.configs:
            print(
                f"running fix node with {fix_attempt_config.attempts * len(best_k)} total attempts."
            )
            if fix_attempt_config.attempts == 0:
                continue
            local_attempts: list[Attempt] = []
            tasks = []
            for parent_attempt in best_k:
                for _ in range(fix_attempt_config.attempts):
                    if not edge.pooling:
                        tasks.append(
                            parent_attempt.fix(
                                attempt_config=fix_attempt_config.model_copy(deep=True),
                                raise_exception=False,
                            )
                        )
                        """
                        tasks.append(
                            Attempt.run(
                                challenge=parent_attempt.challenge,
                                attempt_config=fix_attempt_config.model_copy(deep=True),
                                raise_exception=False,
                                fixing=[parent_attempt],
                            )
                        )
                        """
                    else:
                        # get the pool of attempts
                        # get diversity of correct examples here...
                        attempts_to_use = get_diverse_attempts(
                            root_attempt=parent_attempt,
                            sorted_attempts=get_best_attempts(
                                attempts=parent_attempts,
                                k_top=100_000,
                                unique_code=edge.k_top_config.unique_code,
                                unique_output=edge.k_top_config.unique_output,
                            ),
                            limit=edge.pooling.size,
                        )
                        tasks.append(
                            Attempt.run(
                                challenge=parent_attempt.challenge,
                                attempt_config=fix_attempt_config.model_copy(deep=True),
                                raise_exception=False,
                                fixing=attempts_to_use,
                            )
                        )
            local_attempts.extend(
                [
                    a
                    for a in await tqdm_asyncio.gather(
                        *tasks, desc="Processing fix attempts", file=TqdmLogfire()
                    )
                    if a
                ]
            )
            start_eval = time.time()
            eval_attempts(attempts=local_attempts, config=fix_attempt_config, plot=PLOT)
            logfire.debug(f"eval took {(time.time() - start_eval)} secs")
            all_attempts.extend(local_attempts)
            # now see if you have a solution
            attempts_with_perfect_train_accuracy = [
                a for a in all_attempts if a.train_accuracy == 1
            ]
            if len(attempts_with_perfect_train_accuracy) >= 2:
                message = f"found 2 solutions with {len(attempts_with_perfect_train_accuracy)} attempts"
                logfire.debug(message)
                print(message)
                return all_attempts

            # now run the fixes
            all_attempts.extend(
                await run_fixes_tree(
                    parent_attempts=local_attempts,
                    edges=fix_attempt_config.fixes,
                    warm_cache=warm_cache,
                )
            )
    return all_attempts


async def run_tree(
    tree: list[RootAttemptConfig],
    challenge: Challenge,
    warm_cache_root: bool,
    warm_cache_fix: bool,
) -> list[Attempt]:
    # run DFS on this tree

    all_attempts: list[Attempt] = []
    for root_attempt_config in tree:
        print(
            f"[{challenge.id}] running root node with {root_attempt_config.attempts} attempts."
        )
        local_attempts: list[Attempt] = []
        if warm_cache_root:
            first_attempt = await Attempt.run(
                challenge=challenge,
                attempt_config=root_attempt_config.model_copy(deep=True),
                raise_exception=False,
                fixing=[],
            )
            if first_attempt:
                local_attempts.append(first_attempt)

        tasks = []
        for _ in range(root_attempt_config.attempts - (1 if warm_cache_root else 0)):
            tasks.append(
                Attempt.run(
                    challenge=challenge,
                    attempt_config=root_attempt_config.model_copy(deep=True),
                    raise_exception=False,
                    fixing=[],
                )
            )

        task_chunks = chunk_list(lst=tasks, n=100)
        for i, task_chunk in enumerate(task_chunks):
            local_attempts.extend(
                a
                for a in await tqdm_asyncio.gather(
                    *task_chunk,
                    desc=f"[{challenge.id}] Processing root attempts, chunk {i + 1}/{len(task_chunks)}",
                    file=TqdmLogfire(),
                )
                if a
            )

        start_eval = time.time()
        eval_attempts(attempts=local_attempts, config=root_attempt_config, plot=PLOT)
        logfire.debug(f"eval took {(time.time() - start_eval)} secs")
        all_attempts.extend(local_attempts)
        # now see if you have a solution
        attempts_with_perfect_train_accuracy = [
            a for a in all_attempts if a.train_accuracy == 1
        ]
        if len(attempts_with_perfect_train_accuracy) >= 2:
            message = f"found 2 solutions with {len(attempts_with_perfect_train_accuracy)} attempts"
            logfire.debug(message)
            print(message)
            return all_attempts

        # now run the fixes
        all_attempts.extend(
            await run_fixes_tree(
                parent_attempts=local_attempts,
                edges=root_attempt_config.fixes,
                warm_cache=warm_cache_fix,
            )
        )

    # remove duplicates
    has_seen: set[str] = set()
    _all_attempts = []
    for a in all_attempts:
        if a.id not in has_seen:
            _all_attempts.append(a)
        has_seen.add(a.id)

    return _all_attempts


def get_grids_from_attempt(attempt: Attempt) -> list[GRID]:
    challenge = attempt.challenge
    if len(challenge.test) == 1:
        return [attempt.test_attempt]
    transform_results = run_python_transform(
        code=attempt.python_code_str,
        grid_lists=[deepcopy(test.input) for test in challenge.test],
        timeout=5,
        raise_exception=True,
    )
    logfire.debug(
        f"FINAL: Transform results took {transform_results.latency_ms:.2f} ms"
    )
    return transform_results.transform_results


async def solve_challenge(
    tree: list[RootAttemptConfig], challenge: Challenge
) -> tuple[list[GRID], list[GRID]]:
    # DFS tree, so always exit early if we find a solution (works for all examples)

    run_id = f"run_{random_string(10)}"
    started_at_ms = time.time() * 1000

    attempts = await run_tree(
        tree=tree, challenge=challenge, warm_cache_root=True, warm_cache_fix=False
    )
    ended_at_ms = time.time() * 1000

    if os.environ.get("NEON_DB_DSN"):
        await Attempt.insert_run(
            run_id=run_id, started_at_ms=started_at_ms, ended_at_ms=ended_at_ms
        )
        await Attempt.insert_many(attempts=attempts, run_id=run_id)

    top_two = get_best_attempts(
        attempts=attempts, k_top=2, unique_code=True, unique_output=True
    )

    if len(top_two) == 1:
        top_two.append(top_two[0])

    first_solution = top_two[0]
    second_solution = top_two[1]

    if PLOT:
        first_solution.plot(ignore_fixing=True)
        second_solution.plot(ignore_fixing=True)

    return get_grids_from_attempt(first_solution), get_grids_from_attempt(
        second_solution
    )
