import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import run_agent

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

EVALS_DIR = Path(__file__).resolve().parent

TEST_CASES_PATH = EVALS_DIR / "test_cases.json"

RESULTS_DIR = EVALS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Value matching
# ---------------------------------------------------------


def normalize_string(value: str) -> str:
    return value.strip().lower()


def value_matches(
    actual: Any,
    expected: Any,
) -> bool:

    # ---------------------------------------------
    # Matcher object
    # ---------------------------------------------

    if isinstance(expected, dict):
        # Approximate numeric comparison
        if "approx" in expected:
            if not isinstance(actual, (int, float)):
                return False

            target = expected["approx"]
            tolerance = expected.get(
                "tolerance",
                0.01,
            )

            return abs(actual - target) <= tolerance

        # String contains comparison
        if "contains" in expected:
            if not isinstance(actual, str):
                return False

            return normalize_string(expected["contains"]) in normalize_string(actual)

    # ---------------------------------------------
    # Normal string comparison
    # ---------------------------------------------

    if isinstance(actual, str) and isinstance(
        expected,
        str,
    ):
        return normalize_string(actual) == normalize_string(expected)

    return actual == expected


# ---------------------------------------------------------
# Argument evaluation
# ---------------------------------------------------------


def evaluate_arguments(
    actual_trace: list[dict],
    expected_args: dict,
) -> tuple[int, int]:

    correct = 0
    total = 0

    # tool name -> calls
    calls_by_tool: dict[str, list[dict]] = {}

    for call in actual_trace:
        calls_by_tool.setdefault(
            call["name"],
            [],
        ).append(call)

    for tool_name, expected_tool_args in expected_args.items():
        actual_calls = calls_by_tool.get(
            tool_name,
            [],
        )

        if not actual_calls:
            total += len(expected_tool_args)
            continue

        # For this project we expect at most one call
        # of each tool per test.
        actual_args = actual_calls[0].get(
            "arguments",
            {},
        )

        for argument_name, expected_value in expected_tool_args.items():
            total += 1

            actual_value = actual_args.get(argument_name)

            if value_matches(
                actual_value,
                expected_value,
            ):
                correct += 1

    return correct, total


# ---------------------------------------------------------
# Tool execution success
# ---------------------------------------------------------


def tool_call_succeeded(call: dict) -> bool:

    if call.get("is_error", False):
        return False

    output = str(call.get("output", "")).lower()

    error_markers = [
        '"error"',
        "unable to fetch",
        "unable to find",
        "could not",
        "failed",
    ]

    return not any(marker in output for marker in error_markers)


# ---------------------------------------------------------
# Single test
# ---------------------------------------------------------


async def evaluate_test_case(
    test_case: dict,
) -> dict:

    query = test_case["query"]

    started = time.perf_counter()

    try:
        result = await run_agent(
            history=[],
            query=query,
        )

        latency = time.perf_counter() - started

        trace = result.get(
            "tool_trace",
            [],
        )

        answer = result.get(
            "answer",
            "",
        )

        exception = None

    except Exception as exc:
        latency = time.perf_counter() - started

        trace = []
        answer = ""
        exception = repr(exc)

    # -----------------------------------------------------
    # Tool selection
    # -----------------------------------------------------

    actual_tools = [call["name"] for call in trace]

    expected_tools = test_case.get(
        "expected_tools",
        [],
    )

    # Exact multiset match, order-independent
    tool_selection_correct = Counter(actual_tools) == Counter(expected_tools)

    # -----------------------------------------------------
    # First tool
    # -----------------------------------------------------

    expected_first_tool = test_case.get("expected_first_tool")

    if expected_first_tool is None:
        first_tool_correct = len(actual_tools) == 0
    else:
        first_tool_correct = len(actual_tools) > 0 and actual_tools[0] == expected_first_tool

    # -----------------------------------------------------
    # Arguments
    # -----------------------------------------------------

    correct_args, total_args = evaluate_arguments(
        actual_trace=trace,
        expected_args=test_case.get(
            "expected_args",
            {},
        ),
    )

    argument_accuracy = correct_args / total_args if total_args else 1.0

    # -----------------------------------------------------
    # Successful execution
    # -----------------------------------------------------

    successful_calls = sum(1 for call in trace if tool_call_succeeded(call))

    execution_success_rate = successful_calls / len(trace) if trace else 1.0

    # -----------------------------------------------------
    # Unnecessary tool calls
    # -----------------------------------------------------

    expected_counter = Counter(expected_tools)

    actual_counter = Counter(actual_tools)

    unnecessary_calls = sum(
        max(
            0,
            actual_counter[tool] - expected_counter[tool],
        )
        for tool in actual_counter
    )

    unnecessary_tool_rate = unnecessary_calls / len(actual_tools) if actual_tools else 0.0

    return {
        "id": test_case["id"],
        "query": query,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "tool_selection_correct": tool_selection_correct,
        "first_tool_correct": first_tool_correct,
        "argument_accuracy": round(argument_accuracy, 4),
        "execution_success_rate": round(
            execution_success_rate,
            4,
        ),
        "unnecessary_tool_rate": round(
            unnecessary_tool_rate,
            4,
        ),
        "latency_seconds": round(latency, 3),
        "answer": answer,
        "tool_trace": trace,
        "exception": exception,
    }


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------


async def main():

    with open(
        TEST_CASES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        test_cases = json.load(file)

    print(f"\nRunning {len(test_cases)} evaluation cases...\n")

    results = []

    for index, test_case in enumerate(
        test_cases,
        start=1,
    ):
        print(f"[{index}/{len(test_cases)}] {test_case['id']}")

        result = await evaluate_test_case(test_case)

        results.append(result)

        status = "PASS" if result["tool_selection_correct"] else "FAIL"

        print(f"  Tool routing: {status}")

        print(f"  Expected: {result['expected_tools']}")

        print(f"  Actual:   {result['actual_tools']}")

        print(f"  Argument accuracy: {result['argument_accuracy']:.0%}")

        print(f"  Latency: {result['latency_seconds']}s\n")

    # -----------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------

    count = len(results)

    tool_selection_accuracy = sum(result["tool_selection_correct"] for result in results) / count

    first_tool_accuracy = sum(result["first_tool_correct"] for result in results) / count

    argument_accuracy = sum(result["argument_accuracy"] for result in results) / count

    execution_success_rate = sum(result["execution_success_rate"] for result in results) / count

    unnecessary_tool_rate = sum(result["unnecessary_tool_rate"] for result in results) / count

    average_latency = sum(result["latency_seconds"] for result in results) / count

    exception_rate = sum(result["exception"] is not None for result in results) / count

    summary = {
        "number_of_tests": count,
        "tool_selection_accuracy": round(
            tool_selection_accuracy,
            4,
        ),
        "first_tool_accuracy": round(
            first_tool_accuracy,
            4,
        ),
        "argument_accuracy": round(
            argument_accuracy,
            4,
        ),
        "execution_success_rate": round(
            execution_success_rate,
            4,
        ),
        "unnecessary_tool_rate": round(
            unnecessary_tool_rate,
            4,
        ),
        "average_latency_seconds": round(
            average_latency,
            3,
        ),
        "exception_rate": round(
            exception_rate,
            4,
        ),
    }

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_path = RESULTS_DIR / f"eval_{timestamp}.json"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "summary": summary,
                "results": results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)

    print(f"Tests:                    {count}")

    print(f"Tool selection accuracy:  {tool_selection_accuracy:.1%}")

    print(f"First tool accuracy:      {first_tool_accuracy:.1%}")

    print(f"Argument accuracy:        {argument_accuracy:.1%}")

    print(f"Execution success rate:   {execution_success_rate:.1%}")

    print(f"Unnecessary tool rate:    {unnecessary_tool_rate:.1%}")

    print(f"Average latency:           {average_latency:.2f}s")

    print(f"Exception rate:            {exception_rate:.1%}")

    print("=" * 55)

    print(f"\nResults saved to:\n{output_path}")


if __name__ == "__main__":
    asyncio.run(main())
