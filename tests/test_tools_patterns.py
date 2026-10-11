
import pandas as pd
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


@pytest.fixture
def pattern_tool():
    session = SessionState()

    session.dataframe = pd.DataFrame({
        "text": [
            "alpha beta",
            "alpha",
            "alpha",
            "beta",
            "gamma delta",
            "gamma",
            "gamma",
            "gamma"
        ],
        "category_name": [
            "catA", "catA", "catA", "catA",
            "catB", "catB", "catB", "catB"
        ]
    })

    tools, errors = load_workspace_tools(
        "agent_dev_workspace", session
    )
    assert not errors, errors

    return next(
        t for t in tools
        if t.name == "mine_patterns_tool"
    )


@pytest.mark.parametrize(
    "category,term,support",
    [
        ("catA", "alpha", 3),
        ("catB", "gamma", 4)
    ]
)
def test_fpgrowth(pattern_tool, category, term, support):
    result = pattern_tool.invoke({
        "category_name": category,
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert result["patterns"] == [
        {"items": [term], "support": support}
    ]
    assert result["result_id"]


@pytest.mark.parametrize(
    "category,term,support",
    [
        ("catA", "alpha", 3),
        ("catB", "gamma", 4)
    ]
)
def test_topk(pattern_tool, category, term, support):
    result = pattern_tool.invoke({
        "category_name": category,
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 2
    })

    assert result["patterns"] == [
        {"items": [term], "support": support}
    ]


@pytest.mark.parametrize(
    "category",
    ["catA", "catB"]
)
def test_maxfpgrowth(pattern_tool, category):
    result = pattern_tool.invoke({
        "category_name": category,
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })

    assert result["patterns"] == []


@pytest.mark.parametrize(
    "category,method",
    [
        ("catA", "variance"),
        ("catA", "term_frequency"),
        ("catB", "variance"),
        ("catB", "term_frequency")
    ]
)
def test_filtering_empty_result(
    pattern_tool, category, method
):
    result = pattern_tool.invoke({
        "category_name": category,
        "filtering_method": method,
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert result["patterns"] == []
    assert result["kept_terms"] == []
    assert result["pattern_count"] == 0
    assert result["note"] == (
        "no terms survived filtering"
    )
    assert result["result_id"]


@pytest.mark.parametrize(
    "args",
    [
        {
            "category_name": "catA",
            "filtering_method": "invalid",
            "min_sup": 1
        },
        {
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "invalid",
            "min_sup": 1
        },
        {
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": 0
        },
        {
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": 0
        }
    ]
)
def test_invalid_parameters(pattern_tool, args):
    result = pattern_tool.invoke(args)

    assert "error" in result
    assert isinstance(result["error"], str)
    assert result["error"]


@pytest.mark.parametrize(
    "method,expected_patterns",
    [
        (
            "tfidf",
            [{"items": ["alpha"], "support": 3}]
        ),
        (
            "variance",
            []
        )
    ]
)
def test_session_storage(method, expected_patterns):
    session = SessionState()

    session.dataframe = pd.DataFrame({
        "text": [
            "alpha beta",
            "alpha",
            "alpha",
            "beta"
        ],
        "category_name": ["catA"] * 4
    })

    tools, errors = load_workspace_tools(
        "agent_dev_workspace", session
    )
    assert not errors, errors

    pattern_tool = next(
        t for t in tools
        if t.name == "mine_patterns_tool"
    )

    result = pattern_tool.invoke({
        "category_name": "catA",
        "filtering_method": method,
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    result_id = result["result_id"]
    assert result_id in session.results_store

    stored = session.results_store[result_id]
    assert stored["record"].result_id == result_id
    assert stored["full_report"]["patterns"] == expected_patterns

    artifact_key = f"patterns_catA_{method}_fpgrowth"

    assert artifact_key in session.artifacts
    assert session.artifacts[artifact_key] == expected_patterns
    assert result["patterns"] == expected_patterns
    assert result["pattern_count"] == len(expected_patterns)
