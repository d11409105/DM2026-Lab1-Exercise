
import numpy as np
import pandas as pd
import pytest

from scipy.stats import pearsonr, spearmanr
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


@pytest.fixture
def filtering_environment():
    session = SessionState()

    df = pd.read_csv("agent_dev/sample_fixture.csv")
    session.dataframe = pd.DataFrame({
        "text": df["text"],
        "category_name": df["label"],
    })

    tools, problems = load_workspace_tools(
        "agent_dev_workspace", session
    )
    assert not problems, problems

    tool_map = {tool.name: tool for tool in tools}

    result = tool_map["build_dtm_tool"].invoke({
        "ngram_range": [1, 1]
    })
    assert "error" not in result, result

    return session, tool_map


def test_variance_known_values(filtering_environment):
    session, tools = filtering_environment

    matrix_before = session.feature_matrix.copy()
    counts = matrix_before.toarray()
    expected = np.var(counts, axis=0, ddof=0)

    result = tools["variance_filter_tool"].invoke({
        "threshold": 0.0
    })

    assert "error" not in result, result

    for term, variance in zip(session.feature_names, expected):
        assert result["variances"][term] == pytest.approx(
            variance, abs=1e-10
        )

    assert result["n_kept"] == 5
    assert result["n_removed"] == 0
    assert (session.feature_matrix != matrix_before).nnz == 0


def test_variance_strict_threshold(filtering_environment):
    _, tools = filtering_environment

    result = tools["variance_filter_tool"].invoke({
        "threshold": 0.0
    })

    for term, variance in result["variances"].items():
        if variance > 0:
            assert term in result["kept_terms"]
        else:
            assert term in result["removed_terms"]


def test_pearson_known_values(filtering_environment):
    session, tools = filtering_environment

    counts = session.feature_matrix.toarray()
    labels = np.asarray(session.labels).astype(str)
    target = labels[0]
    y = (labels == target).astype(int)

    result = tools["pearson_filter_tool"].invoke({
        "target_class": target,
        "threshold": 0.0
    })

    assert "error" not in result, result

    for i, term in enumerate(session.feature_names):
        x = counts[:, i]
        expected = (
            0.0 if np.std(x) == 0
            else float(pearsonr(x, y).statistic)
        )
        assert result["correlations"][term] == pytest.approx(
            expected, abs=1e-10
        )


def test_spearman_known_values(filtering_environment):
    session, tools = filtering_environment

    counts = session.feature_matrix.toarray()
    labels = np.asarray(session.labels).astype(str)
    target = labels[0]
    y = (labels == target).astype(int)

    result = tools["spearman_filter_tool"].invoke({
        "target_class": target,
        "threshold": 0.0
    })

    assert "error" not in result, result

    for i, term in enumerate(session.feature_names):
        x = counts[:, i]
        expected = (
            0.0 if np.std(x) == 0
            else float(spearmanr(x, y).statistic)
        )
        assert result["correlations"][term] == pytest.approx(
            expected, abs=1e-10
        )


@pytest.mark.parametrize(
    "tool_name",
    ["pearson_filter_tool", "spearman_filter_tool"]
)
def test_unknown_target_class(filtering_environment, tool_name):
    _, tools = filtering_environment

    result = tools[tool_name].invoke({
        "target_class": "nonexistent_category",
        "threshold": 0.05
    })

    assert "error" in result


@pytest.mark.parametrize(
    "tool_name",
    [
        "variance_filter_tool",
        "pearson_filter_tool",
        "spearman_filter_tool",
    ]
)
def test_negative_threshold(filtering_environment, tool_name):
    _, tools = filtering_environment

    args = {"threshold": -0.1}

    if tool_name != "variance_filter_tool":
        args["target_class"] = "groupA"

    result = tools[tool_name].invoke(args)

    assert "error" in result
