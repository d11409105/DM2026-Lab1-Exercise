
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


EXPECTED_TERMS = [
    "alpha", "always", "beta", "delta", "gamma"
]

EXPECTED_FREQUENCIES = {
    "alpha": 6,
    "always": 7,
    "beta": 3,
    "delta": 4,
    "gamma": 7,
}

EXPECTED_MATRIX = [
    [3, 1, 1, 0, 0],
    [2, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 1, 0, 1, 3],
    [0, 1, 0, 1, 2],
    [0, 1, 0, 1, 1],
    [0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0],
]


@pytest.fixture
def dtm_environment():
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

    build_result = tool_map["build_dtm_tool"].invoke({
        "ngram_range": [1, 1]
    })

    assert "error" not in build_result, build_result

    yield session, tool_map, build_result

    if session.pending_figure is not None:
        import matplotlib.pyplot as plt
        plt.close(session.pending_figure)


def test_build_dtm_shape_and_vocabulary(dtm_environment):
    session, _, result = dtm_environment

    assert result["shape"] == [8, 5]
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["feature_names"] == EXPECTED_TERMS
    assert session.feature_matrix.shape == (8, 5)


def test_build_dtm_sparsity(dtm_environment):
    _, _, result = dtm_environment

    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5)


def test_build_dtm_counts(dtm_environment):
    session, _, _ = dtm_environment

    assert (
        session.feature_matrix.toarray().tolist()
        == EXPECTED_MATRIX
    )


def test_term_frequency_known_answer(dtm_environment):
    _, tools, _ = dtm_environment

    result = tools["term_frequency_tool"].invoke({})

    assert "error" not in result, result
    assert result["term_frequencies"] == EXPECTED_FREQUENCIES
    assert result["n_terms"] == 5
    assert result["total_frequency"] == 27


def test_dtm_heatmap_default(dtm_environment):
    session, tools, _ = dtm_environment

    result = tools["dtm_heatmap_tool"].invoke({})

    assert "error" not in result, result
    assert result["shape"] == [8, 5]
    assert result["term_labels"] == EXPECTED_TERMS
    assert result["matrix"] == EXPECTED_MATRIX
    assert session.pending_figure is not None


def test_dtm_heatmap_slice(dtm_environment):
    session, tools, _ = dtm_environment

    result = tools["dtm_heatmap_tool"].invoke({
        "n_terms": 2,
        "n_documents": 3
    })

    assert "error" not in result, result
    assert result["shape"] == [3, 2]
    assert result["term_labels"] == ["alpha", "always"]
    assert result["matrix"] == [
        [3, 1],
        [2, 1],
        [1, 1],
    ]
    assert session.pending_figure is not None
