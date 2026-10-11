
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


@pytest.fixture
def data_environment():
    session = SessionState()

    session.dataframe = pd.DataFrame({
        "text": [
            "Apple banana",
            "Banana cherry",
            "Apple banana",
            "",
            None,
        ],
        "category_name": [
            "A", "B", "A", "B", "A"
        ],
    })

    tools, problems = load_workspace_tools(
        "agent_dev_workspace", session
    )
    assert not problems, problems

    tool_map = {tool.name: tool for tool in tools}

    yield session, tool_map

    if session.pending_figure is not None:
        import matplotlib.pyplot as plt
        plt.close(session.pending_figure)


def test_tokenize(data_environment):
    session, tools = data_environment

    result = tools["tokenize_tool"].invoke({})

    assert "error" not in result, result
    assert result["n_documents"] == 5
    assert result["total_tokens"] >= 4
    assert "unigrams" in session.artifacts
    assert len(session.artifacts["unigrams"]) == 5


def test_list_files(data_environment):
    _, tools = data_environment

    result = tools["list_files_tool"].invoke({
        "directory": "agent_dev"
    })

    assert "error" not in result, result
    assert result["n_files"] == len(result["files"])
    assert result["n_folders"] == len(result["folders"])


def test_inspect_data(data_environment):
    _, tools = data_environment

    result = tools["inspect_data_tool"].invoke({
        "n_rows": 2
    })

    assert "error" not in result, result
    assert result["n_documents"] == 5
    assert result["n_rows_shown"] == 2
    assert len(result["rows"]) == 2


def test_check_missing(data_environment):
    _, tools = data_environment

    result = tools["check_missing_tool"].invoke({})

    assert "error" not in result, result
    assert result["n_documents"] == 5
    assert result["missing_count"] == 2
    assert len(result["missing_indices"]) == 2


def test_check_duplicates_without_drop(data_environment):
    session, tools = data_environment

    result = tools["check_duplicates_tool"].invoke({
        "drop": False
    })

    assert "error" not in result, result
    assert result["n_documents_before"] == 5
    assert result["duplicate_count"] == 1
    assert result["n_documents_after"] == 5
    assert len(session.dataframe) == 5


def test_check_duplicates_with_drop(data_environment):
    session, tools = data_environment

    result = tools["check_duplicates_tool"].invoke({
        "drop": True
    })

    assert "error" not in result, result
    assert result["n_documents_before"] == 5
    assert result["n_documents_after"] < 5
    assert len(session.dataframe) == result["n_documents_after"]


def test_sample_data(data_environment):
    session, tools = data_environment

    result = tools["sample_data_tool"].invoke({
        "n": 3,
        "random_state": 42
    })

    assert "error" not in result, result
    assert result["n_documents_before"] == 5
    assert result["n_documents_after"] == 3
    assert len(session.dataframe) == 3
    assert len(result["sampled_indices"]) == 3


def test_describe_data(data_environment):
    session, tools = data_environment

    result = tools["describe_data_tool"].invoke({})

    assert "error" not in result, result
    assert result["n_documents"] == 5
    assert result["plot_created"] is True
    assert session.pending_figure is not None
    assert "overall" in result
    assert "by_category" in result


def test_inspect_data_without_dataset():
    session = SessionState()

    tools, problems = load_workspace_tools(
        "agent_dev_workspace", session
    )
    assert not problems, problems

    tool_map = {tool.name: tool for tool in tools}

    result = tool_map["inspect_data_tool"].invoke({})

    assert "error" in result


def test_tokenize_without_dataset():
    session = SessionState()

    tools, problems = load_workspace_tools(
        "agent_dev_workspace", session
    )
    assert not problems, problems

    tool_map = {tool.name: tool for tool in tools}

    result = tool_map["tokenize_tool"].invoke({})

    assert "error" in result
