import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools

def test_pca_known_answer():
    s = SessionState()
    df = pd.read_csv('agent_dev/sample_fixture.csv')
    s.dataframe = pd.DataFrame({'text': df['text'], 'category_name': df['label']})
    tools, errors = load_workspace_tools('agent_dev_workspace', s)
    assert not errors, errors
    t = {x.name: x for x in tools}
    assert 'error' not in t['build_dtm_tool'].invoke({})
    r = t['reduce_dimensions_tool'].invoke({'method': 'pca'})
    assert 'error' not in r, r
    expected = PCA(n_components=2, svd_solver='full').fit_transform(s.feature_matrix.toarray())
    np.testing.assert_allclose(r['coordinates'], expected, atol=1e-6)
    assert np.asarray(r['coordinates']).shape == (8, 2)
    assert s.pending_figure is not None
    import matplotlib.pyplot as plt
    plt.close(s.pending_figure)

def test_tsne_output():
    s = SessionState()
    df = pd.read_csv('agent_dev/sample_fixture.csv')
    s.dataframe = pd.DataFrame({'text': df['text'], 'category_name': df['label']})
    tools, errors = load_workspace_tools('agent_dev_workspace', s)
    assert not errors, errors
    t = {x.name: x for x in tools}
    assert 'error' not in t['build_dtm_tool'].invoke({})
    args = {'method': 'tsne', 'perplexity': 2.0, 'random_state': 42}
    r1 = t['reduce_dimensions_tool'].invoke(args)
    assert 'error' not in r1, r1
    a = np.asarray(r1['coordinates'])
    assert a.shape == (8, 2)
    assert np.isfinite(a).all()
    assert r1['explained_variance_ratio'] is None
    assert s.pending_figure is not None
    fig1 = s.pending_figure
    r2 = t['reduce_dimensions_tool'].invoke(args)
    assert 'error' not in r2, r2
    np.testing.assert_allclose(a, r2['coordinates'], atol=1e-5)
    import matplotlib.pyplot as plt
    plt.close(fig1)
    plt.close(s.pending_figure)

def test_umap_output():
    s = SessionState()
    df = pd.read_csv('agent_dev/sample_fixture.csv')
    s.dataframe = pd.DataFrame({'text': df['text'], 'category_name': df['label']})
    tools, errors = load_workspace_tools('agent_dev_workspace', s)
    assert not errors, errors
    t = {x.name: x for x in tools}
    assert 'error' not in t['build_dtm_tool'].invoke({})
    args = {'method': 'umap', 'n_neighbors': 3, 'random_state': 42}
    r1 = t['reduce_dimensions_tool'].invoke(args)
    assert 'error' not in r1, r1
    a = np.asarray(r1['coordinates'])
    assert a.shape == (8, 2)
    assert np.isfinite(a).all()
    assert r1['explained_variance_ratio'] is None
    assert s.pending_figure is not None
    fig1 = s.pending_figure
    r2 = t['reduce_dimensions_tool'].invoke(args)
    assert 'error' not in r2, r2
    np.testing.assert_allclose(a, r2['coordinates'], atol=1e-5)
    import matplotlib.pyplot as plt
    plt.close(fig1)
    plt.close(s.pending_figure)

def test_one_hot_labels():
    s = SessionState()
    df = pd.read_csv('agent_dev/sample_fixture.csv')
    s.dataframe = pd.DataFrame({'category_name': df['label']})
    tools, errors = load_workspace_tools('agent_dev_workspace', s)
    assert not errors, errors
    t = {x.name: x for x in tools}
    r = t['binarize_labels_tool'].invoke({})
    assert 'error' not in r, r
    expected = pd.get_dummies(s.dataframe['category_name'], dtype=int)
    assert r['categories'] == expected.columns.tolist()
    assert r['one_hot_matrix'] == expected.values.tolist()
    assert r['n_documents'] == 8
    assert r['n_classes'] == 2
    assert 'binarized_labels' in s.artifacts
    assert 'result_id' in r

import pytest

@pytest.mark.parametrize("method,options", [
    ("invalid", {}),
    ("tsne", {"perplexity": 8.0}),
    ("umap", {"n_neighbors": 8}),
])
def test_invalid_reduction(method, options):
    s = SessionState()
    df = pd.read_csv("agent_dev/sample_fixture.csv")
    s.dataframe = pd.DataFrame({
        "text": df["text"],
        "category_name": df["label"]
    })
    tools, errors = load_workspace_tools("agent_dev_workspace", s)
    assert not errors, errors
    t = {x.name: x for x in tools}
    assert "error" not in t["build_dtm_tool"].invoke({})
    r = t["reduce_dimensions_tool"].invoke({
        "method": method, **options
    })
    assert "error" in r


def test_one_hot_invalid_column():
    s = SessionState()
    s.dataframe = pd.DataFrame({'category_name': ['A', 'B']})
    tools, errors = load_workspace_tools('agent_dev_workspace', s)
    assert not errors, errors
    t = {x.name: x for x in tools}
    r = t['binarize_labels_tool'].invoke({'label_column': 'unknown_column'})
    assert 'error' in r
    assert 'unknown_column' in r['error']
