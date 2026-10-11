import pytest
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools

@pytest.mark.parametrize('a,b', [
    (0, 1), (0, 3), (3, 4), (0, 7)
])
def test_cosine_similarity_known_answer(a, b):
    s = SessionState()
    df = pd.read_csv('agent_dev/sample_fixture.csv')
    s.dataframe = pd.DataFrame({'text': df['text'], 'category_name': df['label']})
    tools, errors = load_workspace_tools('agent_dev_workspace', s)
    assert not errors, errors
    t = {x.name: x for x in tools}
    assert 'error' not in t['build_dtm_tool'].invoke({})
    r = t['cosine_similarity_tool'].invoke({'doc_index_a': a, 'doc_index_b': b})
    assert 'error' not in r, r
    expected = cosine_similarity(s.feature_matrix[a], s.feature_matrix[b])[0, 0]
    np.testing.assert_allclose(r['cosine_similarity'], expected, atol=1e-6)
    assert 'result_id' in r

def test_feature_correlation_matrix():
    s = SessionState()
    df = pd.read_csv('agent_dev/sample_fixture.csv')
    s.dataframe = pd.DataFrame({'text': df['text'], 'category_name': df['label']})
    tools, errors = load_workspace_tools('agent_dev_workspace', s)
    assert not errors, errors
    t = {x.name: x for x in tools}
    assert 'error' not in t['build_dtm_tool'].invoke({})
    r = t['feature_correlation_matrix_tool'].invoke({'top_n': 5})
    assert 'error' not in r, r
    a = np.asarray(r['correlation_matrix'])
    assert a.shape == (5, 5)
    np.testing.assert_allclose(a, a.T, atol=1e-6)
    np.testing.assert_allclose(np.diag(a), np.ones(5), atol=1e-6)
    assert len(r['terms']) == 5
    expected_terms = ['alpha', 'gamma', 'delta', 'beta', 'always']
    assert r['terms'] == expected_terms
    names = list(s.feature_names)
    indices = [names.index(term) for term in expected_terms]
    X = s.feature_matrix.toarray()[:, indices]
    expected_corr = np.corrcoef(X, rowvar=False)
    np.testing.assert_allclose(a, expected_corr, atol=1e-6)
    assert s.pending_figure is not None
    import matplotlib.pyplot as plt
    plt.close(s.pending_figure)
