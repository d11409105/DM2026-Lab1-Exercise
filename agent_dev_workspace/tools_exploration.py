
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from typing import Any, List

from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


def make_tools(session) -> List[Any]:

    @tool
    def cosine_similarity_tool(
        doc_index_a: int,
        doc_index_b: int
    ) -> dict:
        """Calculate cosine similarity between two documents.

        Uses raw document-term counts stored in the session.

        Args:
            doc_index_a: Zero-based index of the first document.
            doc_index_b: Zero-based index of the second document.

        Returns:
            Similarity score and document indices.
        """

        matrix = session.feature_matrix

        if matrix is None:
            return {
                "error": "Document-term matrix is not available."
            }

        n_documents = matrix.shape[0]

        for index in (doc_index_a, doc_index_b):
            if not 0 <= index < n_documents:
                return {
                    "error": (
                        f"Document index {index} is out of range. "
                        f"Valid range: 0 to {n_documents - 1}."
                    )
                }

        vector_a = matrix[doc_index_a]
        vector_b = matrix[doc_index_b]

        similarity = float(
            cosine_similarity(vector_a, vector_b)[0, 0]
        )

        result_id = session.next_result_id(
            "cosine_similarity_tool"
        )

        result = {
            "result_id": result_id,
            "doc_index_a": doc_index_a,
            "doc_index_b": doc_index_b,
            "cosine_similarity": similarity
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={
                "doc_index_a": doc_index_a,
                "doc_index_b": doc_index_b
            },
            summary=result
        )

        return result


    @tool
    def feature_correlation_matrix_tool(
        top_n: int = 20
    ) -> dict:
        """Compute and plot correlations among top-variance terms.

        Args:
            top_n: Maximum number of terms, ranked by variance.

        Returns:
            Selected term names and Pearson correlation matrix.
        """

        matrix = session.feature_matrix
        names = session.feature_names

        if matrix is None or names is None:
            return {"error": "Document-term matrix is unavailable."}

        if top_n < 1:
            return {"error": "top_n must be at least 1."}

        if matrix.shape[1] != len(names):
            return {"error": "Feature names do not match matrix columns."}

        if matrix.shape[0] < 2:
            return {"error": "At least two documents are required."}

        data = (
            matrix.toarray()
            if hasattr(matrix, "toarray")
            else np.asarray(matrix)
        )

        # Calculate population variance for each term
        variances = np.var(data, axis=0)

        # Select top terms by descending variance
        # Stable sorting preserves vocabulary order on ties
        ranked_indices = np.argsort(
            -variances,
            kind="stable"
        )

        selected_indices = ranked_indices[:top_n]
        selected_names = [
            str(names[i]) for i in selected_indices
        ]

        selected_data = data[:, selected_indices]

        if np.any(np.var(selected_data, axis=0) == 0):
            return {
                "error": (
                    "Selected terms include zero-variance columns; "
                    "Pearson correlation is undefined."
                )
            }

        # Pearson feature-vs-feature correlation
        correlations = np.atleast_2d(
            np.corrcoef(selected_data, rowvar=False)
        )

        # Create annotated heatmap
        fig, ax = plt.subplots(
            figsize=(
                max(7, len(selected_names) * 0.65),
                max(5, len(selected_names) * 0.55)
            )
        )

        sns.heatmap(
            correlations,
            xticklabels=selected_names,
            yticklabels=selected_names,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax
        )

        ax.set_title("Feature Correlation Matrix")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Terms")
        fig.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id(
            "feature_correlation_matrix_tool"
        )

        result = {
            "result_id": result_id,
            "top_n": top_n,
            "n_terms": len(selected_names),
            "terms": selected_names,
            "correlation_matrix": correlations.tolist()
        }

        session.artifacts[
            "feature_correlation_matrix"
        ] = correlations

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={"top_n": top_n},
            summary=result
        )

        return result

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
