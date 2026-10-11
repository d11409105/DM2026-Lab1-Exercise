
from typing import Any, Dict, List

from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    """Create document-term matrix analysis tools."""

    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1]
    ) -> Dict[str, Any]:
        """Build a global document-term matrix from loaded text.

        Uses CountVectorizer without English stopword removal.
        Stores the sparse matrix, vocabulary, fitted vectorizer,
        and document labels in the shared session.

        Args:
            ngram_range: Minimum and maximum n-gram lengths,
                as a two-element integer list. Default [1, 1]
                creates unigrams. For example, [1, 2] includes
                unigrams and bigrams.

        Returns:
            A summary containing matrix shape, vocabulary,
            non-zero elements, total elements, and sparsity.
        """

        if session.dataframe is None:
            return {
                "error": "No dataset loaded. Run load_dataset_tool first."
            }

        df = session.dataframe

        if "text" not in df.columns:
            return {"error": "'text' column not found."}

        if "category_name" not in df.columns:
            return {"error": "'category_name' column not found."}

        if len(ngram_range) != 2:
            return {
                "error": "ngram_range must contain exactly two integers."
            }

        if (
            any(type(value) is not int for value in ngram_range)
            or ngram_range[0] < 1
            or ngram_range[1] < ngram_range[0]
        ):
            return {
                "error": "Invalid ngram_range. Expected [min_n, max_n]."
            }

        if df.empty:
            return {"error": "Dataset is empty."}

        texts = df["text"].fillna("").astype(str).tolist()

        vectorizer = CountVectorizer(
            ngram_range=tuple(ngram_range)
        )

        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError as exc:
            return {"error": str(exc)}

        feature_names = vectorizer.get_feature_names_out().tolist()

        n_documents, n_terms = matrix.shape

        non_zero = int(matrix.nnz)
        total_elements = int(n_documents * n_terms)

        sparsity_pct = (
            100.0 * (1.0 - non_zero / total_elements)
            if total_elements > 0
            else 0.0
        )

        session.feature_matrix = matrix
        session.feature_names = feature_names

        session.artifacts["count_vectorizer"] = vectorizer

        session.set_labels(
            df["category_name"].astype(str).to_numpy()
        )

        result_id = session.next_result_id("build_dtm")

        summary = {
            "result_id": result_id,
            "n_documents": int(n_documents),
            "n_terms": int(n_terms),
            "shape": [int(n_documents), int(n_terms)],
            "feature_names": feature_names,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": float(sparsity_pct),
            "ngram_range": list(ngram_range),
        }

        session.store_result(
            tool_name="build_dtm_tool",
            args={"ngram_range": list(ngram_range)},
            summary=summary,
            full_report=summary,
        )

        return summary


    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """Compute total term frequencies from the existing global DTM.

        Sums each column of session.feature_matrix and maps the
        resulting counts to session.feature_names. Does not rebuild
        the document-term matrix.

        Returns:
            Per-term frequencies, vocabulary size, and total frequency.
        """
        if session.feature_matrix is None:
            return {
                "error": (
                    "No document-term matrix available. "
                    "Run build_dtm_tool first."
                )
            }

        if session.feature_names is None:
            return {"error": "Feature names are not available."}

        matrix = session.feature_matrix
        feature_names = list(session.feature_names)

        if matrix.shape[1] != len(feature_names):
            return {
                "error": (
                    "DTM column count does not match feature names."
                )
            }

        column_totals = matrix.sum(axis=0).A1

        frequencies = {
            term: int(count)
            for term, count in zip(feature_names, column_totals)
        }

        result_id = session.next_result_id("term_frequency")

        summary = {
            "result_id": result_id,
            "term_frequencies": frequencies,
            "n_terms": len(feature_names),
            "total_frequency": int(column_totals.sum()),
        }

        session.store_result(
            tool_name="term_frequency_tool",
            args={},
            summary=summary,
            full_report=summary,
        )

        return summary


    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """Plot a positional slice of the existing document-term matrix.

        Uses the first n_documents rows and first n_terms columns
        without sorting or filtering. Creates an annotated seaborn
        heatmap and stores its figure in session.pending_figure.

        Args:
            n_terms: Maximum number of vocabulary columns. Default 20.
            n_documents: Maximum number of document rows. Default 20.

        Returns:
            The sliced count matrix, term labels, document labels,
            matrix shape, and stored result ID.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if session.feature_matrix is None:
            return {
                "error": (
                    "No document-term matrix available. "
                    "Run build_dtm_tool first."
                )
            }

        if session.feature_names is None:
            return {"error": "Feature names are not available."}

        if n_terms < 1 or n_documents < 1:
            return {
                "error": (
                    "n_terms and n_documents must be positive integers."
                )
            }

        matrix = session.feature_matrix
        feature_names = list(session.feature_names)

        if matrix.shape[1] != len(feature_names):
            return {
                "error": "DTM columns do not match feature names."
            }

        actual_documents = min(n_documents, matrix.shape[0])
        actual_terms = min(n_terms, matrix.shape[1])

        if actual_documents == 0 or actual_terms == 0:
            return {"error": "Cannot plot an empty DTM slice."}

        # Preserve the original row and column order.
        sliced_matrix = matrix[
            0:actual_documents,
            0:actual_terms
        ].toarray()

        selected_terms = feature_names[0:actual_terms]
        document_labels = list(range(actual_documents))

        # Draw the raw counts, without normalization or reordering.
        fig, ax = plt.subplots(
            figsize=(
                max(7, actual_terms * 0.8),
                max(5, actual_documents * 0.45)
            )
        )

        sns.heatmap(
            sliced_matrix,
            annot=True,
            fmt="g",
            cmap="Blues",
            xticklabels=selected_terms,
            yticklabels=document_labels,
            ax=ax
        )

        ax.set_title("Document-Term Matrix Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")

        fig.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")

        summary = {
            "result_id": result_id,
            "matrix": sliced_matrix.tolist(),
            "term_labels": selected_terms,
            "document_labels": document_labels,
            "shape": [
                int(actual_documents),
                int(actual_terms)
            ],
            "n_documents": int(actual_documents),
            "n_terms": int(actual_terms),
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={
                "n_terms": n_terms,
                "n_documents": n_documents
            },
            summary=summary,
            full_report=summary,
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
