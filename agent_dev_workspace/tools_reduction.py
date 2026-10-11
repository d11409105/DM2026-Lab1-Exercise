import pandas as pd

from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def make_tools(session) -> List[Any]:

    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> Dict[str, Any]:
        """Reduce the global DTM to two dimensions and plot categories."""

        if session.feature_matrix is None:
            return {"error": "Build the DTM first."}

        if session.dataframe is None:
            return {"error": "Load the dataset first."}

        method = method.lower()

        if method not in ("pca", "tsne", "umap"):
            return {
                "error": "Supported methods: pca, tsne, umap."
            }

        matrix = session.feature_matrix
        n_documents, n_features = matrix.shape

        if n_documents < 2 or n_features < 2:
            return {
                "error": "PCA requires at least 2 documents and 2 features."
            }

        if len(session.dataframe) != n_documents:
            return {
                "error": "DTM rows do not match dataframe rows."
            }

        if hasattr(matrix, "toarray"):
            X = matrix.toarray()
        else:
            X = np.asarray(matrix)

        if method == "pca":
            model = PCA(n_components=2, svd_solver="full")
            coordinates = model.fit_transform(X)
            explained_variance_ratio = (
                model.explained_variance_ratio_.tolist()
            )

        elif method == "tsne":
            if not 0 < perplexity < n_documents:
                return {
                    "error": (
                        "t-SNE perplexity must be greater than 0 "
                        "and smaller than the number of documents."
                    )
                }

            model = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=random_state,
                init="pca",
                learning_rate="auto"
            )
            coordinates = model.fit_transform(X)
            explained_variance_ratio = None

        elif method == "umap":
            if not 2 <= n_neighbors < n_documents:
                return {
                    "error": (
                        "UMAP n_neighbors must be at least 2 "
                        "and smaller than the number of documents."
                    )
                }

            model = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            coordinates = model.fit_transform(X)
            explained_variance_ratio = None

        if "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].astype(str)
        elif "category" in session.dataframe.columns:
            labels = session.dataframe["category"].astype(str)
        else:
            return {"error": "Category labels are missing."}

        fig, ax = plt.subplots(figsize=(8, 6))

        for category in labels.unique():
            mask = (labels == category).to_numpy()
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                label=category,
                alpha=0.8
            )

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(f"{method.upper()}: Document Dimension Reduction")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id(
            "reduce_dimensions_tool"
        )

        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": 2,
            "n_documents": int(n_documents),
            "coordinates": coordinates.tolist(),
            "explained_variance_ratio": explained_variance_ratio
        }

        session.artifacts["reduced_coordinates"] = coordinates

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors
            },
            summary=summary,
            full_report=summary
        )

        return summary


    @tool
    def binarize_labels_tool(
        label_column: str = "category_name"
    ) -> dict:
        """One-hot encode document category labels.

        Args:
            label_column: DataFrame column containing category labels.

        Returns:
            Category names, one-hot matrix, and dimensions.
        """
        df = session.dataframe

        if df is None or df.empty:
            return {"error": "No dataset loaded."}

        if label_column not in df.columns:
            return {
                "error": f"Column not found: {label_column}"
            }

        labels = df[label_column]

        if labels.isna().any():
            return {
                "error": "Category labels contain missing values."
            }

        encoded = pd.get_dummies(
            labels,
            dtype=int
        )

        categories = encoded.columns.tolist()
        matrix = encoded.to_numpy(dtype=int)

        result = {
            "label_column": label_column,
            "n_documents": int(matrix.shape[0]),
            "n_classes": int(matrix.shape[1]),
            "categories": categories,
            "one_hot_matrix": matrix.tolist()
        }

        session.artifacts["binarized_labels"] = encoded

        result_id = session.next_result_id(
            "binarize_labels_tool"
        )

        result["result_id"] = result_id

        session.store_result(
            tool_name="binarize_labels_tool",
            args={
                "label_column": label_column
            },
            summary=result
        )

        return result

    return [reduce_dimensions_tool, binarize_labels_tool]
