from typing import Dict, Any, List
from pathlib import Path
from langchain_core.tools import tool
import pandas as pd
import nltk


def _ensure_nltk_punkt():
    """Lazy loader for NLTK tokenizers inside function body rather than import time."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def make_tools(session) -> List[Any]:
    """Factory function that returns tools for tools_data.py bound to session."""

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenize text in session.dataframe into lowercased unigrams.

        Returns:
            Dict containing summary of tokenization process including document count,
            total token count, and sample tokens.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please run load_dataset_tool first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "'text' column not found in session.dataframe."}

        _ensure_nltk_punkt()

        unigrams_list: List[List[str]] = []
        for text in df["text"]:
            if pd.isna(text) or not str(text).strip():
                unigrams_list.append([])
            else:
                tokens = nltk.word_tokenize(str(text).lower())
                unigrams_list.append(tokens)

        df["unigrams"] = unigrams_list
        session.artifacts["unigrams"] = unigrams_list

        total_tokens = sum(len(toks) for toks in unigrams_list)
        result_id = session.next_result_id("tokenize")

        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "total_tokens": total_tokens,
            "sample_unigrams": unigrams_list[:3] if len(unigrams_list) >= 3 else unigrams_list
        }

        session.store_result(
            tool_name="tokenize_tool",
            args={},
            summary=summary,
            full_report={"unigrams": unigrams_list}
        )

        return summary


    @tool
    def list_files_tool(directory: str = ".") -> Dict[str, Any]:
        """List files and folders inside the project repository.

        Args:
            directory: Directory relative to the repository root.
                       Defaults to the repository root.
        """
        repo_root = Path(__file__).resolve().parent.parent
        target = (repo_root / directory).resolve()

        if not target.is_relative_to(repo_root):
            return {"error": "Directory must be inside the repository."}

        if not target.exists():
            return {"error": f"Directory does not exist: {directory}"}

        if not target.is_dir():
            return {"error": f"Not a directory: {directory}"}

        entries = sorted(target.iterdir(), key=lambda p: p.name.lower())

        files = [p.name for p in entries if p.is_file()]
        folders = [p.name for p in entries if p.is_dir()]

        result_id = session.next_result_id("list_files")

        summary = {
            "result_id": result_id,
            "directory": str(target.relative_to(repo_root)),
            "files": files,
            "folders": folders,
            "n_files": len(files),
            "n_folders": len(folders),
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"directory": directory},
            summary=summary,
            full_report=summary,
        )

        return summary


    @tool
    def inspect_data_tool(n_rows: int = 5) -> Dict[str, Any]:
        """Preview the first n_rows of the working DataFrame."""
        if session.dataframe is None:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        if n_rows < 1:
            return {"error": "n_rows must be a positive integer."}

        df = session.dataframe
        preview = df.head(n_rows).copy()

        # Convert values into JSON-friendly representations.
        preview = preview.astype(object)
        preview = preview.map(
            lambda value: None
            if value is None or (
                not isinstance(value, (list, dict, tuple))
                and bool(pd.isna(value))
            )
            else value
        )

        result_id = session.next_result_id("inspect_data")

        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "n_rows_shown": len(preview),
            "rows": preview.to_dict(orient="records"),
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"n_rows": n_rows},
            summary=summary,
            full_report=summary,
        )

        return summary


    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Count missing or blank text values in the working DataFrame."""
        if session.dataframe is None:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        df = session.dataframe

        if "text" not in df.columns:
            return {"error": "'text' column not found in session.dataframe."}

        missing_mask = df["text"].apply(
            lambda value: (
                pd.isna(value)
                if not isinstance(value, (list, dict, tuple))
                else False
            ) or (
                isinstance(value, str) and not value.strip()
            )
        )

        missing_indices = [
            int(position)
            for position, is_missing in enumerate(missing_mask)
            if is_missing
        ]

        result_id = session.next_result_id("check_missing")

        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "missing_count": len(missing_indices),
            "missing_indices": missing_indices,
        }

        session.store_result(
            tool_name="check_missing_tool",
            args={},
            summary=summary,
            full_report=summary,
        )

        return summary


    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Check duplicate text rows and optionally remove all copies.

        Args:
            drop: If True, remove every row in a duplicate group
                  using keep=False.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        df = session.dataframe

        if "text" not in df.columns:
            return {"error": "'text' column not found in session.dataframe."}

        duplicate_extra = df["text"].duplicated(keep="first")
        duplicate_all = df["text"].duplicated(keep=False)

        duplicate_indices = [
            int(i)
            for i, flag in enumerate(duplicate_all)
            if flag
        ]

        original_count = len(df)
        duplicate_count = int(duplicate_extra.sum())

        if drop:
            session.dataframe = df.loc[~duplicate_all].copy()

        remaining_count = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")

        summary = {
            "result_id": result_id,
            "n_documents_before": original_count,
            "duplicate_count": duplicate_count,
            "duplicate_indices": duplicate_indices,
            "drop": drop,
            "n_documents_after": remaining_count,
        }

        session.store_result(
            tool_name="check_duplicates_tool",
            args={"drop": drop},
            summary=summary,
            full_report=summary,
        )

        return summary


    @tool
    def sample_data_tool(
        n: int = 5,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Randomly subsample the working DataFrame without replacement.

        Args:
            n: Number of documents to retain.
            random_state: Seed for reproducible random sampling.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        df = session.dataframe
        original_count = len(df)

        if n < 1:
            return {"error": "n must be a positive integer."}

        if n > original_count:
            return {
                "error": (
                    f"Requested {n} rows, but only "
                    f"{original_count} rows are available."
                )
            }

        sampled_df = df.sample(
            n=n,
            replace=False,
            random_state=random_state
        ).copy()

        session.dataframe = sampled_df

        result_id = session.next_result_id("sample_data")

        summary = {
            "result_id": result_id,
            "n_documents_before": original_count,
            "n_documents_after": len(sampled_df),
            "n_requested": n,
            "random_state": random_state,
            "sampled_indices": [
                int(i) for i in sampled_df.index
            ],
        }

        session.store_result(
            tool_name="sample_data_tool",
            args={
                "n": n,
                "random_state": random_state
            },
            summary=summary,
            full_report=summary,
        )

        return summary


    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Describe document character lengths and create a box plot."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if session.dataframe is None:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        df = session.dataframe

        if "text" not in df.columns:
            return {"error": "'text' column not found."}

        if "category_name" not in df.columns:
            return {"error": "'category_name' column not found."}

        if df.empty:
            return {"error": "Dataset is empty."}

        # Master's contract: count characters, not words.
        text_length = df["text"].fillna("").astype(str).apply(len)

        # Work on a copy to preserve the original DataFrame.
        plot_df = df[["category_name"]].copy()
        plot_df["text_length"] = text_length

        stat_keys = [
            "count", "mean", "std", "min",
            "25%", "50%", "75%", "max"
        ]

        def clean_stats(series):
            stats = series.describe()
            return {
                key: float(stats[key])
                for key in stat_keys
            }

        overall = clean_stats(text_length)

        by_category = {
            str(category): clean_stats(group["text_length"])
            for category, group in plot_df.groupby("category_name")
        }

        # Build the required category-wise box plot.
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.boxplot(
            data=plot_df,
            x="category_name",
            y="text_length",
            ax=ax
        )

        ax.set_title("Document Text Length by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Text Length (characters)")
        fig.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")

        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "overall": overall,
            "by_category": by_category,
            "text_lengths": [
                int(value) for value in text_length
            ],
            "plot_created": True,
        }

        session.store_result(
            tool_name="describe_data_tool",
            args={},
            summary=summary,
            full_report=summary,
        )

        return summary

    return [tokenize_tool, list_files_tool, inspect_data_tool, check_missing_tool, check_duplicates_tool, sample_data_tool, describe_data_tool]
