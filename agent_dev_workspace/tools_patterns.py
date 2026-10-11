from typing import Optional

import numpy as np

import pandas as pd
import tempfile
from pathlib import Path
from PAMI.extras.convert import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth

from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer
)

from langchain_core.tools import tool


def make_tools(session):

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> dict:
        """Mine frequent word patterns from one document category.

        Filtering methods: variance, tfidf, term_frequency.
        Algorithms: fpgrowth, topk, maxfpgrowth.
        """

        valid_filters = {"variance", "tfidf", "term_frequency"}
        valid_algorithms = {"fpgrowth", "topk", "maxfpgrowth"}

        if filtering_method not in valid_filters:
            return {"error": "Invalid filtering_method"}

        if algorithm not in valid_algorithms:
            return {"error": "Invalid algorithm"}

        if algorithm == "topk":
            if k is None or k < 1:
                return {"error": "topk requires positive k"}
        else:
            if min_sup is None or min_sup < 1:
                return {"error": "Mining requires positive min_sup"}


        df = session.dataframe

        if df is None:
            return {"error": "Dataset not loaded"}

        subset = df.loc[
            df["category_name"] == category_name, "text"
        ].fillna("").astype(str)

        if subset.empty:
            return {"error": "Category not found or empty"}

        vectorizer = CountVectorizer(stop_words="english")

        try:
            X = vectorizer.fit_transform(subset)
        except ValueError:
            return {"patterns": [], "note": "empty vocabulary"}

        terms = np.array(vectorizer.get_feature_names_out())

        if filtering_method == "variance":
            scores = np.var(X.toarray(), axis=0)
        elif filtering_method == "term_frequency":
            scores = np.asarray(X.sum(axis=0)).ravel()
        else:
            tfidf = TfidfTransformer().fit_transform(X)
            scores = np.asarray(tfidf.mean(axis=0)).ravel()

        if filtering_method == "tfidf":
            cutoff = np.percentile(scores, 20)
            mask = scores >= cutoff
        else:
            low, high = np.percentile(scores, [5, 95])
            mask = (scores >= low) & (scores <= high)

        kept_terms = terms[mask].tolist()


        if not kept_terms:
            args = {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            }

            result_id = session.next_result_id(
                "mine_patterns_tool"
            )

            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "pattern_count": 0
            }

            artifact_key = (
                f"patterns_{category_name}_"
                f"{filtering_method}_{algorithm}"
            )
            session.artifacts[artifact_key] = []

            session.store_result(
                "mine_patterns_tool",
                args,
                summary,
                full_report={"patterns": []}
            )

            return {
                **summary,
                "patterns": [],
                "kept_terms": [],
                "note": "no terms survived filtering"
            }

        filtered = pd.DataFrame(
            X[:, mask].toarray(),
            columns=kept_terms
        )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "transactions.txt")

            converter = DF2DB.DF2DB(filtered)
            converter.convert2TransactionalDatabase(
                db_path, ">=", 1
            )

            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(
                    db_path, minSup=min_sup
                )
            elif algorithm == "topk":
                miner = FAE.FAE(db_path, k=k)
            else:
                miner = MaxFPGrowth.MaxFPGrowth(
                    db_path, minSup=min_sup
                )

            miner.mine()
            raw_patterns = miner.getPatterns()

        patterns = []

        for items, support in raw_patterns.items():
            if isinstance(items, str):
                terms_in_pattern = items.split()
            else:
                terms_in_pattern = list(items)

            patterns.append({
                "items": terms_in_pattern,
                "support": int(support)
            })

        args = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k
        }

        result_id = session.next_result_id(
            "mine_patterns_tool"
        )

        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "pattern_count": len(patterns)
        }

        artifact_key = (
            f"patterns_{category_name}_"
            f"{filtering_method}_{algorithm}"
        )
        session.artifacts[artifact_key] = patterns

        result_id = session.store_result(
            "mine_patterns_tool",
            args,
            summary,
            full_report={"patterns": patterns}
        )

        return {
            "result_id": result_id,
            **summary,
            "kept_terms": kept_terms,
            "patterns": patterns,
            "note": "mining completed"
        }

    return [mine_patterns_tool]
