
from typing import Any, Dict

import numpy as np
import pandas as pd
from langchain_core.tools import tool


def make_tools(session):
    """Create feature-filtering tools."""

    @tool
    def variance_filter_tool(
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Calculate term variances and flag low-variance features.

        Uses the existing global document-term matrix.
        Computes population variance (ddof=0) for each term.
        Terms with variance greater than threshold are retained.

        The original DTM is not modified.

        Args:
            threshold: Minimum variance required to retain a term.
                A term is kept when variance > threshold.

        Returns:
            Per-term variances, kept terms, removed terms,
            threshold, and result ID.
        """
        if session.feature_matrix is None:
            return {
                "error": (
                    "No document-term matrix available. "
                    "Run build_dtm_tool first."
                )
            }

        if session.feature_names is None:
            return {
                "error": "Feature names are not available."
            }

        if not np.isfinite(threshold) or threshold < 0:
            return {
                "error": (
                    "threshold must be a finite, non-negative number."
                )
            }

        matrix = session.feature_matrix
        feature_names = list(session.feature_names)

        if matrix.shape[1] != len(feature_names):
            return {
                "error": (
                    "DTM columns do not match feature names."
                )
            }

        if matrix.shape[0] == 0:
            return {
                "error": "Cannot calculate variance on an empty DTM."
            }

        # Population variance: E[X^2] - E[X]^2.
        # Works directly with the existing sparse DTM.
        mean_values = np.asarray(
            matrix.mean(axis=0)
        ).ravel()

        squared_mean_values = np.asarray(
            matrix.multiply(matrix).mean(axis=0)
        ).ravel()

        variances = (
            squared_mean_values - mean_values ** 2
        )

        # Avoid tiny negative values from floating-point rounding.
        variances = np.maximum(variances, 0.0)

        # Create a DataFrame compatible with visualize_result_tool.
        report = pd.DataFrame({
            "term": feature_names,
            "variance": variances.astype(float)
        })

        report = report.sort_values(
            by=["variance", "term"],
            ascending=[False, True]
        ).reset_index(drop=True)

        kept_terms = report.loc[
            report["variance"] > threshold,
            "term"
        ].tolist()

        removed_terms = report.loc[
            report["variance"] <= threshold,
            "term"
        ].tolist()

        variance_by_term = {
            str(term): float(value)
            for term, value in zip(
                feature_names,
                variances
            )
        }

        result_id = session.next_result_id(
            "variance_filter"
        )

        summary = {
            "result_id": result_id,
            "threshold": float(threshold),
            "variances": variance_by_term,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={
                "threshold": float(threshold)
            },
            summary=summary,
            full_report=report,
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Calculate Pearson correlation for each DTM term.

        Uses one-vs-rest labels for the selected target class.
        Retains terms whose absolute correlation exceeds threshold.
        Does not modify the original document-term matrix.
        """
        from scipy.stats import pearsonr

        if session.feature_matrix is None:
            return {'error': 'Run build_dtm_tool first.'}

        if session.feature_names is None or session.labels is None:
            return {'error': 'Feature names or labels unavailable.'}

        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            return {'error': 'threshold must be between 0 and 1.'}

        matrix = session.feature_matrix
        terms = list(session.feature_names)
        labels = np.asarray(session.labels).astype(str)

        if matrix.shape[0] != len(labels):
            return {'error': 'DTM and labels have different lengths.'}

        if matrix.shape[1] != len(terms):
            return {'error': 'DTM and feature names do not match.'}

        if target_class not in set(labels):
            return {'error': 'Unknown target_class: ' + target_class}

        y = (labels == target_class).astype(int)

        if len(np.unique(y)) != 2:
            return {'error': 'One-vs-rest requires two label groups.'}

        correlations = {}

        for i, term in enumerate(terms):
            counts = matrix.getcol(i).toarray().ravel()

            if np.std(counts) == 0:
                r = 0.0
            else:
                r = float(pearsonr(counts, y).statistic)

            correlations[str(term)] = r

        report = pd.DataFrame({
            'term': list(correlations.keys()),
            'pearson_r': list(correlations.values())
        })

        report['_abs_r'] = report['pearson_r'].abs()
        report = report.sort_values(
            by=['_abs_r', 'term'],
            ascending=[False, True]
        ).drop(columns=['_abs_r']).reset_index(drop=True)

        kept_terms = report.loc[
            report['pearson_r'].abs() > threshold, 'term'
        ].tolist()

        removed_terms = report.loc[
            report['pearson_r'].abs() <= threshold, 'term'
        ].tolist()

        result_id = session.next_result_id('pearson_filter')

        summary = {
            'result_id': result_id,
            'target_class': target_class,
            'threshold': float(threshold),
            'correlations': correlations,
            'kept_terms': kept_terms,
            'removed_terms': removed_terms,
            'n_kept': len(kept_terms),
            'n_removed': len(removed_terms)
        }

        session.store_result(
            tool_name='pearson_filter_tool',
            args={
                'target_class': target_class,
                'threshold': float(threshold)
            },
            summary=summary,
            full_report=report
        )

        return summary

    @tool
    def spearman_filter_tool(
            target_class: str,
            threshold: float = 0.0
        ) -> Dict[str, Any]:
            """Calculate Spearman correlation for each DTM term.

            Uses one-vs-rest labels for the selected target class.
            Retains terms whose absolute correlation exceeds threshold.
            Does not modify the original document-term matrix.
            """
            from scipy.stats import spearmanr

            if session.feature_matrix is None:
                return {'error': 'Run build_dtm_tool first.'}

            if session.feature_names is None or session.labels is None:
                return {'error': 'Feature names or labels unavailable.'}

            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                return {'error': 'threshold must be between 0 and 1.'}

            matrix = session.feature_matrix
            terms = list(session.feature_names)
            labels = np.asarray(session.labels).astype(str)

            if matrix.shape[0] != len(labels):
                return {'error': 'DTM and labels have different lengths.'}

            if matrix.shape[1] != len(terms):
                return {'error': 'DTM and feature names do not match.'}

            if target_class not in set(labels):
                return {'error': 'Unknown target_class: ' + target_class}

            y = (labels == target_class).astype(int)

            if len(np.unique(y)) != 2:
                return {'error': 'One-vs-rest requires two label groups.'}

            correlations = {}

            for i, term in enumerate(terms):
                counts = matrix.getcol(i).toarray().ravel()

                if np.std(counts) == 0:
                    r = 0.0
                else:
                    r = float(spearmanr(counts, y).statistic)

                correlations[str(term)] = r

            report = pd.DataFrame({
                'term': list(correlations.keys()),
                'spearman_rho': list(correlations.values())
            })

            report['_abs_r'] = report['spearman_rho'].abs()
            report = report.sort_values(
                by=['_abs_r', 'term'],
                ascending=[False, True]
            ).drop(columns=['_abs_r']).reset_index(drop=True)

            kept_terms = report.loc[
                report['spearman_rho'].abs() > threshold, 'term'
            ].tolist()

            removed_terms = report.loc[
                report['spearman_rho'].abs() <= threshold, 'term'
            ].tolist()

            result_id = session.next_result_id('spearman_filter')

            summary = {
                'result_id': result_id,
                'target_class': target_class,
                'threshold': float(threshold),
                'correlations': correlations,
                'kept_terms': kept_terms,
                'removed_terms': removed_terms,
                'n_kept': len(kept_terms),
                'n_removed': len(removed_terms)
            }

            session.store_result(
                tool_name='spearman_filter_tool',
                args={
                    'target_class': target_class,
                    'threshold': float(threshold)
                },
                summary=summary,
                full_report=report
            )

            return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
