"""Unit tests for the print_summary function in risk_scoring.py."""

import pytest

from risk_scoring import print_summary


def _make_transaction(risk_category):
    """Helper to create a minimal scored transaction dict."""
    return {"risk_category": risk_category}


class TestPrintSummaryMixedCategories:
    """Test with a mix of LOW, MEDIUM, and HIGH risk categories."""

    def test_mixed_categories_counts(self, capsys):
        transactions = [
            _make_transaction("LOW"),
            _make_transaction("LOW"),
            _make_transaction("MEDIUM"),
            _make_transaction("HIGH"),
            _make_transaction("HIGH"),
            _make_transaction("HIGH"),
        ]
        print_summary(transactions)
        captured = capsys.readouterr().out

        assert "Total transactions scored: 6" in captured
        assert "LOW  risk (score < 40):   2" in captured
        assert "MEDIUM risk (40-70):      1" in captured
        assert "HIGH risk (> 70):         3" in captured


class TestPrintSummaryAllLow:
    """Test when all transactions are LOW risk."""

    def test_all_low(self, capsys):
        transactions = [_make_transaction("LOW") for _ in range(4)]
        print_summary(transactions)
        captured = capsys.readouterr().out

        assert "Total transactions scored: 4" in captured
        assert "LOW  risk (score < 40):   4" in captured
        assert "MEDIUM risk (40-70):      0" in captured
        assert "HIGH risk (> 70):         0" in captured


class TestPrintSummaryAllHigh:
    """Test when all transactions are HIGH risk."""

    def test_all_high(self, capsys):
        transactions = [_make_transaction("HIGH") for _ in range(5)]
        print_summary(transactions)
        captured = capsys.readouterr().out

        assert "Total transactions scored: 5" in captured
        assert "LOW  risk (score < 40):   0" in captured
        assert "MEDIUM risk (40-70):      0" in captured
        assert "HIGH risk (> 70):         5" in captured


class TestPrintSummaryEmptyList:
    """Test with an empty transaction list."""

    def test_empty_list(self, capsys):
        print_summary([])
        captured = capsys.readouterr().out

        assert "Total transactions scored: 0" in captured
        assert "LOW  risk (score < 40):   0" in captured
        assert "MEDIUM risk (40-70):      0" in captured
        assert "HIGH risk (> 70):         0" in captured


class TestPrintSummarySingleTransaction:
    """Test with a single transaction."""

    def test_single_transaction(self, capsys):
        transactions = [_make_transaction("MEDIUM")]
        print_summary(transactions)
        captured = capsys.readouterr().out

        assert "Total transactions scored: 1" in captured


class TestPrintSummaryOutputFormat:
    """Test that the output contains expected header and separator lines."""

    def test_output_contains_header(self, capsys):
        print_summary([])
        captured = capsys.readouterr().out

        assert "TRANSACTION RISK SCORING SUMMARY" in captured
        assert "=" * 60 in captured

    def test_separator_lines_count(self, capsys):
        print_summary([])
        captured = capsys.readouterr().out

        # The separator line appears three times in the output
        assert captured.count("=" * 60) == 3


class TestPrintSummaryTotalCount:
    """Test that the total count matches the length of the input list."""

    def test_total_matches_input_length(self, capsys):
        for size in [0, 1, 7, 25]:
            transactions = [_make_transaction("LOW") for _ in range(size)]
            print_summary(transactions)
            captured = capsys.readouterr().out
            assert f"Total transactions scored: {size}" in captured
