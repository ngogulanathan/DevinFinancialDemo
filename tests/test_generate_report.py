"""Comprehensive unit tests for the generate_report function."""

import csv
import os

import pytest

from risk_scoring import generate_report


EXPECTED_FIELDNAMES = [
    "transaction_id",
    "amount",
    "type",
    "risk_score",
    "risk_category",
    "triggered_rules",
]


def _read_csv(path):
    """Helper to read a CSV file and return (fieldnames, rows)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames, rows


def _sample_transaction(**overrides):
    """Return a single scored-transaction dict with sensible defaults."""
    txn = {
        "transaction_id": "TXN-0001",
        "amount": 5000.0,
        "type": "PAYMENT",
        "risk_score": 20,
        "risk_category": "LOW",
        "triggered_rules": "new_destination",
    }
    txn.update(overrides)
    return txn


# ---------- 1. Basic report generation ----------

class TestBasicReportGeneration:
    """Pass a list of scored transaction dicts and verify the CSV file is
    created with correct headers and data."""

    def test_file_is_created(self, tmp_path):
        out = tmp_path / "report.csv"
        generate_report([_sample_transaction()], str(out))
        assert out.exists()

    def test_headers_present(self, tmp_path):
        out = tmp_path / "report.csv"
        generate_report([_sample_transaction()], str(out))
        fieldnames, _ = _read_csv(str(out))
        assert fieldnames == EXPECTED_FIELDNAMES

    def test_single_row_written(self, tmp_path):
        out = tmp_path / "report.csv"
        txn = _sample_transaction()
        generate_report([txn], str(out))
        _, rows = _read_csv(str(out))
        assert len(rows) == 1
        assert rows[0]["transaction_id"] == txn["transaction_id"]
        assert rows[0]["amount"] == str(txn["amount"])
        assert rows[0]["type"] == txn["type"]
        assert rows[0]["risk_score"] == str(txn["risk_score"])
        assert rows[0]["risk_category"] == txn["risk_category"]
        assert rows[0]["triggered_rules"] == txn["triggered_rules"]


# ---------- 2. Empty transactions ----------

class TestEmptyTransactions:
    """Pass an empty list — should create a CSV with headers only."""

    def test_empty_list_creates_file(self, tmp_path):
        out = tmp_path / "empty.csv"
        generate_report([], str(out))
        assert out.exists()

    def test_empty_list_has_headers(self, tmp_path):
        out = tmp_path / "empty.csv"
        generate_report([], str(out))
        fieldnames, rows = _read_csv(str(out))
        assert fieldnames == EXPECTED_FIELDNAMES
        assert rows == []


# ---------- 3. Multiple rows ----------

class TestMultipleRows:
    """Verify all rows are written correctly when multiple scored
    transactions are provided."""

    def test_row_count_matches(self, tmp_path):
        out = tmp_path / "multi.csv"
        txns = [
            _sample_transaction(transaction_id="TXN-0001", amount=100.0,
                                risk_score=10, risk_category="LOW"),
            _sample_transaction(transaction_id="TXN-0002", amount=50000.0,
                                type="TRANSFER", risk_score=70,
                                risk_category="MEDIUM"),
            _sample_transaction(transaction_id="TXN-0003", amount=99999.99,
                                type="CASH_OUT", risk_score=95,
                                risk_category="HIGH"),
        ]
        generate_report(txns, str(out))
        _, rows = _read_csv(str(out))
        assert len(rows) == 3

    def test_each_row_matches_input(self, tmp_path):
        out = tmp_path / "multi.csv"
        txns = [
            _sample_transaction(transaction_id=f"TXN-{i:04d}", amount=i * 1000.0)
            for i in range(1, 6)
        ]
        generate_report(txns, str(out))
        _, rows = _read_csv(str(out))
        for txn, row in zip(txns, rows):
            assert row["transaction_id"] == txn["transaction_id"]
            assert row["amount"] == str(txn["amount"])


# ---------- 4. Field ordering ----------

class TestFieldOrdering:
    """Verify the CSV columns are in the correct order."""

    def test_column_order_via_fieldnames(self, tmp_path):
        out = tmp_path / "order.csv"
        generate_report([_sample_transaction()], str(out))
        fieldnames, _ = _read_csv(str(out))
        assert fieldnames == EXPECTED_FIELDNAMES

    def test_column_order_via_raw_header_line(self, tmp_path):
        """Double-check by reading the raw first line of the file."""
        out = tmp_path / "order.csv"
        generate_report([_sample_transaction()], str(out))
        with open(str(out), encoding="utf-8") as f:
            header_line = f.readline().strip()
        assert header_line == ",".join(EXPECTED_FIELDNAMES)


# ---------- 5. Data integrity ----------

class TestDataIntegrity:
    """Read back the generated CSV and verify values match input exactly."""

    def test_all_fields_roundtrip(self, tmp_path):
        out = tmp_path / "integrity.csv"
        txn = _sample_transaction(
            transaction_id="TXN-0042",
            amount=12345.67,
            type="TRANSFER",
            risk_score=65,
            risk_category="MEDIUM",
            triggered_rules="high_amount; risky_type; new_destination",
        )
        generate_report([txn], str(out))
        _, rows = _read_csv(str(out))
        row = rows[0]
        assert row["transaction_id"] == "TXN-0042"
        assert row["amount"] == "12345.67"
        assert row["type"] == "TRANSFER"
        assert row["risk_score"] == "65"
        assert row["risk_category"] == "MEDIUM"
        assert row["triggered_rules"] == "high_amount; risky_type; new_destination"

    def test_numeric_values_preserved(self, tmp_path):
        """Ensure numeric fields can be converted back to their original types."""
        out = tmp_path / "integrity_num.csv"
        txn = _sample_transaction(amount=99999.99, risk_score=100)
        generate_report([txn], str(out))
        _, rows = _read_csv(str(out))
        assert float(rows[0]["amount"]) == 99999.99
        assert int(rows[0]["risk_score"]) == 100


# ---------- 6. Overwrite existing file ----------

class TestOverwriteExistingFile:
    """If the output file already exists, it should be overwritten."""

    def test_overwrite_replaces_content(self, tmp_path):
        out = tmp_path / "overwrite.csv"
        # Write an initial report with 3 rows
        initial = [_sample_transaction(transaction_id=f"OLD-{i}") for i in range(3)]
        generate_report(initial, str(out))

        # Overwrite with a single-row report
        replacement = [_sample_transaction(transaction_id="NEW-0001")]
        generate_report(replacement, str(out))

        _, rows = _read_csv(str(out))
        assert len(rows) == 1
        assert rows[0]["transaction_id"] == "NEW-0001"

    def test_overwrite_updates_headers(self, tmp_path):
        """Headers should still be correct after overwrite."""
        out = tmp_path / "overwrite.csv"
        generate_report([_sample_transaction()], str(out))
        generate_report([], str(out))
        fieldnames, rows = _read_csv(str(out))
        assert fieldnames == EXPECTED_FIELDNAMES
        assert rows == []


# ---------- 7. Special characters in triggered_rules ----------

class TestSpecialCharactersInTriggeredRules:
    """Test that semicolon-separated rules like 'high_amount; risky_type'
    are written correctly."""

    def test_semicolon_separated_rules(self, tmp_path):
        out = tmp_path / "special.csv"
        txn = _sample_transaction(
            triggered_rules="high_amount; risky_type",
        )
        generate_report([txn], str(out))
        _, rows = _read_csv(str(out))
        assert rows[0]["triggered_rules"] == "high_amount; risky_type"

    def test_multiple_semicolons(self, tmp_path):
        out = tmp_path / "special2.csv"
        rules = "high_amount; risky_type; new_destination; rapid_transactions"
        txn = _sample_transaction(triggered_rules=rules)
        generate_report([txn], str(out))
        _, rows = _read_csv(str(out))
        assert rows[0]["triggered_rules"] == rules

    def test_no_rules_string(self, tmp_path):
        out = tmp_path / "special3.csv"
        txn = _sample_transaction(triggered_rules="none")
        generate_report([txn], str(out))
        _, rows = _read_csv(str(out))
        assert rows[0]["triggered_rules"] == "none"
