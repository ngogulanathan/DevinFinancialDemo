"""Comprehensive unit tests for the load_transactions function."""

import csv
import os

import pytest

from risk_scoring import load_transactions

REQUIRED_FIELDS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
]


def _write_csv(path, headers, rows):
    """Helper to write a CSV file with given headers and rows."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_row(**overrides):
    """Return a valid transaction row dict, with optional overrides."""
    base = {
        "step": "1",
        "type": "PAYMENT",
        "amount": "1000.50",
        "nameOrig": "C100",
        "oldbalanceOrg": "5000.00",
        "newbalanceOrig": "4000.00",
        "nameDest": "M200",
        "oldbalanceDest": "0.00",
        "newbalanceDest": "1000.50",
    }
    base.update(overrides)
    return base


class TestLoadTransactionsValidCSV:
    """Test successfully loading a valid CSV with all required fields."""

    def test_correct_number_of_transactions(self, tmp_path):
        csv_file = tmp_path / "txns.csv"
        rows = [_make_row(), _make_row(step="2", amount="200.00")]
        _write_csv(csv_file, REQUIRED_FIELDS, rows)

        result = load_transactions(str(csv_file))
        assert len(result) == 2

    def test_transaction_id_assignment(self, tmp_path):
        csv_file = tmp_path / "txns.csv"
        rows = [_make_row(), _make_row(), _make_row()]
        _write_csv(csv_file, REQUIRED_FIELDS, rows)

        result = load_transactions(str(csv_file))
        assert result[0]["transaction_id"] == "TXN-0001"
        assert result[1]["transaction_id"] == "TXN-0002"
        assert result[2]["transaction_id"] == "TXN-0003"

    def test_correct_types_after_parsing(self, tmp_path):
        csv_file = tmp_path / "txns.csv"
        _write_csv(csv_file, REQUIRED_FIELDS, [_make_row()])

        result = load_transactions(str(csv_file))
        txn = result[0]
        assert isinstance(txn["amount"], float)
        assert isinstance(txn["step"], int)
        assert isinstance(txn["oldbalanceOrg"], float)
        assert isinstance(txn["newbalanceOrig"], float)
        assert isinstance(txn["oldbalanceDest"], float)
        assert isinstance(txn["newbalanceDest"], float)
        # Non-numeric fields remain strings
        assert isinstance(txn["type"], str)
        assert isinstance(txn["nameOrig"], str)
        assert isinstance(txn["nameDest"], str)


class TestLoadTransactionsMissingField:
    """Test that ValueError is raised when a required field is missing."""

    @pytest.mark.parametrize("missing_field", REQUIRED_FIELDS)
    def test_missing_required_field_raises_value_error(self, tmp_path, missing_field):
        headers = [f for f in REQUIRED_FIELDS if f != missing_field]
        row = _make_row()
        row.pop(missing_field)
        csv_file = tmp_path / "txns.csv"
        _write_csv(csv_file, headers, [row])

        with pytest.raises(ValueError, match=f"Missing required field: {missing_field}"):
            load_transactions(str(csv_file))


class TestLoadTransactionsEmptyCSV:
    """Test handling an empty CSV (headers only, no data rows)."""

    def test_headers_only_returns_empty_list(self, tmp_path):
        csv_file = tmp_path / "txns.csv"
        _write_csv(csv_file, REQUIRED_FIELDS, [])

        result = load_transactions(str(csv_file))
        assert result == []


class TestLoadTransactionsNumericParsing:
    """Verify all numeric fields are correctly parsed."""

    def test_amount_parsed_as_float(self, tmp_path):
        csv_file = tmp_path / "txns.csv"
        _write_csv(csv_file, REQUIRED_FIELDS, [_make_row(amount="12345.67")])

        result = load_transactions(str(csv_file))
        assert result[0]["amount"] == pytest.approx(12345.67)

    def test_step_parsed_as_int(self, tmp_path):
        csv_file = tmp_path / "txns.csv"
        _write_csv(csv_file, REQUIRED_FIELDS, [_make_row(step="42")])

        result = load_transactions(str(csv_file))
        assert result[0]["step"] == 42

    def test_old_balance_org_parsed(self, tmp_path):
        csv_file = tmp_path / "txns.csv"
        _write_csv(csv_file, REQUIRED_FIELDS, [_make_row(oldbalanceOrg="99999.99")])

        result = load_transactions(str(csv_file))
        assert result[0]["oldbalanceOrg"] == pytest.approx(99999.99)

    def test_new_balance_orig_parsed(self, tmp_path):
        csv_file = tmp_path / "txns.csv"
        _write_csv(csv_file, REQUIRED_FIELDS, [_make_row(newbalanceOrig="88888.88")])

        result = load_transactions(str(csv_file))
        assert result[0]["newbalanceOrig"] == pytest.approx(88888.88)

    def test_old_balance_dest_parsed(self, tmp_path):
        csv_file = tmp_path / "txns.csv"
        _write_csv(csv_file, REQUIRED_FIELDS, [_make_row(oldbalanceDest="77777.77")])

        result = load_transactions(str(csv_file))
        assert result[0]["oldbalanceDest"] == pytest.approx(77777.77)

    def test_new_balance_dest_parsed(self, tmp_path):
        csv_file = tmp_path / "txns.csv"
        _write_csv(csv_file, REQUIRED_FIELDS, [_make_row(newbalanceDest="66666.66")])

        result = load_transactions(str(csv_file))
        assert result[0]["newbalanceDest"] == pytest.approx(66666.66)

    def test_zero_values(self, tmp_path):
        csv_file = tmp_path / "txns.csv"
        _write_csv(csv_file, REQUIRED_FIELDS, [
            _make_row(amount="0.0", step="0", oldbalanceOrg="0.0",
                      newbalanceOrig="0.0", oldbalanceDest="0.0", newbalanceDest="0.0")
        ])

        result = load_transactions(str(csv_file))
        txn = result[0]
        assert txn["amount"] == 0.0
        assert txn["step"] == 0
        assert txn["oldbalanceOrg"] == 0.0
        assert txn["newbalanceOrig"] == 0.0
        assert txn["oldbalanceDest"] == 0.0
        assert txn["newbalanceDest"] == 0.0


class TestLoadTransactionsFileNotFound:
    """Test handling a file that does not exist."""

    def test_nonexistent_file_raises_file_not_found_error(self, tmp_path):
        fake_path = str(tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            load_transactions(fake_path)


class TestLoadTransactionsExtraColumns:
    """Verify extra columns in the CSV are preserved in the output dicts."""

    def test_extra_columns_preserved(self, tmp_path):
        extra_headers = REQUIRED_FIELDS + ["isFraud", "isFlaggedFraud"]
        row = _make_row()
        row["isFraud"] = "1"
        row["isFlaggedFraud"] = "0"
        csv_file = tmp_path / "txns.csv"
        _write_csv(csv_file, extra_headers, [row])

        result = load_transactions(str(csv_file))
        txn = result[0]
        assert txn["isFraud"] == "1"
        assert txn["isFlaggedFraud"] == "0"

    def test_extra_columns_do_not_interfere_with_required_fields(self, tmp_path):
        extra_headers = REQUIRED_FIELDS + ["extra_col"]
        row = _make_row()
        row["extra_col"] = "hello"
        csv_file = tmp_path / "txns.csv"
        _write_csv(csv_file, extra_headers, [row])

        result = load_transactions(str(csv_file))
        txn = result[0]
        # Required fields still parsed correctly
        assert isinstance(txn["amount"], float)
        assert isinstance(txn["step"], int)
        assert txn["extra_col"] == "hello"
        assert "transaction_id" in txn
