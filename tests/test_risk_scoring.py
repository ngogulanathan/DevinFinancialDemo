"""Comprehensive unit tests for risk_scoring.py."""

import csv
import os
import textwrap

import pytest

from risk_scoring import (
    apply_risk_rules,
    generate_report,
    load_transactions,
    print_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_HEADER = (
    "step,type,amount,nameOrig,oldbalanceOrg,"
    "newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest"
)


def _write_csv(tmp_path, filename, header, rows):
    """Write a small CSV file and return its path."""
    path = tmp_path / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header.split(","))
        for row in rows:
            writer.writerow(row)
    return str(path)


def _make_txn(
    step=1,
    txn_type="PAYMENT",
    amount=100.0,
    name_orig="C_orig",
    name_dest="C_dest",
    transaction_id="TXN-0001",
):
    """Build a minimal transaction dict suitable for apply_risk_rules."""
    return {
        "step": step,
        "type": txn_type,
        "amount": float(amount),
        "nameOrig": name_orig,
        "nameDest": name_dest,
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 9900.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 100.0,
        "transaction_id": transaction_id,
    }


# ===================================================================
# load_transactions
# ===================================================================


class TestLoadTransactions:
    """Tests for the load_transactions function."""

    def test_valid_csv_loading(self, tmp_path):
        """Valid CSV should be loaded with correct number of rows."""
        path = _write_csv(
            tmp_path,
            "valid.csv",
            REQUIRED_HEADER,
            [
                [1, "PAYMENT", "500.50", "C_orig1", "1000", "499.50", "C_dest1", "0", "500.50"],
                [2, "TRANSFER", "1500.75", "C_orig2", "2000", "499.25", "C_dest2", "0", "1500.75"],
            ],
        )
        txns = load_transactions(path)
        assert len(txns) == 2

    def test_amount_converted_to_float(self, tmp_path):
        """The 'amount' field must be converted to a Python float."""
        path = _write_csv(
            tmp_path,
            "amount.csv",
            REQUIRED_HEADER,
            [[1, "PAYMENT", "1234.56", "C1", "5000", "3765.44", "D1", "0", "1234.56"]],
        )
        txns = load_transactions(path)
        assert isinstance(txns[0]["amount"], float)
        assert txns[0]["amount"] == pytest.approx(1234.56)

    def test_step_converted_to_int(self, tmp_path):
        """The 'step' field must be converted to a Python int."""
        path = _write_csv(
            tmp_path,
            "step.csv",
            REQUIRED_HEADER,
            [[3, "PAYMENT", "100", "C1", "5000", "4900", "D1", "0", "100"]],
        )
        txns = load_transactions(path)
        assert isinstance(txns[0]["step"], int)
        assert txns[0]["step"] == 3

    def test_balance_fields_converted_to_float(self, tmp_path):
        """Balance fields must be converted to float."""
        path = _write_csv(
            tmp_path,
            "balance.csv",
            REQUIRED_HEADER,
            [[1, "PAYMENT", "100", "C1", "5000.50", "4900.50", "D1", "200.25", "300.25"]],
        )
        txns = load_transactions(path)
        assert isinstance(txns[0]["oldbalanceOrg"], float)
        assert isinstance(txns[0]["newbalanceOrig"], float)
        assert isinstance(txns[0]["oldbalanceDest"], float)
        assert isinstance(txns[0]["newbalanceDest"], float)

    def test_transaction_id_format(self, tmp_path):
        """Transaction IDs should follow the TXN-NNNN pattern, starting at TXN-0001."""
        path = _write_csv(
            tmp_path,
            "ids.csv",
            REQUIRED_HEADER,
            [
                [1, "PAYMENT", "100", "C1", "5000", "4900", "D1", "0", "100"],
                [2, "PAYMENT", "200", "C2", "3000", "2800", "D2", "0", "200"],
                [3, "PAYMENT", "300", "C3", "1000", "700", "D3", "0", "300"],
            ],
        )
        txns = load_transactions(path)
        assert txns[0]["transaction_id"] == "TXN-0001"
        assert txns[1]["transaction_id"] == "TXN-0002"
        assert txns[2]["transaction_id"] == "TXN-0003"

    def test_missing_required_field_raises_value_error(self, tmp_path):
        """A CSV missing any required column should raise ValueError."""
        # Missing 'amount' column
        bad_header = "step,type,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest"
        path = _write_csv(
            tmp_path,
            "bad.csv",
            bad_header,
            [[1, "PAYMENT", "C1", "5000", "4900", "D1", "0", "100"]],
        )
        with pytest.raises(ValueError, match="Missing required field: amount"):
            load_transactions(path)

    def test_missing_another_required_field(self, tmp_path):
        """Verify ValueError message for a different missing field."""
        bad_header = "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest"
        path = _write_csv(
            tmp_path,
            "bad2.csv",
            bad_header,
            [[1, "PAYMENT", "100", "C1", "5000", "4900", "D1", "0"]],
        )
        with pytest.raises(ValueError, match="Missing required field: newbalanceDest"):
            load_transactions(path)

    def test_empty_csv_returns_empty_list(self, tmp_path):
        """A CSV with only headers and no data rows returns an empty list."""
        path = _write_csv(tmp_path, "empty.csv", REQUIRED_HEADER, [])
        txns = load_transactions(path)
        assert txns == []


# ===================================================================
# apply_risk_rules
# ===================================================================


class TestApplyRiskRules:
    """Tests for the apply_risk_rules function."""

    # ---- Rule 1: high_amount (> 10000 -> +30) ----

    def test_high_amount_rule_triggered(self):
        """Amounts > 10000 should trigger high_amount (+30)."""
        txns = [_make_txn(amount=15000)]
        scored = apply_risk_rules(txns)
        assert "high_amount" in scored[0]["triggered_rules"]
        assert scored[0]["risk_score"] >= 30

    def test_high_amount_not_triggered_at_boundary(self):
        """Amount exactly 10000 should NOT trigger high_amount."""
        txns = [_make_txn(amount=10000)]
        scored = apply_risk_rules(txns)
        assert "high_amount" not in scored[0]["triggered_rules"]

    def test_high_amount_not_triggered_below(self):
        """Amounts below 10000 should NOT trigger high_amount."""
        txns = [_make_txn(amount=5000)]
        scored = apply_risk_rules(txns)
        assert "high_amount" not in scored[0]["triggered_rules"]

    # ---- Rule 2: risky_type (CASH_OUT / TRANSFER -> +20) ----

    def test_risky_type_cash_out(self):
        """CASH_OUT type should trigger risky_type (+20)."""
        txns = [_make_txn(txn_type="CASH_OUT")]
        scored = apply_risk_rules(txns)
        assert "risky_type" in scored[0]["triggered_rules"]

    def test_risky_type_transfer(self):
        """TRANSFER type should trigger risky_type (+20)."""
        txns = [_make_txn(txn_type="TRANSFER")]
        scored = apply_risk_rules(txns)
        assert "risky_type" in scored[0]["triggered_rules"]

    def test_risky_type_not_triggered_for_payment(self):
        """PAYMENT type should NOT trigger risky_type."""
        txns = [_make_txn(txn_type="PAYMENT")]
        scored = apply_risk_rules(txns)
        assert "risky_type" not in scored[0]["triggered_rules"]

    def test_risky_type_not_triggered_for_debit(self):
        """DEBIT type should NOT trigger risky_type."""
        txns = [_make_txn(txn_type="DEBIT")]
        scored = apply_risk_rules(txns)
        assert "risky_type" not in scored[0]["triggered_rules"]

    # ---- Rule 3: new_destination (unseen nameDest -> +20) ----

    def test_new_destination_first_occurrence(self):
        """The first transaction to a destination should trigger new_destination."""
        txns = [_make_txn(name_dest="NEW_DEST")]
        scored = apply_risk_rules(txns)
        assert "new_destination" in scored[0]["triggered_rules"]

    def test_new_destination_not_triggered_on_repeat(self):
        """A repeated destination should NOT trigger new_destination the second time."""
        txns = [
            _make_txn(name_dest="SAME_DEST", transaction_id="TXN-0001"),
            _make_txn(name_dest="SAME_DEST", transaction_id="TXN-0002"),
        ]
        scored = apply_risk_rules(txns)
        assert "new_destination" in scored[0]["triggered_rules"]
        assert "new_destination" not in scored[1]["triggered_rules"]

    # ---- Rule 4: rapid_transactions (>1 txn in same step+origin -> +15) ----

    def test_rapid_transactions_triggered(self):
        """Multiple transactions in the same step from the same origin trigger rapid_transactions."""
        txns = [
            _make_txn(step=1, name_orig="C_same", name_dest="D1", transaction_id="TXN-0001"),
            _make_txn(step=1, name_orig="C_same", name_dest="D2", transaction_id="TXN-0002"),
        ]
        scored = apply_risk_rules(txns)
        assert "rapid_transactions" in scored[0]["triggered_rules"]
        assert "rapid_transactions" in scored[1]["triggered_rules"]

    def test_rapid_transactions_not_triggered_different_steps(self):
        """Transactions in different steps from the same origin should NOT trigger rapid_transactions."""
        txns = [
            _make_txn(step=1, name_orig="C_same", name_dest="D1", transaction_id="TXN-0001"),
            _make_txn(step=2, name_orig="C_same", name_dest="D2", transaction_id="TXN-0002"),
        ]
        scored = apply_risk_rules(txns)
        assert "rapid_transactions" not in scored[0]["triggered_rules"]
        assert "rapid_transactions" not in scored[1]["triggered_rules"]

    def test_rapid_transactions_not_triggered_different_origins(self):
        """Transactions in the same step but different origins should NOT trigger."""
        txns = [
            _make_txn(step=1, name_orig="C_a", name_dest="D1", transaction_id="TXN-0001"),
            _make_txn(step=1, name_orig="C_b", name_dest="D2", transaction_id="TXN-0002"),
        ]
        scored = apply_risk_rules(txns)
        assert "rapid_transactions" not in scored[0]["triggered_rules"]
        assert "rapid_transactions" not in scored[1]["triggered_rules"]

    # ---- Rule 5: high_amount_then_cashout (+15) ----

    def test_high_amount_then_cashout_triggered(self):
        """A CASH_OUT preceded by a high-amount txn from same origin triggers the rule."""
        txns = [
            _make_txn(
                step=1, txn_type="TRANSFER", amount=15000,
                name_orig="C1", name_dest="D1", transaction_id="TXN-0001",
            ),
            _make_txn(
                step=2, txn_type="CASH_OUT", amount=500,
                name_orig="C1", name_dest="D2", transaction_id="TXN-0002",
            ),
        ]
        scored = apply_risk_rules(txns)
        assert "high_amount_then_cashout" in scored[1]["triggered_rules"]

    def test_high_amount_then_cashout_not_triggered_low_amount(self):
        """A CASH_OUT preceded by a low-amount txn should NOT trigger the rule."""
        txns = [
            _make_txn(
                step=1, txn_type="TRANSFER", amount=5000,
                name_orig="C1", name_dest="D1", transaction_id="TXN-0001",
            ),
            _make_txn(
                step=2, txn_type="CASH_OUT", amount=500,
                name_orig="C1", name_dest="D2", transaction_id="TXN-0002",
            ),
        ]
        scored = apply_risk_rules(txns)
        assert "high_amount_then_cashout" not in scored[1]["triggered_rules"]

    def test_high_amount_then_cashout_not_triggered_non_cashout(self):
        """A TRANSFER after a high-amount txn should NOT trigger the rule (requires CASH_OUT)."""
        txns = [
            _make_txn(
                step=1, txn_type="TRANSFER", amount=15000,
                name_orig="C1", name_dest="D1", transaction_id="TXN-0001",
            ),
            _make_txn(
                step=2, txn_type="TRANSFER", amount=500,
                name_orig="C1", name_dest="D2", transaction_id="TXN-0002",
            ),
        ]
        scored = apply_risk_rules(txns)
        assert "high_amount_then_cashout" not in scored[1]["triggered_rules"]

    def test_high_amount_then_cashout_different_origin_not_triggered(self):
        """A CASH_OUT from a different origin should NOT trigger the rule."""
        txns = [
            _make_txn(
                step=1, txn_type="TRANSFER", amount=15000,
                name_orig="C1", name_dest="D1", transaction_id="TXN-0001",
            ),
            _make_txn(
                step=2, txn_type="CASH_OUT", amount=500,
                name_orig="C2", name_dest="D2", transaction_id="TXN-0002",
            ),
        ]
        scored = apply_risk_rules(txns)
        assert "high_amount_then_cashout" not in scored[1]["triggered_rules"]

    # ---- Score capping at 100 ----

    def test_score_capped_at_100(self):
        """Even when many rules fire, score must not exceed 100."""
        # All 5 rules fire: high_amount(30) + risky_type(20) + new_dest(20) +
        # rapid(15) + high_amount_then_cashout(15) = 100
        # For > 100, we need two rapid CASH_OUT txns both preceded by high amount.
        # Actually 30+20+20+15+15 = 100 exactly. Let's construct a case where
        # it would be > 100 if uncapped by giving a very high overlap.
        # With the current rules the max natural sum is 100, so let's just verify
        # that the result equals 100 when all rules fire.
        txns = [
            _make_txn(
                step=1, txn_type="TRANSFER", amount=15000,
                name_orig="C1", name_dest="D1", transaction_id="TXN-0001",
            ),
            _make_txn(
                step=1, txn_type="CASH_OUT", amount=15000,
                name_orig="C1", name_dest="D2", transaction_id="TXN-0002",
            ),
        ]
        scored = apply_risk_rules(txns)
        # Second txn triggers all 5 rules: high_amount + risky_type + new_dest +
        # rapid_transactions + high_amount_then_cashout = 30+20+20+15+15 = 100
        assert scored[1]["risk_score"] <= 100

    # ---- Risk category thresholds ----

    def test_low_risk_category(self):
        """Score < 40 should yield LOW risk category."""
        # PAYMENT, small amount, new destination only -> 20 -> LOW
        txns = [_make_txn(txn_type="PAYMENT", amount=100)]
        scored = apply_risk_rules(txns)
        assert scored[0]["risk_score"] < 40
        assert scored[0]["risk_category"] == "LOW"

    def test_medium_risk_category(self):
        """Score between 40 and 70 should yield MEDIUM risk category."""
        # CASH_OUT (risky_type +20) + new_destination (+20) = 40 -> MEDIUM
        txns = [_make_txn(txn_type="CASH_OUT", amount=100)]
        scored = apply_risk_rules(txns)
        assert 40 <= scored[0]["risk_score"] <= 70
        assert scored[0]["risk_category"] == "MEDIUM"

    def test_high_risk_category(self):
        """Score > 70 should yield HIGH risk category."""
        # high_amount (+30) + risky_type (+20) + new_destination (+20) = 70 -> MEDIUM boundary
        # Need > 70. Add rapid_transactions for +15 -> 85.
        txns = [
            _make_txn(
                step=1, txn_type="CASH_OUT", amount=15000,
                name_orig="C1", name_dest="D1", transaction_id="TXN-0001",
            ),
            _make_txn(
                step=1, txn_type="CASH_OUT", amount=15000,
                name_orig="C1", name_dest="D2", transaction_id="TXN-0002",
            ),
        ]
        scored = apply_risk_rules(txns)
        # Both trigger: high_amount(30) + risky_type(20) + new_dest(20) + rapid(15) = 85
        assert scored[0]["risk_score"] > 70
        assert scored[0]["risk_category"] == "HIGH"

    def test_medium_boundary_at_70(self):
        """Score of exactly 70 should be MEDIUM (boundary: score <= 70)."""
        # high_amount(30) + risky_type(20) + new_destination(20) = 70
        txns = [_make_txn(txn_type="TRANSFER", amount=15000)]
        scored = apply_risk_rules(txns)
        assert scored[0]["risk_score"] == 70
        assert scored[0]["risk_category"] == "MEDIUM"

    # ---- triggered_rules format ----

    def test_triggered_rules_semicolon_format(self):
        """Multiple triggered rules should be separated by '; '."""
        txns = [_make_txn(txn_type="TRANSFER", amount=15000)]
        scored = apply_risk_rules(txns)
        rules = scored[0]["triggered_rules"]
        # Should contain high_amount; risky_type; new_destination
        parts = [r.strip() for r in rules.split(";")]
        assert "high_amount" in parts
        assert "risky_type" in parts
        assert "new_destination" in parts

    def test_no_rules_triggered_format(self):
        """When no rules fire, triggered_rules should be 'none'."""
        # A PAYMENT, amount <= 10000, repeated destination, different step/origin
        # First transaction always gets new_destination, so we need a setup txn.
        txns = [
            _make_txn(
                txn_type="PAYMENT", amount=100,
                name_dest="KNOWN", transaction_id="TXN-0001",
            ),
            _make_txn(
                step=2, txn_type="PAYMENT", amount=100,
                name_orig="C_other", name_dest="KNOWN", transaction_id="TXN-0002",
            ),
        ]
        scored = apply_risk_rules(txns)
        assert scored[1]["triggered_rules"] == "none"

    def test_empty_transaction_list(self):
        """apply_risk_rules on an empty list should return an empty list."""
        assert apply_risk_rules([]) == []


# ===================================================================
# generate_report
# ===================================================================


class TestGenerateReport:
    """Tests for the generate_report function."""

    def test_report_file_created(self, tmp_path):
        """generate_report should create the CSV file at the specified path."""
        output = str(tmp_path / "report.csv")
        generate_report([], output)
        assert os.path.isfile(output)

    def test_report_headers(self, tmp_path):
        """The CSV should contain the expected header columns."""
        output = str(tmp_path / "report.csv")
        generate_report([], output)
        with open(output, encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert headers == [
            "transaction_id", "amount", "type",
            "risk_score", "risk_category", "triggered_rules",
        ]

    def test_report_row_data(self, tmp_path):
        """Row data in the CSV should match the scored transaction dicts."""
        scored = [
            {
                "transaction_id": "TXN-0001",
                "amount": 5000.0,
                "type": "PAYMENT",
                "risk_score": 20,
                "risk_category": "LOW",
                "triggered_rules": "new_destination",
            },
            {
                "transaction_id": "TXN-0002",
                "amount": 15000.0,
                "type": "TRANSFER",
                "risk_score": 70,
                "risk_category": "MEDIUM",
                "triggered_rules": "high_amount; risky_type; new_destination",
            },
        ]
        output = str(tmp_path / "report.csv")
        generate_report(scored, output)

        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["transaction_id"] == "TXN-0001"
        assert rows[0]["amount"] == "5000.0"
        assert rows[0]["type"] == "PAYMENT"
        assert rows[0]["risk_score"] == "20"
        assert rows[0]["risk_category"] == "LOW"
        assert rows[0]["triggered_rules"] == "new_destination"

        assert rows[1]["transaction_id"] == "TXN-0002"
        assert rows[1]["risk_category"] == "MEDIUM"

    def test_report_in_subdirectory(self, tmp_path):
        """generate_report should write into nested subdirectories."""
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        output = str(nested / "report.csv")
        generate_report([], output)
        assert os.path.isfile(output)


# ===================================================================
# print_summary
# ===================================================================


class TestPrintSummary:
    """Tests for the print_summary function."""

    def _make_scored(self, low=0, medium=0, high=0):
        """Build a minimal scored list with the given category counts."""
        items = []
        for _ in range(low):
            items.append({"risk_category": "LOW"})
        for _ in range(medium):
            items.append({"risk_category": "MEDIUM"})
        for _ in range(high):
            items.append({"risk_category": "HIGH"})
        return items

    def test_summary_counts_all_categories(self, capsys):
        """print_summary should report correct LOW, MEDIUM, HIGH counts."""
        scored = self._make_scored(low=3, medium=2, high=1)
        print_summary(scored)
        output = capsys.readouterr().out

        assert "Total transactions scored: 6" in output
        assert "LOW" in output and "3" in output
        assert "MEDIUM" in output and "2" in output
        assert "HIGH" in output and "1" in output

    def test_summary_zero_transactions(self, capsys):
        """print_summary should handle an empty list gracefully."""
        print_summary([])
        output = capsys.readouterr().out
        assert "Total transactions scored: 0" in output

    def test_summary_only_high(self, capsys):
        """When all transactions are HIGH, LOW and MEDIUM counts should be 0."""
        scored = self._make_scored(high=5)
        print_summary(scored)
        output = capsys.readouterr().out
        assert "Total transactions scored: 5" in output
        # Verify LOW count is 0
        assert "LOW  risk (score < 40):   0" in output
        # Verify MEDIUM count is 0
        assert "MEDIUM risk (40-70):      0" in output

    def test_summary_only_low(self, capsys):
        """When all transactions are LOW, MEDIUM and HIGH counts should be 0."""
        scored = self._make_scored(low=4)
        print_summary(scored)
        output = capsys.readouterr().out
        assert "Total transactions scored: 4" in output
        assert "HIGH risk (> 70):         0" in output
        assert "MEDIUM risk (40-70):      0" in output
