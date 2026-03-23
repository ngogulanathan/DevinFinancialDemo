"""Comprehensive unit tests for the apply_risk_rules function."""

import pytest
from risk_scoring import apply_risk_rules


def _make_txn(
    transaction_id="TXN-0001",
    step=1,
    txn_type="PAYMENT",
    amount=100.0,
    name_orig="C_ORIG_1",
    name_dest="C_DEST_1",
):
    """Helper to build a minimal transaction dict with sensible defaults."""
    return {
        "transaction_id": transaction_id,
        "step": step,
        "type": txn_type,
        "amount": amount,
        "nameOrig": name_orig,
        "nameDest": name_dest,
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 50000.0 - amount,
        "oldbalanceDest": 10000.0,
        "newbalanceDest": 10000.0 + amount,
    }


# ---------- Test 1: Rule 1 — High amount (+30) ----------

class TestRule1HighAmount:
    def test_high_amount_triggers(self):
        txn = _make_txn(amount=15000.0)
        result = apply_risk_rules([txn])
        assert len(result) == 1
        assert "high_amount" in result[0]["triggered_rules"]
        assert result[0]["risk_score"] >= 30

    def test_amount_at_boundary_does_not_trigger(self):
        txn = _make_txn(amount=10000.0)
        result = apply_risk_rules([txn])
        assert "high_amount" not in result[0]["triggered_rules"]

    def test_amount_just_above_boundary_triggers(self):
        txn = _make_txn(amount=10000.01)
        result = apply_risk_rules([txn])
        assert "high_amount" in result[0]["triggered_rules"]


# ---------- Test 2: Rule 2 — Risky type (+20) ----------

class TestRule2RiskyType:
    def test_cash_out_triggers(self):
        txn = _make_txn(txn_type="CASH_OUT", amount=500.0)
        result = apply_risk_rules([txn])
        assert "risky_type" in result[0]["triggered_rules"]

    def test_transfer_triggers(self):
        txn = _make_txn(txn_type="TRANSFER", amount=500.0)
        result = apply_risk_rules([txn])
        assert "risky_type" in result[0]["triggered_rules"]

    def test_payment_does_not_trigger(self):
        txn = _make_txn(txn_type="PAYMENT", amount=500.0)
        result = apply_risk_rules([txn])
        assert "risky_type" not in result[0]["triggered_rules"]

    def test_debit_does_not_trigger(self):
        txn = _make_txn(txn_type="DEBIT", amount=500.0)
        result = apply_risk_rules([txn])
        assert "risky_type" not in result[0]["triggered_rules"]


# ---------- Test 3: Rule 3 — New destination (+20) ----------

class TestRule3NewDestination:
    def test_first_destination_triggers(self):
        txn = _make_txn(name_dest="NEW_DEST_1")
        result = apply_risk_rules([txn])
        assert "new_destination" in result[0]["triggered_rules"]

    def test_second_same_destination_does_not_trigger(self):
        txn1 = _make_txn(transaction_id="TXN-0001", name_dest="SHARED_DEST", amount=500.0)
        txn2 = _make_txn(transaction_id="TXN-0002", name_dest="SHARED_DEST", amount=500.0)
        result = apply_risk_rules([txn1, txn2])
        assert "new_destination" in result[0]["triggered_rules"]
        assert "new_destination" not in result[1]["triggered_rules"]

    def test_different_destinations_both_trigger(self):
        txn1 = _make_txn(transaction_id="TXN-0001", name_dest="DEST_A", amount=500.0)
        txn2 = _make_txn(transaction_id="TXN-0002", name_dest="DEST_B", amount=500.0)
        result = apply_risk_rules([txn1, txn2])
        assert "new_destination" in result[0]["triggered_rules"]
        assert "new_destination" in result[1]["triggered_rules"]


# ---------- Test 4: Rule 4 — Rapid transactions (+15) ----------

class TestRule4RapidTransactions:
    def test_rapid_transactions_same_step_same_origin(self):
        txn1 = _make_txn(
            transaction_id="TXN-0001", step=1, name_orig="C_ORIG_1",
            name_dest="DEST_A", amount=500.0,
        )
        txn2 = _make_txn(
            transaction_id="TXN-0002", step=1, name_orig="C_ORIG_1",
            name_dest="DEST_B", amount=600.0,
        )
        result = apply_risk_rules([txn1, txn2])
        assert "rapid_transactions" in result[0]["triggered_rules"]
        assert "rapid_transactions" in result[1]["triggered_rules"]

    def test_different_steps_no_rapid(self):
        txn1 = _make_txn(
            transaction_id="TXN-0001", step=1, name_orig="C_ORIG_1",
            name_dest="DEST_A", amount=500.0,
        )
        txn2 = _make_txn(
            transaction_id="TXN-0002", step=2, name_orig="C_ORIG_1",
            name_dest="DEST_B", amount=600.0,
        )
        result = apply_risk_rules([txn1, txn2])
        assert "rapid_transactions" not in result[0]["triggered_rules"]
        assert "rapid_transactions" not in result[1]["triggered_rules"]

    def test_same_step_different_origins_no_rapid(self):
        txn1 = _make_txn(
            transaction_id="TXN-0001", step=1, name_orig="C_ORIG_1",
            name_dest="DEST_A", amount=500.0,
        )
        txn2 = _make_txn(
            transaction_id="TXN-0002", step=1, name_orig="C_ORIG_2",
            name_dest="DEST_B", amount=600.0,
        )
        result = apply_risk_rules([txn1, txn2])
        assert "rapid_transactions" not in result[0]["triggered_rules"]
        assert "rapid_transactions" not in result[1]["triggered_rules"]


# ---------- Test 5: Rule 5 — High amount then cashout (+15) ----------

class TestRule5HighAmountThenCashout:
    def test_cashout_preceded_by_high_amount_triggers(self):
        txn1 = _make_txn(
            transaction_id="TXN-0001", name_orig="C_ORIG_1",
            txn_type="TRANSFER", amount=20000.0, name_dest="DEST_A",
        )
        txn2 = _make_txn(
            transaction_id="TXN-0002", name_orig="C_ORIG_1",
            txn_type="CASH_OUT", amount=5000.0, name_dest="DEST_B",
        )
        result = apply_risk_rules([txn1, txn2])
        assert "high_amount_then_cashout" in result[1]["triggered_rules"]

    def test_cashout_preceded_by_low_amount_does_not_trigger(self):
        txn1 = _make_txn(
            transaction_id="TXN-0001", name_orig="C_ORIG_1",
            txn_type="TRANSFER", amount=5000.0, name_dest="DEST_A",
        )
        txn2 = _make_txn(
            transaction_id="TXN-0002", name_orig="C_ORIG_1",
            txn_type="CASH_OUT", amount=3000.0, name_dest="DEST_B",
        )
        result = apply_risk_rules([txn1, txn2])
        assert "high_amount_then_cashout" not in result[1]["triggered_rules"]

    def test_non_cashout_after_high_amount_does_not_trigger(self):
        txn1 = _make_txn(
            transaction_id="TXN-0001", name_orig="C_ORIG_1",
            txn_type="TRANSFER", amount=20000.0, name_dest="DEST_A",
        )
        txn2 = _make_txn(
            transaction_id="TXN-0002", name_orig="C_ORIG_1",
            txn_type="PAYMENT", amount=5000.0, name_dest="DEST_B",
        )
        result = apply_risk_rules([txn1, txn2])
        assert "high_amount_then_cashout" not in result[1]["triggered_rules"]

    def test_cashout_from_different_origin_does_not_trigger(self):
        txn1 = _make_txn(
            transaction_id="TXN-0001", name_orig="C_ORIG_1",
            txn_type="TRANSFER", amount=20000.0, name_dest="DEST_A",
        )
        txn2 = _make_txn(
            transaction_id="TXN-0002", name_orig="C_ORIG_2",
            txn_type="CASH_OUT", amount=5000.0, name_dest="DEST_B",
        )
        result = apply_risk_rules([txn1, txn2])
        assert "high_amount_then_cashout" not in result[1]["triggered_rules"]

    def test_first_txn_cashout_does_not_trigger(self):
        """The very first transaction from an origin can't have a predecessor."""
        txn = _make_txn(
            transaction_id="TXN-0001", name_orig="C_ORIG_1",
            txn_type="CASH_OUT", amount=5000.0,
        )
        result = apply_risk_rules([txn])
        assert "high_amount_then_cashout" not in result[0]["triggered_rules"]


# ---------- Test 6: Score capping at 100 ----------

class TestScoreCapping:
    def test_score_capped_at_100(self):
        """Trigger all 5 rules: 30+20+20+15+15 = 100, already at cap."""
        # First a high-amount transfer so the cashout can trigger rule 5
        txn1 = _make_txn(
            transaction_id="TXN-0001", step=1, name_orig="C_ORIG_1",
            txn_type="TRANSFER", amount=20000.0, name_dest="DEST_A",
        )
        # Second: CASH_OUT, same step, same origin, high amount, new dest
        # Rules: high_amount(+30), risky_type(+20), new_destination(+20),
        #        rapid_transactions(+15), high_amount_then_cashout(+15) = 100
        txn2 = _make_txn(
            transaction_id="TXN-0002", step=1, name_orig="C_ORIG_1",
            txn_type="CASH_OUT", amount=20000.0, name_dest="DEST_B",
        )
        result = apply_risk_rules([txn1, txn2])
        assert result[1]["risk_score"] <= 100
        assert result[1]["risk_score"] == 100


# ---------- Test 7: Risk categories ----------

class TestRiskCategories:
    def test_low_category_score_0(self):
        # Pre-seed destination so new_destination doesn't fire on the real txn
        seed = _make_txn(
            transaction_id="TXN-0000", step=99, amount=100.0,
            name_dest="SEEN_DEST", name_orig="C_SEED",
        )
        txn = _make_txn(
            transaction_id="TXN-0001", step=2, amount=500.0,
            txn_type="PAYMENT", name_dest="SEEN_DEST", name_orig="C_ORIG_OTHER",
        )
        result = apply_risk_rules([seed, txn])
        assert result[1]["risk_score"] == 0
        assert result[1]["risk_category"] == "LOW"

    def test_low_category_score_below_40(self):
        # Only new_destination fires → score 20
        txn = _make_txn(amount=500.0, txn_type="PAYMENT")
        result = apply_risk_rules([txn])
        assert result[0]["risk_score"] == 20
        assert result[0]["risk_category"] == "LOW"

    def test_medium_category_at_40(self):
        # new_destination(20) + risky_type(20) = 40
        txn = _make_txn(amount=500.0, txn_type="CASH_OUT")
        result = apply_risk_rules([txn])
        assert result[0]["risk_score"] == 40
        assert result[0]["risk_category"] == "MEDIUM"

    def test_medium_category_at_70(self):
        # high_amount(30) + risky_type(20) + new_destination(20) = 70
        txn = _make_txn(amount=15000.0, txn_type="TRANSFER")
        result = apply_risk_rules([txn])
        assert result[0]["risk_score"] == 70
        assert result[0]["risk_category"] == "MEDIUM"

    def test_high_category_above_70(self):
        # high_amount(30) + risky_type(20) + new_destination(20) + rapid(15) = 85
        txn1 = _make_txn(
            transaction_id="TXN-0001", step=1, name_orig="C_ORIG_1",
            txn_type="TRANSFER", amount=15000.0, name_dest="DEST_A",
        )
        txn2 = _make_txn(
            transaction_id="TXN-0002", step=1, name_orig="C_ORIG_1",
            txn_type="TRANSFER", amount=15000.0, name_dest="DEST_B",
        )
        result = apply_risk_rules([txn1, txn2])
        assert result[0]["risk_score"] == 85
        assert result[0]["risk_category"] == "HIGH"


# ---------- Test 8: No rules triggered ----------

class TestNoRulesTriggered:
    def test_no_rules_triggered(self):
        """Small PAYMENT to a previously seen destination, no rapid/cashout pattern → score 0."""
        seed = _make_txn(transaction_id="TXN-0000", amount=100.0, name_dest="SEEN_DEST")
        txn = _make_txn(
            transaction_id="TXN-0001", step=2, amount=500.0,
            txn_type="PAYMENT", name_dest="SEEN_DEST", name_orig="C_ORIG_OTHER",
        )
        result = apply_risk_rules([seed, txn])
        assert result[1]["risk_score"] == 0
        assert result[1]["triggered_rules"] == "none"
        assert result[1]["risk_category"] == "LOW"


# ---------- Test 9: Multiple rules combining ----------

class TestMultipleRulesCombining:
    def test_three_rules_combine(self):
        # high_amount(30) + risky_type(20) + new_destination(20) = 70
        txn = _make_txn(amount=15000.0, txn_type="CASH_OUT")
        result = apply_risk_rules([txn])
        assert result[0]["risk_score"] == 70
        rules = result[0]["triggered_rules"]
        assert "high_amount" in rules
        assert "risky_type" in rules
        assert "new_destination" in rules

    def test_all_five_rules_combine(self):
        # Setup: two txns same step/origin so rapid fires; first is high-amount
        # so rule 5 fires on the CASH_OUT
        txn1 = _make_txn(
            transaction_id="TXN-0001", step=1, name_orig="C_ORIG_1",
            txn_type="TRANSFER", amount=20000.0, name_dest="DEST_A",
        )
        txn2 = _make_txn(
            transaction_id="TXN-0002", step=1, name_orig="C_ORIG_1",
            txn_type="CASH_OUT", amount=20000.0, name_dest="DEST_B",
        )
        result = apply_risk_rules([txn1, txn2])
        rules = result[1]["triggered_rules"]
        for rule in ["high_amount", "risky_type", "new_destination",
                      "rapid_transactions", "high_amount_then_cashout"]:
            assert rule in rules, f"Expected rule '{rule}' to be triggered"
        assert result[1]["risk_score"] == 100

    def test_two_rules_combine_risky_type_and_new_dest(self):
        # risky_type(20) + new_destination(20) = 40
        txn = _make_txn(amount=500.0, txn_type="TRANSFER")
        result = apply_risk_rules([txn])
        assert result[0]["risk_score"] == 40
        rules = result[0]["triggered_rules"]
        assert "risky_type" in rules
        assert "new_destination" in rules
        assert "high_amount" not in rules


# ---------- Test 10: Empty transaction list ----------

class TestEmptyTransactionList:
    def test_empty_list_returns_empty(self):
        result = apply_risk_rules([])
        assert result == []


# ---------- Additional edge-case tests ----------

class TestOutputStructure:
    def test_output_keys(self):
        txn = _make_txn()
        result = apply_risk_rules([txn])
        expected_keys = {
            "transaction_id", "amount", "type",
            "risk_score", "risk_category", "triggered_rules",
        }
        assert set(result[0].keys()) == expected_keys

    def test_output_preserves_transaction_id_and_amount(self):
        txn = _make_txn(transaction_id="TXN-9999", amount=42.0)
        result = apply_risk_rules([txn])
        assert result[0]["transaction_id"] == "TXN-9999"
        assert result[0]["amount"] == 42.0
        assert result[0]["type"] == "PAYMENT"
