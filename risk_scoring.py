"""
Transaction Risk Scoring Engine

Assigns a risk score (0-100) to each transaction based on configurable rules,
classifies transactions as LOW, MEDIUM, or HIGH risk, and generates a report.
"""

import csv
import os
from collections import defaultdict


def load_transactions(filepath):
    """Load transactions from CSV file and validate required fields."""
    required_fields = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
        "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    ]
    transactions = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header_fields = reader.fieldnames or []
        for field in required_fields:
            if field not in header_fields:
                raise ValueError(f"Missing required field: {field}")
        for idx, row in enumerate(reader, start=1):
            row["transaction_id"] = f"TXN-{idx:04d}"
            row["amount"] = float(row["amount"])
            row["step"] = int(row["step"])
            row["oldbalanceOrg"] = float(row["oldbalanceOrg"])
            row["newbalanceOrig"] = float(row["newbalanceOrig"])
            row["oldbalanceDest"] = float(row["oldbalanceDest"])
            row["newbalanceDest"] = float(row["newbalanceDest"])
            transactions.append(row)
    return transactions


def apply_risk_rules(transactions):
    """Apply risk rules to each transaction and return scored transactions."""
    # Track seen destination accounts for "new destination" rule
    seen_destinations = set()

    # Build lookup structures for rapid-transaction and high-amount-then-cashout rules
    # Group transactions by (step, nameOrig) for rapid transaction detection
    step_orig_groups = defaultdict(list)
    for txn in transactions:
        step_orig_groups[(txn["step"], txn["nameOrig"])].append(txn)

    # Build per-origin ordered list for high-amount-then-cashout detection
    orig_ordered = defaultdict(list)
    for txn in transactions:
        orig_ordered[txn["nameOrig"]].append(txn)

    # Pre-compute which transactions are preceded by a high-amount transaction
    # from the same origin account (high amount followed by cash-out)
    high_amount_then_cashout = set()
    for orig, txns in orig_ordered.items():
        for i, txn in enumerate(txns):
            if txn["type"] == "CASH_OUT" and i > 0:
                prev = txns[i - 1]
                if prev["amount"] > 10000:
                    high_amount_then_cashout.add(txn["transaction_id"])

    scored = []
    for txn in transactions:
        rules_triggered = []
        score = 0

        # Rule 1: amount > 10000 -> +30
        if txn["amount"] > 10000:
            score += 30
            rules_triggered.append("high_amount")

        # Rule 2: CASH_OUT or TRANSFER -> +20
        if txn["type"] in ("CASH_OUT", "TRANSFER"):
            score += 20
            rules_triggered.append("risky_type")

        # Rule 3: new (previously unseen) destination account -> +20
        dest = txn["nameDest"]
        if dest not in seen_destinations:
            score += 20
            rules_triggered.append("new_destination")
        seen_destinations.add(dest)

        # Rule 4: rapid transactions (multiple txns in same step from same account) -> +15
        group = step_orig_groups[(txn["step"], txn["nameOrig"])]
        if len(group) > 1:
            score += 15
            rules_triggered.append("rapid_transactions")

        # Rule 5: high amount followed by cash-out -> +15
        if txn["transaction_id"] in high_amount_then_cashout:
            score += 15
            rules_triggered.append("high_amount_then_cashout")

        # Cap score at 100
        score = min(score, 100)

        # Assign risk category
        if score < 40:
            category = "LOW"
        elif score <= 70:
            category = "MEDIUM"
        else:
            category = "HIGH"

        scored.append({
            "transaction_id": txn["transaction_id"],
            "amount": txn["amount"],
            "type": txn["type"],
            "risk_score": score,
            "risk_category": category,
            "triggered_rules": "; ".join(rules_triggered) if rules_triggered else "none",
        })

    return scored


def generate_report(scored_transactions, output_path):
    """Save scored transactions as a CSV report."""
    fieldnames = [
        "transaction_id", "amount", "type",
        "risk_score", "risk_category", "triggered_rules",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_transactions)


def print_summary(scored_transactions):
    """Print a summary of the risk scoring results."""
    total = len(scored_transactions)
    low = sum(1 for t in scored_transactions if t["risk_category"] == "LOW")
    medium = sum(1 for t in scored_transactions if t["risk_category"] == "MEDIUM")
    high = sum(1 for t in scored_transactions if t["risk_category"] == "HIGH")
    print(f"\n{'='*60}")
    print("TRANSACTION RISK SCORING SUMMARY")
    print(f"{'='*60}")
    print(f"Total transactions scored: {total}")
    print(f"  LOW  risk (score < 40):   {low}")
    print(f"  MEDIUM risk (40-70):      {medium}")
    print(f"  HIGH risk (> 70):         {high}")
    print(f"{'='*60}\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "Data", "Example1.csv")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "risk_report.csv")

    print("Loading transactions...")
    transactions = load_transactions(input_path)
    print(f"Loaded {len(transactions)} transactions.")

    print("Applying risk rules...")
    scored = apply_risk_rules(transactions)

    print(f"Saving report to {output_path}...")
    generate_report(scored, output_path)

    print_summary(scored)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
