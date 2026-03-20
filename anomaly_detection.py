"""
Transaction Sequence Anomaly Detection Engine

Detects anomalous transaction sequences for each customer by evaluating:
- Transaction order and amount behavior
- Transaction type transitions
- Short time-window activity
- Repeated high-value transactions
- Sudden spikes in amount relative to customer baseline

All thresholds are configurable via the CONFIG dictionary.
"""

import csv
import json
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------
CONFIG = {
    # A transaction amount above this value is considered "high value"
    "high_value_threshold": 10000,
    # Rolling window size (in step units) for grouping close transactions
    "short_window_duration": 1,
    # A transaction amount exceeding the customer's recent average by this
    # multiplier is flagged as a sudden spike
    "spike_multiplier": 3.0,
    # Minimum number of transactions in a sequence to consider for anomaly
    "min_sequence_length": 2,
    # Number of recent transactions used to compute the customer baseline
    "baseline_window_size": 5,
    # Minimum number of repeated high-value transactions to flag
    "repeated_high_value_min_count": 2,
}

# Risk transaction types (from knowledge notes)
RISKY_TYPES = {"CASH_OUT", "TRANSFER"}

# Severity levels
SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW = "LOW"


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------
def load_transactions(filepath):
    """Load transactions from CSV and validate required fields.

    Maps the dataset columns to canonical names:
      - customer_id   <- nameOrig
      - transaction_id <- generated TXN-XXXX
      - transaction_type <- type
      - amount         <- amount
      - timestamp      <- step
    """
    raw_required_fields = ["step", "type", "amount", "nameOrig"]
    transactions = []

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header_fields = reader.fieldnames or []
        for field in raw_required_fields:
            if field not in header_fields:
                raise ValueError(f"Missing required field: {field}")

        for idx, row in enumerate(reader, start=1):
            transactions.append({
                "customer_id": row["nameOrig"],
                "transaction_id": f"TXN-{idx:04d}",
                "transaction_type": row["type"],
                "amount": float(row["amount"]),
                "timestamp": int(row["step"]),
                "nameDest": row.get("nameDest", ""),
            })

    return transactions


def validate_transactions(transactions):
    """Validate that every transaction has the required canonical fields."""
    required = ["customer_id", "transaction_id", "transaction_type", "amount", "timestamp"]
    for txn in transactions:
        for field in required:
            if field not in txn or txn[field] is None:
                raise ValueError(
                    f"Transaction {txn.get('transaction_id', '?')} missing field: {field}"
                )


# ---------------------------------------------------------------------------
# Grouping and sorting
# ---------------------------------------------------------------------------
def group_by_customer(transactions):
    """Group transactions by customer_id and sort each group by timestamp."""
    groups = defaultdict(list)
    for txn in transactions:
        groups[txn["customer_id"]].append(txn)

    # Sort each customer's transactions by timestamp ascending
    for customer_id in groups:
        groups[customer_id].sort(key=lambda t: (t["timestamp"], t["transaction_id"]))

    return dict(groups)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def build_windows(customer_txns, window_duration):
    """Build rolling windows of transactions within *window_duration* steps.

    Returns a list of windows where each window is a list of transactions
    that fall within the same time window.
    """
    if not customer_txns:
        return []

    windows = []
    current_window = [customer_txns[0]]

    for txn in customer_txns[1:]:
        if txn["timestamp"] - current_window[0]["timestamp"] <= window_duration:
            current_window.append(txn)
        else:
            windows.append(current_window)
            current_window = [txn]

    if current_window:
        windows.append(current_window)

    return windows


# ---------------------------------------------------------------------------
# Pattern detection helpers
# ---------------------------------------------------------------------------
def _compute_baseline(customer_txns, up_to_index, baseline_window_size):
    """Compute the average amount over the recent baseline window."""
    start = max(0, up_to_index - baseline_window_size)
    window = customer_txns[start:up_to_index]
    if not window:
        return 0.0
    return sum(t["amount"] for t in window) / len(window)


def _normal_types_for_customer(customer_txns, up_to_index):
    """Return the set of transaction types seen so far for the customer."""
    return {t["transaction_type"] for t in customer_txns[:up_to_index]}


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------
def detect_repeated_high_value(window, config):
    """Detect repeated high-value transactions in a window."""
    high_value_txns = [
        t for t in window if t["amount"] > config["high_value_threshold"]
    ]
    if len(high_value_txns) >= config["repeated_high_value_min_count"]:
        return {
            "pattern": "REPEATED_HIGH_VALUE",
            "reason": (
                f"{len(high_value_txns)} transactions above "
                f"{config['high_value_threshold']} in short window"
            ),
            "severity": SEVERITY_HIGH,
            "involved_txns": high_value_txns,
        }
    return None


def detect_transfer_then_cashout(window, config):
    """Detect TRANSFER followed immediately by CASH_OUT."""
    results = []
    for i in range(len(window) - 1):
        if (
            window[i]["transaction_type"] == "TRANSFER"
            and window[i + 1]["transaction_type"] == "CASH_OUT"
        ):
            results.append({
                "pattern": "TRANSFER_THEN_CASHOUT",
                "reason": "TRANSFER followed immediately by CASH_OUT — high risk layering pattern",
                "severity": SEVERITY_HIGH,
                "involved_txns": [window[i], window[i + 1]],
            })
    return results


def detect_transfer_chain(window, config):
    """Detect TRANSFER -> TRANSFER -> CASH_OUT chain."""
    results = []
    for i in range(len(window) - 2):
        if (
            window[i]["transaction_type"] == "TRANSFER"
            and window[i + 1]["transaction_type"] == "TRANSFER"
            and window[i + 2]["transaction_type"] == "CASH_OUT"
        ):
            results.append({
                "pattern": "TRANSFER_CHAIN_CASHOUT",
                "reason": (
                    "TRANSFER -> TRANSFER -> CASH_OUT chain detected"
                    " -- strong suspicious pattern"
                ),
                "severity": SEVERITY_HIGH,
                "involved_txns": [window[i], window[i + 1], window[i + 2]],
            })
    return results


def detect_sudden_spike(customer_txns, config):
    """Detect sudden spike in amount relative to customer baseline."""
    results = []
    baseline_window_size = config["baseline_window_size"]
    spike_multiplier = config["spike_multiplier"]

    for i in range(1, len(customer_txns)):
        baseline = _compute_baseline(customer_txns, i, baseline_window_size)
        if baseline > 0 and customer_txns[i]["amount"] > baseline * spike_multiplier:
            results.append({
                "pattern": "SUDDEN_AMOUNT_SPIKE",
                "reason": (
                    f"Amount {customer_txns[i]['amount']:.2f} is "
                    f"{customer_txns[i]['amount'] / baseline:.1f}x the recent "
                    f"baseline ({baseline:.2f})"
                ),
                "severity": SEVERITY_MEDIUM,
                "involved_txns": [customer_txns[i]],
            })
    return results


def detect_sudden_type_change(customer_txns, config):
    """Detect sudden change to risky transaction types."""
    results = []
    for i in range(1, len(customer_txns)):
        seen = _normal_types_for_customer(customer_txns, i)
        current_type = customer_txns[i]["transaction_type"]
        # If current type is risky and was never seen before for this customer
        if current_type in RISKY_TYPES and current_type not in seen:
            results.append({
                "pattern": "SUDDEN_TYPE_CHANGE",
                "reason": (
                    f"Transaction type changed to {current_type} "
                    f"(previously only: {', '.join(sorted(seen))})"
                ),
                "severity": SEVERITY_MEDIUM,
                "involved_txns": [customer_txns[i]],
            })
    return results


# ---------------------------------------------------------------------------
# Main anomaly detection pipeline
# ---------------------------------------------------------------------------
def detect_anomalies(transactions, config=None):
    """Run the full anomaly detection pipeline.

    Returns a list of anomaly records, each containing:
      - customer_id
      - involved_transaction_ids
      - sequence_start_time
      - sequence_end_time
      - pattern
      - reason
      - severity
    """
    if config is None:
        config = CONFIG

    validate_transactions(transactions)
    customer_groups = group_by_customer(transactions)
    anomalies = []

    for customer_id, txns in customer_groups.items():
        # --- Window-based detections ---
        windows = build_windows(txns, config["short_window_duration"])

        for window in windows:
            if len(window) < config["min_sequence_length"]:
                continue

            # 1. Repeated high-value transactions
            result = detect_repeated_high_value(window, config)
            if result:
                anomalies.append(_build_anomaly_record(customer_id, result))

            # 2. TRANSFER → CASH_OUT
            for result in detect_transfer_then_cashout(window, config):
                anomalies.append(_build_anomaly_record(customer_id, result))

            # 3. TRANSFER → TRANSFER → CASH_OUT chain
            for result in detect_transfer_chain(window, config):
                anomalies.append(_build_anomaly_record(customer_id, result))

        # --- Full-history detections (need customer baseline) ---
        # 4. Sudden spike in amount
        for result in detect_sudden_spike(txns, config):
            anomalies.append(_build_anomaly_record(customer_id, result))

        # 5. Sudden change in transaction type
        for result in detect_sudden_type_change(txns, config):
            anomalies.append(_build_anomaly_record(customer_id, result))

    return anomalies


def _build_anomaly_record(customer_id, detection_result):
    """Convert a raw detection result into a standardised anomaly record."""
    involved = detection_result["involved_txns"]
    return {
        "customer_id": customer_id,
        "involved_transaction_ids": ", ".join(t["transaction_id"] for t in involved),
        "sequence_start_time": min(t["timestamp"] for t in involved),
        "sequence_end_time": max(t["timestamp"] for t in involved),
        "pattern": detection_result["pattern"],
        "reason": detection_result["reason"],
        "severity": detection_result["severity"],
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
REPORT_FIELDS = [
    "customer_id",
    "involved_transaction_ids",
    "sequence_start_time",
    "sequence_end_time",
    "pattern",
    "reason",
    "severity",
]


def generate_csv_report(anomalies, output_path):
    """Write anomalies to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(anomalies)


def generate_json_report(anomalies, output_path):
    """Write anomalies to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(anomalies, f, indent=2)


def print_summary(anomalies):
    """Print a human-readable summary of detected anomalies."""
    total = len(anomalies)
    by_pattern = defaultdict(int)
    by_severity = defaultdict(int)
    for a in anomalies:
        by_pattern[a["pattern"]] += 1
        by_severity[a["severity"]] += 1

    print()
    print("=" * 70)
    print("TRANSACTION SEQUENCE ANOMALY DETECTION SUMMARY")
    print("=" * 70)
    print(f"Total anomalous sequences detected: {total}")
    print("\nBy severity:")
    for sev in [SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW]:
        print(f"  {sev:8s}: {by_severity.get(sev, 0)}")
    print("\nBy pattern:")
    for pattern, count in sorted(by_pattern.items()):
        print(f"  {pattern:30s}: {count}")
    print("=" * 70)

    unique_customers = len({a["customer_id"] for a in anomalies})
    print(f"Unique customers with anomalies: {unique_customers}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Configuration summary (for PR / report documentation)
# ---------------------------------------------------------------------------
def config_summary(config=None):
    """Return a human-readable summary of the active configuration."""
    if config is None:
        config = CONFIG
    lines = ["Anomaly Detection Configuration:", "-" * 40]
    for key, value in config.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "Data", "sample_transactions.csv")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    csv_output = os.path.join(output_dir, "anomaly_report.csv")
    json_output = os.path.join(output_dir, "anomaly_report.json")

    print(config_summary())
    print()

    print("Loading transactions...")
    transactions = load_transactions(input_path)
    print(f"Loaded {len(transactions)} transactions.")

    print("\nRunning anomaly detection...")
    anomalies = detect_anomalies(transactions)

    print(f"\nSaving CSV report to {csv_output}...")
    generate_csv_report(anomalies, csv_output)

    print(f"Saving JSON report to {json_output}...")
    generate_json_report(anomalies, json_output)

    print_summary(anomalies)

    # Print first few sample results
    print("Sample anomaly records (first 10):")
    print("-" * 70)
    for a in anomalies[:10]:
        print(
            f"  Customer: {a['customer_id']} | Pattern: {a['pattern']} | "
            f"Severity: {a['severity']}"
        )
        print(f"    TXNs: {a['involved_transaction_ids']}")
        print(f"    Time: {a['sequence_start_time']} - {a['sequence_end_time']}")
        print(f"    Reason: {a['reason']}")
        print()

    print(f"Reports saved to:\n  CSV:  {csv_output}\n  JSON: {json_output}")


if __name__ == "__main__":
    main()
