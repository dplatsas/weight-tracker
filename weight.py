"""
Daily Weight Tracker with Trend Analysis
Run once per day to log your weight and see progress with trend lines.
"""

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────────────────
CSV_FILENAME = "weight_log.csv"
GOAL_WEIGHT_KG = 72.0
MOVING_AVG_WINDOW = 7
MIN_ENTRIES_FOR_TREND = 5
# ──────────────────────────────────────────────────────────────────────────────

CSV_PATH = Path(__file__).parent / CSV_FILENAME


def initialize_csv() -> None:
    """Create CSV file with headers if it doesn't exist."""
    if not CSV_PATH.exists():
        CSV_PATH.write_text("date,weight_kg\n")


def read_entries() -> list[dict]:
    """Read and return all entries sorted by date."""
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open(newline="") as f:
        entries = list(csv.DictReader(f))
    return sorted(entries, key=lambda e: e["date"])


def write_entries(entries: list[dict]) -> None:
    """Overwrite CSV with given entries."""
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "weight_kg"])
        for e in entries:
            writer.writerow([e["date"], e["weight_kg"]])


def get_weight_input() -> float:
    """Prompt until a valid weight (kg) is entered."""
    while True:
        raw = input("Enter your weight (kg): ").strip()
        try:
            weight = float(raw)
            if 1 < weight < 300:
                return weight
            print("Please enter a realistic weight (1–300 kg).")
        except ValueError:
            print("Invalid input — please enter a number, e.g. 70.5")


def parse_entries(entries: list[dict]) -> tuple[list, list[float]]:
    """Return (dates, weights) as parallel lists."""
    dates = [datetime.strptime(e["date"], "%Y-%m-%d").date() for e in entries]
    weights = [float(e["weight_kg"]) for e in entries]
    return dates, weights


def trend_line(
    dates: list, weights: list[float]
) -> tuple[list, np.ndarray, float, float]:
    """Fit a linear trend; return (trend_dates, trend_y, slope_per_day, r²)."""
    x = np.array([(d - dates[0]).days for d in dates], dtype=float)
    slope, intercept, r, *_ = stats.linregress(x, weights)
    trend_dates = [dates[0] + timedelta(days=i) for i in range(int(x[-1]) + 1)]
    trend_y = intercept + slope * np.arange(len(trend_dates))
    return trend_dates, trend_y, slope, r ** 2


def moving_average(weights: list[float], window: int = MOVING_AVG_WINDOW) -> np.ndarray | None:
    """Return simple moving average, or None if insufficient data."""
    if len(weights) < window:
        return None
    return np.convolve(weights, np.ones(window) / window, mode="valid")


def new_low_indices(weights: list[float]) -> list[int]:
    """Return indices where a new minimum was reached."""
    lows, current_min = [], float("inf")
    for i, w in enumerate(weights):
        if w < current_min:
            current_min = w
            lows.append(i)
    return lows


def plot_weight_history(entries: list[dict]) -> None:
    """Render the weight history chart with trend and daily-change subplots."""
    if len(entries) < 2:
        print("Need at least 2 entries to generate a chart.")
        return

    dates, weights = parse_entries(entries)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    # ── Main chart ────────────────────────────────────────────────────────────
    ax1.plot(dates, weights, "bo-", lw=2, ms=8, label="Daily Weight", alpha=0.7)

    has_trend = len(dates) >= MIN_ENTRIES_FOR_TREND
    if has_trend:
        tx, ty, slope, r2 = trend_line(dates, weights)
        weekly = abs(slope * 7)
        ax1.plot(tx, ty, "r--", lw=2, label=f"Trend ({weekly:.2f} kg/week loss)")

        ma = moving_average(weights)
        if ma is not None:
            ax1.plot(
                dates[MOVING_AVG_WINDOW - 1:], ma, "g-", lw=2,
                label=f"{MOVING_AVG_WINDOW}-day Average", alpha=0.8
            )

    # Mark new lows
    lows = new_low_indices(weights)
    for i, idx in enumerate(lows):
        ax1.plot(
            dates[idx], weights[idx], "g*", ms=15,
            label="New Low" if i == 0 else "_nolegend_"
        )

    ax1.set(ylabel="Weight (kg)", title="Your Weight Progress — Focus on the Trend!")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # ── Daily-change bar chart ────────────────────────────────────────────────
    changes = np.diff(weights)
    change_dates = dates[1:]
    colors = ["green" if c < 0 else "red" for c in changes]
    ax2.bar(change_dates, changes, color=colors, alpha=0.6, width=0.7)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set(
        xlabel="Date", ylabel="Daily Change (kg)",
        title="Daily Fluctuations — Don't Stress These!"
    )
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    for ax in (ax1, ax2):
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_loss = weights[0] - weights[-1]
    print(f"""
{'='*50}
📊 CURRENT SUMMARY
{'='*50}
Current weight : {weights[-1]:.2f} kg
Starting weight: {weights[0]:.2f} kg
Total change   : {'-' if total_loss > 0 else '+'}{abs(total_loss):.2f} kg
Days tracking  : {len(dates)}""")

    if has_trend:
        _, _, slope, r2 = trend_line(dates, weights)
        print(f"Weekly trend   : {abs(slope * 7):.2f} kg/week loss")
        print(f"Trend strength : R² = {r2:.2f}  (1.0 = perfect)")
        if slope < 0 and weights[-1] > GOAL_WEIGHT_KG:
            weeks = (weights[-1] - GOAL_WEIGHT_KG) / abs(slope * 7)
            print(f"\n🎯 At this rate, you'll reach {GOAL_WEIGHT_KG} kg in {weeks:.1f} weeks")

    print("\n💡 Focus on the RED trend line, not the blue daily dots!")


def log_weight(entries: list[dict], date_today: str) -> list[dict]:
    """Handle today's weight entry; returns updated entries list."""
    already_logged = any(e["date"] == date_today for e in entries)

    if already_logged:
        print(f"\n⚠️  Weight already logged for {date_today}")
        if input("Overwrite? (y/n): ").strip().lower() != "y":
            return entries
        entries = [e for e in entries if e["date"] != date_today]

    print(f"\n📝 Logging for {date_today}")
    weight = get_weight_input()
    entries.append({"date": date_today, "weight_kg": str(weight)})
    write_entries(entries)
    print(f"✓ Logged: {date_today}, {weight} kg")
    return entries


def main() -> None:
    date_today = datetime.now().strftime("%Y-%m-%d")

    initialize_csv()
    entries = read_entries()

    if entries:
        last = entries[-1]
        print(f"📁 {len(entries)} previous entries  |  Last: {last['date']} — {last['weight_kg']} kg")

    entries = log_weight(entries, date_today)

    if len(entries) >= 2:
        plot_weight_history(entries)
    elif len(entries) == 1:
        print(
            "\n🎉 First entry logged! Come back tomorrow to start seeing your trend.\n"
            "Your graph will show:\n"
            "  • Daily weights (blue dots)\n"
            "  • Trend line (red dashed) — THIS is what matters\n"
            f"  • {MOVING_AVG_WINDOW}-day average (green) — smooths out noise\n"
            "  • Daily changes (bottom chart) — don't stress these!"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Keep up the great work!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("If this persists, check the CSV file isn't open in another program.")
