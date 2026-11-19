"""gta6-trophy-rarity-forecast-01 entry: trophy earn-rate forecaster.

For every GTA VI trophy, forecasts its Exophase earn rate at the case target
as five quantiles (5/25/50/75/95%), scored by pinball loss on the logit
scale. Pure Python 3.13 standard library, deterministic, no network.

Model, per trophy, fitted at run time from the case's own history:
  * Collapse the history to its distinct rate steps (Exophase refreshes move
    the rates in steps; identical consecutive reads carry no new signal).
  * Fit a least-squares line of logit(rate) against ln(1 + hours since the
    release day began), over the last 24 hours of steps (all steps if fewer).
    Logit-linear in log-time saturates, which matches story trophies rising
    toward a plateau while the tracked population keeps growing.
  * The median at the target is the fitted line read at the target, guarded
    against wild extrapolation (at most a 2.5-logit move from the last rate).
  * The spread combines the fit's prediction error at the target, a
    random-walk base that grows with the horizon, and the observed
    refresh-jump volatility between steps. A wide, honest spread: a refresh
    can move every rate at once, and the pinball loss punishes a band that
    misses it far more than a band a little too wide.
  * When the history is flat (fewer than 3 distinct steps) there is no trend
    to fit and no jump to measure, yet a refresh is overdue: the rates have
    sat unchanged for hours while the player base grows. Then the median is
    shifted down by one expected refresh dilution and the spread is widened
    to cover it. Once the history shows real steps, the data overrides this
    prior entirely.

Reads every *.in file in the input directory ($MAKEFUN_INPUT_DIR, else
/input) and writes /output/<id>.out ($MAKEFUN_OUTPUT_DIR, else /output):
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release day begins here; log-time is measured from it.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Random-walk spread, logit units per square-root day: the irreducible drift
# of a rate over the horizon not already captured by the fit and the jumps.
# Kept well below the house baseline's 0.5: the trend fit and the measured
# jump volatility already carry most of the movement, and the pinball loss
# pays for every unit of spread.
RW_SIGMA = 0.25
# Shrinkage of the fitted trend toward persistence (the last observed rate).
# The log-time line extrapolates the recent rise; the true curve saturates
# as the cohort mix stabilises, so the raw extrapolation overshoots. A
# shrunk trend keeps most of the signal with far less overshoot.
SLOPE_SHRINK = 0.4
# One refresh's dilution, in logit units, assumed when the history is flat
# and a refresh is overdue (launch night: the first refresh moved typical
# trophies about a logit). Applied only with no trend to fit.
FLAT_SHIFT = 0.9
FLAT_SIGMA = 0.9
# Spread floor / cap, in logit units.
SIGMA_MIN = 0.35
SIGMA_MAX = 1.6
# Largest median move from the last observed rate, in logit units.
MAX_MOVE = 2.5


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_epoch(when):
    return (when - EPOCH).total_seconds() / 3600.0


def distinct_steps(history):
    """Collapse consecutive identical reads into steps.

    Returns [(hours, players, rates)] with one entry per distinct rate
    vector, keeping the first read that showed it.
    """
    steps = []
    last_rates = None
    for read in history:
        rates = read["rates"]
        if rates != last_rates:
            steps.append((hours_since_epoch(parse_time(read["at"])),
                          read["players_tracked"], rates))
            last_rates = rates
    return steps


def fit_line(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_std, n, xbar, sxx)."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        return ybar, 0.0, 0.0, n, xbar, sxx
    b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
    a = ybar - b * xbar
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid, n, xbar, sxx


def forecast_trophy(steps, target_hours, horizon_days, trophy_index):
    """Return the five quantiles for one trophy (index into the rates)."""
    # Series of (hours, log-hours, logit rate) for this trophy, one per step.
    pts = []
    for hours, _players, rates in steps:
        r = rates[trophy_index]
        if r is None:
            continue
        pts.append((hours, math.log1p(hours), logit(r)))
    last_rate = steps[-1][2][trophy_index]
    if last_rate is None:
        last_rate = 1.0
    last_logit = logit(last_rate)
    target_x = math.log1p(target_hours)

    # Refresh jumps between steps, in logit units (dilution shocks).
    jumps = [pts[k][2] - pts[k - 1][2] for k in range(1, len(pts))]
    if len(jumps) >= 2:
        jbar = sum(jumps) / len(jumps)
        jump_std = math.sqrt(sum((j - jbar) ** 2 for j in jumps) / (len(jumps) - 1))
    else:
        jump_std = 0.0

    if len(pts) >= 3:
        # Fit over the steps from the last 24 hours when there are enough
        # there, else over every step.
        last_h = pts[-1][0]
        recent = [p for p in pts if p[0] >= last_h - 24.0]
        use = recent if len(recent) >= 3 else pts
        a, b, resid, n, xbar, sxx = fit_line([p[1] for p in use], [p[2] for p in use])
        # Prediction error at the target (standard prediction interval).
        if n > 2 and sxx > 0:
            lev = 1.0 + 1.0 / n + (target_x - xbar) ** 2 / sxx
        else:
            lev = 1.0
        pred_std = resid * math.sqrt(lev)
        trend_logit = a + b * target_x
        # Shrink the extrapolated trend toward persistence: the raw line
        # overshoots because the true curve saturates.
        median_logit = last_logit + SLOPE_SHRINK * (trend_logit - last_logit)
        # Guard the extrapolation: never move more than MAX_MOVE from the
        # last observed rate.
        median_logit = min(last_logit + MAX_MOVE, max(last_logit - MAX_MOVE, median_logit))
        median_logit = min(logit(99.9), max(logit(0.01), median_logit))
        sigma = math.sqrt(pred_std ** 2 + (RW_SIGMA * math.sqrt(horizon_days)) ** 2 + jump_std ** 2)
    else:
        # Flat history: no trend, no measured jumps. A refresh is overdue,
        # so shift the median down by one dilution and widen the spread.
        median_logit = last_logit - FLAT_SHIFT
        median_logit = min(logit(99.9), max(logit(0.01), median_logit))
        sigma = max(FLAT_SIGMA, RW_SIGMA * math.sqrt(horizon_days))

    sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))
    qs = []
    for z in Z:
        qs.append(round4(clamp_pct(inv_logit(median_logit + z * sigma))))
    # Rounding can break monotonicity at the extremes; enforce it.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    steps = distinct_steps(case["history"])
    target_hours = hours_since_epoch(parse_time(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    rows = []
    for i, trophy in enumerate(trophies):
        rows.append({"id": trophy["id"], "q": forecast_trophy(steps, target_hours, horizon_days, i)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
