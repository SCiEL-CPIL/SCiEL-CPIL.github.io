"""Difference triangle set (4,7) — randomized greedy with multiple restarts."""
import time
import random

# EVOLVE-BLOCK-START
def _build_rows_random(n_rows, k, rng, order_bias=0.0, jitter=0.0):
    """Build n_rows increasing rows starting at 0 using a randomized greedy rule.
    order_bias: probability of picking the *next* smallest feasible value vs a random feasible one.
    jitter: extra random skip probability to diversify.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                if rng.random() < order_bias:
                    row.append(x)
                    used |= new
                elif rng.random() > jitter:
                    row.append(x)
                    used |= new
                # else: skip this candidate, try the next x
        rows.append(row)
    return rows


def _max_entry(rows):
    return max(max(r) for r in rows)


def construct(n_rows, k):
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Deterministic greedy first (fast baseline)
    best = _build_rows_random(n_rows, k, random.Random(0), order_bias=1.0, jitter=0.0)
    best_score = _max_entry(best)

    # Randomized restarts with varied heuristics
    rng = random.Random(12345)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Vary the bias: mostly greedy-ish but with some randomness
        ob = rng.choice([1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.0])
        jt = rng.choice([0.0, 0.0, 0.1, 0.2, 0.3, 0.5])
        trial = _build_rows_random(n_rows, k, rng, order_bias=ob, jitter=jt)
        sc = _max_entry(trial)
        if sc < best_score:
            best_score = sc
            best = trial

    return best
# EVOLVE-BLOCK-END
