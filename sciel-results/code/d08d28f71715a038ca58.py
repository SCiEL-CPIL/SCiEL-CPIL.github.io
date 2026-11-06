"""Fresh design: fast greedy + time-bounded local search for minimal-diameter Sidon set covering 1..30."""
import time
import random

_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def _valid(s):
    """Check Sidon (all positive pairwise differences distinct) and coverage 1..30."""
    n = len(s)
    diffs = set()
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            d = s[j] - si
            if d > 30:
                break
            if d in diffs:
                return False
            diffs.add(d)
    return len(diffs) == 30


def _greedy():
    """Greedy construction: add elements to realize each missing difference."""
    a = [0]
    diffs = set()

    def _all_diffs(arr):
        out = set()
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                out.add(arr[j] - arr[i])
        return out

    for d in range(1, 31):
        if d in diffs:
            continue
        # Try adding a single element next to existing ones
        found = False
        for v in sorted(a):
            for cand in (v + d, v - d):
                if cand < 0 or cand in a:
                    continue
                trial = sorted(a + [cand])
                td = _all_diffs(trial)
                if len(td) == len(trial) * (len(trial) - 1) // 2:
                    a = trial
                    diffs = td
                    found = True
                    break
            if found:
                break
        if found:
            continue
        # Add two elements far out
        y = 2 * max(a) + 3 * d + 1
        while True:
            trial = sorted(a + [y, y + d])
            td = _all_diffs(trial)
            if len(td) == len(trial) * (len(trial) - 1) // 2:
                a = trial
                diffs = td
                break
            y += 1
    return a


def _local_search(best, deadline):
    """Time-bounded local search: relocate and remove elements to shrink diameter."""
    best = list(best)
    best_diam = max(best)
    rng = random.Random(12345)

    while time.time() < deadline:
        s = best[:]
        # Randomly perturb: relocate a random element
        if len(s) > 2:
            idx = rng.randrange(1, len(s))
            old = s[idx]
            # Try moving it to a random small position
            for _ in range(200):
                cand = rng.randrange(0, best_diam + 5)
                if cand in s:
                    continue
                trial = sorted(s[:idx] + [cand] + s[idx + 1:])
                if _valid(trial):
                    d = max(trial)
                    if d < best_diam:
                        best = trial
                        best_diam = d
                    break
        # Try removing an element (if still valid)
        if len(s) > 2:
            for idx in range(len(s) - 1, 1, -1):
                trial = s[:idx] + s[idx + 1:]
                if _valid(trial):
                    d = max(trial)
                    if d < best_diam:
                        best = trial
                        best_diam = d
                    break
        # Occasionally do a full greedy restart with different seed
        if rng.random() < 0.1:
            alt = _greedy()
            if max(alt) < best_diam:
                best = alt
                best_diam = max(alt)
    return best


def construct():
    start = time.time()
    deadline = start + _DEADLINE

    # Initial greedy construction
    best = _greedy()
    best_diam = max(best)

    # Local search to improve
    best = _local_search(best, deadline)

    return best
