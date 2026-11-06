import time
import random

BASE = 51975

# All divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1
_DIVS = []
for a in range(4):          # 3^a
    for b in range(3):      # 5^b
        for c in range(2):  # 7^c
            for d in range(2):  # 11^d
                v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if v > 1:
                    _DIVS.append(v)
_DIVS.sort()
N = len(_DIVS)

# Precompute residue positions (indices in [0, BASE)) for each divisor and residue
# _POS[i] = list of positions for divisor _DIVS[i]
_POS = []
for m in _DIVS:
    lst = []
    for r in range(m):
        lst.append(range(r, BASE, m))
    _POS.append(lst)


def _score(covered):
    # number of residues covered by no class
    return sum(1 for x in covered if not x)


def _greedy(covered, order, start_idx):
    """Apply classes following order (list of divisor indices) starting from start_idx.
    For each, pick the residue covering the most currently-uncovered positions.
    Returns (classes, new_covered)."""
    classes = []
    cov = bytearray(covered)
    pos = _POS
    divs = _DIVS
    for i in order[start_idx:]:
        m = divs[i]
        best_r = 0
        best_c = -1
        for r in range(m):
            c = 0
            for p in pos[i][r]:
                if not cov[p]:
                    c += 1
            if c > best_c:
                best_c = c
                best_r = r
        classes.append([m, best_r])
        for p in pos[i][best_r]:
            cov[p] = 1
    return classes, cov


def _full_greedy(order):
    """Full greedy from scratch over a given order of divisor indices."""
    cov = bytearray(BASE)
    classes = []
    pos = _POS
    divs = _DIVS
    for i in order:
        m = divs[i]
        best_r = 0
        best_c = -1
        for r in range(m):
            c = 0
            for p in pos[i][r]:
                if not cov[p]:
                    c += 1
            if c > best_c:
                best_c = c
                best_r = r
        classes.append([m, best_r])
        for p in pos[i][best_r]:
            cov[p] = 1
    return classes, cov


def construct():
    deadline = time.time() + 110.0
    best_classes = None
    best_score = BASE + 1

    # ---- Phase 1: large-first greedy (deterministic) ----
    # Order: largest moduli first
    order = list(range(N - 1, -1, -1))
    classes, cov = _full_greedy(order)
    s = _score(cov)
    if s < best_score:
        best_score = s
        best_classes = [c[:] for c in classes]
    if best_score == 0:
        return best_classes

    # ---- Phase 2: randomized large-first greedy with restarts ----
    # Build a base order: large moduli first, with random shuffling within
    # "tiers" so that large ones generally come before small ones.
    # Tier by size: sort descending, then apply light random perturbation.
    rng = random.Random(12345)
    # descending list
    desc = list(range(N - 1, -1, -1))

    while time.time() < deadline - 5.0:
        # random order biased toward large-first:
        # take desc and do a few random adjacent swaps (keep large mostly first)
        order = desc[:]
        # number of swaps grows with iterations for diversity
        for _ in range(20):
            i = rng.randrange(N - 1)
            order[i], order[i + 1] = order[i + 1], order[i]
        classes, cov = _full_greedy(order)
        s = _score(cov)
        if s < best_score:
            best_score = s
            best_classes = [c[:] for c in classes]
            if best_score == 0:
                return best_classes

    return best_classes
