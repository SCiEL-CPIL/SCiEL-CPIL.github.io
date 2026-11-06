"""Odd covering systems with moduli dividing 51975 (fewest uncovered residues)."""
import time
import random

BASE = 51975
# All divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1.
DIVS = []
for a in range(4):
    for b in range(3):
        for c in range(2):
            for d in range(2):
                v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if v > 1:
                    DIVS.append(v)
DIVS.sort(reverse=True)  # large first

# Precompute the set of residues covered by class (m, r): {r, r+m, ...}
# stored as a Python int bitmask over 0..BASE-1.
MASKS = {}
for m in DIVS:
    row = []
    for r in range(m):
        mask = 0
        x = r
        while x < BASE:
            mask |= (1 << x)
            x += m
        row.append(mask)
    MASKS[m] = row

FULL = (1 << BASE) - 1


def _score(covered_mask):
    return (FULL ^ covered_mask).bit_count()


def _construct_greedy(order, covered, classes, rng):
    """Add classes for moduli in `order`, choosing residue to maximize newly covered."""
    cur = covered
    for m in order:
        row = MASKS[m]
        best_r = -1
        best_val = -1
        # small moduli: scan all; large: sample
        if m <= 110:
            cands = range(m)
        else:
            k = min(m, 160)
            cands = rng.sample(range(m), k)
        for r in cands:
            v = (row[r] & ~cur).bit_count()
            if v > best_val:
                best_val = v
                best_r = r
        if best_r < 0:
            best_r = 0
        mr = MASKS[m][best_r]
        cur |= mr
        classes.append((m, best_r))
    return cur, classes


def _local_search(covered, classes, budget):
    """Improve via residue swaps, modulus swaps, and dropping redundant classes."""
    n = len(classes)
    cur = covered
    cls = list(classes)
    improved = True
    while improved and time.time() < budget:
        improved = False
        # 1) residue swaps
        for i in range(n):
            m = cls[i][0]
            row = MASKS[m]
            r0 = cls[i][1]
            old = row[r0]
            base_others = cur & ~old
            # candidate residues (sample for large m)
            if m <= 110:
                cands = [r for r in range(m) if r != r0]
            else:
                cands = random.sample([r for r in range(m) if r != r0], min(120, m - 1))
            best_r = r0
            best_val = _score(cur)
            for r in cands:
                cand = base_others | row[r]
                s = _score(cand)
                if s < best_val:
                    best_val = s
                    best_r = r
            if best_r != r0:
                cur = base_others | row[best_r]
                cls[i] = (m, best_r)
                improved = True
        # 2) drop redundant classes
        for i in range(n - 1, -1, -1):
            m, r = cls[i]
            others = cur & ~MASKS[m][r]
            if _score(others) == _score(cur):
                cur = others
                cls.pop(i)
                improved = True
        # 3) modulus swaps (replace one modulus with an unused one)
        used = set(m for m, _ in cls)
        avail = [m for m in DIVS if m not in used]
        if avail and time.time() < budget:
            for i in range(n):
                m0, r0 = cls[i]
                others = cur & ~MASKS[m0][r0]
                best = (m0, r0, _score(cur))
                for m1 in avail[:40]:
                    row = MASKS[m1]
                    if m1 <= 110:
                        cands = range(m1)
                    else:
                        cands = random.sample(range(m1), min(80, m1))
                    for r in cands:
                        cand = others | row[r]
                        s = _score(cand)
                        if s < best[2]:
                            best = (m1, r, s)
                if best[0] != m0:
                    cur = others | MASKS[best[0]][best[1]]
                    cls[i] = (best[0], best[1])
                    improved = True
    return cur, cls


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s
    rng = random.Random(12345)

    best_score = None
    best_classes = None

    def try_order(order):
        nonlocal best_score, best_classes
        covered, classes = _construct_greedy(order, 0, [], rng)
        s = _score(covered)
        if best_score is None or s < best_score:
            best_score = s
            best_classes = list(classes)
        if time.time() < deadline:
            covered, classes = _local_search(covered, classes, deadline)
            s = _score(covered)
            if s < best_score:
                best_score = s
                best_classes = list(classes)

    # Primary: large moduli first
    try_order(DIVS)

    # Randomized restarts while time remains
    guard = 0
    while time.time() < deadline - 3.0 and guard < 60:
        guard += 1
        order = DIVS[:]
        rng.shuffle(order)
        try_order(order)

    # Final polish on the best solution
    if best_classes is not None and time.time() < deadline:
        cov = 0
        for m, r in best_classes:
            cov |= MASKS[m][r]
        cov, cls = _local_search(cov, best_classes, deadline)
        s = _score(cov)
        if s < best_score:
            best_score = s
            best_classes = list(cls)

    if best_classes is None:
        # Fallback: simple deterministic
        best_classes = [[m, 0] for m in DIVS]

    return [[m, r] for m, r in best_classes]
