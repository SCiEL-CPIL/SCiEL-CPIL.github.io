"""Optimize over geometric-progressions U = {floor(b^k)} to maximize the
sumset/difference-set exponent lower bound."""
import math
import time

# EVOLVE-BLOCK-START
def _score_of(U):
    """Return the raw lower-bound score for a concrete set U (must contain 0)."""
    if not U:
        return 0.0
    u = sorted(set(U))
    if 0 not in u:
        u.append(0)
    n = len(u)
    mx = u[-1]
    if mx <= 0:
        return 0.0
    # difference set U - U
    diffs = set()
    for a in u:
        for b in u:
            diffs.add(a - b)
    # sum set U + U
    sums = set()
    for a in u:
        for b in u:
            sums.add(a + b)
    denom = 2.0 * mx + 1.0
    if denom <= 1.0:
        return 0.0
    ratio = len(diffs) / float(len(sums))
    if ratio <= 0.0:
        return 0.0
    return 1.0 + (math.log(ratio) / math.log(denom))

def _build(b, m):
    """Construct U = {floor(b^k) | k=0..m} as a list containing 0."""
    U = [0]
    v = 1.0
    for k in range(1, m + 1):
        v *= b
        U.append(int(v))
    return U

def construct():
    deadline = time.time() + 100.0
    best_score = 0.0
    best_set = [0, 1, 3]

    def consider(U):
        nonlocal best_score, best_set
        s = _score_of(U)
        if s > best_score:
            best_score = s
            best_set = U

    # Coarse sweep over base b and number of terms m.
    # Large b with many terms pushes the score toward 2.
    b_values = [2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0,
                30.0, 50.0, 80.0, 150.0, 300.0, 600.0, 1200.0, 3000.0,
                6000.0, 12000.0, 30000.0, 60000.0, 150000.0, 400000.0]
    m_values = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]

    for b in b_values:
        if time.time() > deadline:
            break
        for m in m_values:
            if time.time() > deadline:
                break
            U = _build(b, m)
            # Keep max(U) reasonably bounded to avoid pathological sizes.
            if U[-1] > 10**12:
                break  # larger m only grows max further for this b
            consider(U)

    # Refine around the best found base with finer b steps and more m.
    if best_score > 0.0:
        # find a base near the current best by re-scanning top candidates
        for b in b_values:
            if time.time() > deadline:
                break
            for m in range(10, 45):
                if time.time() > deadline:
                    break
                U = _build(b, m)
                if U[-1] > 10**12:
                    break
                consider(U)

    return best_set
# EVOLVE-BLOCK-END
