"""Optimize over geometric-progressions U = {floor(b^k)} to maximize the
sumset/difference-set exponent lower bound."""
import math
import time
import itertools

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
    best_set = [0, 1]
    
    # Coarse grid search
    b_start = 1.1
    b_end = 3.0
    b_step = 0.01
    m_max = 20
    
    b = b_start
    while b <= b_end:
        for m in range(2, m_max + 1):
            if time.time() > deadline:
                return best_set
            U = _build(b, m)
            s = _score_of(U)
            if s > best_score:
                best_score = s
                best_set = U
        b += b_step
        
    # Fine grid search if time permits
    if time.time() < deadline - 30:
        b_step_fine = 0.001
        b = b_start
        while b <= b_end:
            for m in range(2, m_max + 1):
                if time.time() > deadline:
                    return best_set
                U = _build(b, m)
                s = _score_of(U)
                if s > best_score:
                    best_score = s
                    best_set = U
            b += b_step_fine

    return best_set
# EVOLVE-BLOCK-END
