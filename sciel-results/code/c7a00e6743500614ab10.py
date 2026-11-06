import time
import random
import numpy as np

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

# Precompute: for each divisor index i, and residue r, the list of positions
_POS = []
for m in _DIVS:
    lst = []
    for r in range(m):
        lst.append(np.arange(r, BASE, m, dtype=np.int32))
    _POS.append(lst)


def _score(covered):
    return int(np.count_nonzero(covered == 0))


def _full_greedy(order, rng=None):
    """Greedy: for each divisor in order, pick residue covering most uncovered."""
    cov = np.zeros(BASE, dtype=np.uint8)
    classes = []
    for i in order:
        m = _DIVS[i]
        best_r = 0
        best_c = -1
        # Vectorized: for each residue r, count uncovered positions
        # _POS[i][r] is an array of positions
        # We want sum of (cov[pos] == 0) for pos in _POS[i][r]
        # Do it in a loop but use numpy for the count
        for r in range(m):
            positions = _POS[i][r]
            c = int(np.count_nonzero(cov[positions] == 0))
            if c > best_c:
                best_c = c
                best_r = r
        classes.append([m, best_r])
        cov[_POS[i][best_r]] = 1
    return classes, cov


def _local_search(classes, deadline, rng):
    """Try to improve by changing residues of individual classes."""
    best_score = _score(np.zeros(BASE, dtype=np.uint8))  # placeholder, we'll recompute
    # Build initial coverage
    cov = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        idx = _DIVS.index(m)
        cov[_POS[idx][r]] = 1
    best_score = _score(cov)
    best_classes = [c[:] for c in classes]
    
    while time.time() < deadline:
        improved = False
        # Try changing residue of a random class
        for _ in range(50):
            if time.time() >= deadline:
                break
            idx = rng.randrange(len(classes))
            m, r = classes[idx]
            div_idx = _DIVS.index(m)
            # Try all residues
            best_r = r
            best_c = best_score
            for new_r in range(m):
                if new_r == r:
                    continue
                # Compute new coverage score
                # Remove old, add new
                old_positions = _POS[div_idx][r]
                new_positions = _POS[div_idx][new_r]
                # Count how many new positions are currently uncovered
                gain = int(np.count_nonzero(cov[new_positions] == 0))
                # Count how many old positions are covered only by this class
                # This is tricky - we need to know if old positions are covered by other classes
                # Approximate: just try it
                cov[old_positions] = 0
                cov[new_positions] = 1
                new_score = _score(cov)
                cov[old_positions] = 1  # restore
                cov[new_positions] = 0  # restore
                if new_score < best_c:
                    best_c = new_score
                    best_r = new_r
            if best_r != r:
                # Apply the change
                classes[idx] = [m, best_r]
                cov[_POS[div_idx][r]] = 0
                cov[_POS[div_idx][best_r]] = 1
                best_score = best_c
                best_classes = [c[:] for c in classes]
                improved = True
        if not improved:
            break
    return best_classes, best_score


def construct():
    deadline = time.time() + 110.0
    best_classes = None
    best_score = BASE + 1
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy with different orderings
    # Ordering 1: smallest moduli first (they cover more per class)
    order_asc = list(range(N))
    classes, cov = _full_greedy(order_asc)
    s = _score(cov)
    if s < best_score:
        best_score = s
        best_classes = [c[:] for c in classes]
    
    # Ordering 2: largest moduli first
    order_desc = list(range(N - 1, -1, -1))
    classes, cov = _full_greedy(order_desc)
    s = _score(cov)
    if s < best_score:
        best_score = s
        best_classes = [c[:] for c in classes]
    
    # Phase 2: Randomized greedy
    while time.time() < deadline - 30.0:
        # Random permutation
        order = list(range(N))
        rng.shuffle(order)
        classes, cov = _full_greedy(order)
        s = _score(cov)
        if s < best_score:
            best_score = s
            best_classes = [c[:] for c in classes]
            if best_score == 0:
                return best_classes
    
    # Phase 3: Local search on best solution
    if best_classes is not None:
        deadline_ls = time.time() + 105.0
        best_classes, best_score = _local_search(best_classes, deadline_ls, rng)
    
    return best_classes
