import time
import random
import itertools

# EVOLVE-BLOCK-START

def _is_sidon_diag(subset):
    """All unordered pair sums (i<=j) unique."""
    seen = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(ambient):
    """Exact largest subset with unique pair sums (incl diagonals)."""
    n = len(ambient)
    for k in range(n, 0, -1):
        for combo in itertools.combinations(ambient, k):
            if _is_sidon_diag(list(combo)):
                return k
    return 0

def _has_k_sidon(ambient, k):
    """True if any k-subset is Sidon (diag)."""
    for combo in itertools.combinations(ambient, k):
        if _is_sidon_diag(list(combo)):
            return True
    return False

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    best_val = None
    best_set = list(range(12))

    # Baseline
    base_val = _max_sidon_size(best_set)
    best_val = base_val

    rng = random.Random(20240614)

    def consider(ambient):
        nonlocal best_val, best_set
        val = _max_sidon_size(ambient)
        if best_val is None or val < best_val:
            best_val = val
            best_set = list(ambient)
            return True
        return False

    # Random search: try to find ambient sets with small max Sidon subset.
    # Use fast pre-filters: if best_val is 4, only accept sets with no 4-Sidon subset (max<=3).
    while time.time() < deadline:
        s = set(rng.sample(range(0, 129), 12))
        ambient = sorted(s)

        if best_val is not None and best_val <= 4:
            # To beat 4 we need max <= 3, i.e. no 4-Sidon subset.
            if not _has_k_sidon(ambient, 4):
                consider(ambient)
                if best_val <= 3:
                    break
        else:
            consider(ambient)
            if best_val is not None and best_val <= 3:
                break

    # Structured families (arithmetic progressions and near-APs)
    if time.time() < deadline:
        for base in range(0, 120):
            if time.time() > deadline:
                break
            for step in (2, 3, 4):
                vals = [base + i * step for i in range(12)]
                if max(vals) <= 128 and min(vals) >= 0:
                    consider(sorted(vals))
                    if best_val is not None and best_val <= 3:
                        return list(best_set)

    return list(best_set)

# EVOLVE-BLOCK-END
