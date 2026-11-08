import time
import math
import numpy as np

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def _fast_score(mask):
    """Fast score using numpy boolean convolutions. mask is 1D bool array length M+1."""
    n = mask.sum()
    if n < 2:
        return 0.0
    M = len(mask) - 1
    # sumset via convolution
    c = np.convolve(mask.astype(np.int8), mask.astype(np.int8))
    sumset_count = (c > 0).sum()
    # difference set: for each shift s in [-(M), M], check overlap of mask[s:] and mask[:M-s]
    diffset_count = 0
    # s = 0 always (a-a)
    diffset_count += 1
    for s in range(1, M + 1):
        # a - b = s  <=>  a = b + s, so mask[b+s] & mask[b] for b in [0, M-s]
        if np.any(mask[s:] & mask[:M - s + 1]):
            diffset_count += 1
        # a - b = -s  <=>  b = a + s, same condition (symmetric)
        if np.any(mask[s:] & mask[:M - s + 1]):
            diffset_count += 1
    ratio = diffset_count / sumset_count
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * M + 1)
    if log_denom <= 0:
        return 0.0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Seed from small exhaustive search
    best_U = [0, 1, 3, 6, 9]
    best_score = compute_score(best_U)

    # Quick exhaustive over small sets to get a good starting point
    from itertools import combinations
    pool = list(range(1, 21))
    for size in [3, 4, 5]:
        for combo in combinations(pool, size):
            if time.time() > deadline:
                break
            U = [0] + list(combo)
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
        if time.time() > deadline:
            break

    # Simulated annealing over binary membership vectors
    # State: mask of length M+1, mask[0]=True always
    M = 200  # max element bound
    mask = np.zeros(M + 1, dtype=bool)
    mask[0] = True
    for v in best_U:
        if v <= M:
            mask[v] = True

    current_score = _fast_score(mask)
    best_mask = mask.copy()
    best_sa_score = current_score

    # Temperature schedule
    T0 = 0.05
    Tend = 0.0005
    n_steps = 150000
    # geometric cooling
    if T0 > Tend:
        cool_factor = (Tend / T0) ** (1.0 / max(1, n_steps))
    else:
        cool_factor = 1.0

    T = T0
    rng = np.random.default_rng(12345)

    for step in range(n_steps):
        if time.time() > deadline - 5:
            break

        # Random move: flip one bit (not bit 0)
        pos = rng.integers(1, M + 1)
        mask[pos] = not mask[pos]

        new_score = _fast_score(mask)

        if new_score >= current_score:
            current_score = new_score
            if new_score > best_sa_score:
                best_sa_score = new_score
                best_mask = mask.copy()
        else:
            # Accept with probability exp((new - cur)/T)
            delta = new_score - current_score
            if delta < -T * 50:  # too bad, reject
                mask[pos] = not mask[pos]  # revert
            else:
                accept_prob = math.exp(delta / T)
                if rng.random() < accept_prob:
                    current_score = new_score
                else:
                    mask[pos] = not mask[pos]  # revert

        T *= cool_factor

    # Extract best set from best_mask
    best_set = [i for i in range(M + 1) if best_mask[i]]

    # Final polish: try adding/removing single elements around the best set
    base = set(best_set)
    base_score = best_sa_score
    for _ in range(3):  # a few polish rounds
        if time.time() > deadline - 2:
            break
        improved = False
        # Try adding each candidate in a range
        for cand in range(1, M + 1):
            if cand in base:
                continue
            new_set = base | {cand}
            s = compute_score(list(new_set))
            if s > base_score:
                base = new_set
                base_score = s
                improved = True
        # Try removing each element (not 0)
        for cand in list(base):
            if cand == 0:
                continue
            new_set = base - {cand}
            if len(new_set) < 2:
                continue
            s = compute_score(list(new_set))
            if s > base_score:
                base = new_set
                base_score = s
                improved = True
        if not improved:
            break

    # Compare with the earlier exhaustive best
    if base_score > best_score:
        best_U = sorted(base)
        best_score = base_score

    return best_U
