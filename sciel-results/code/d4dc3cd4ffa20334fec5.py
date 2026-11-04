import time
from math import log, floor, log1p
import random
import numpy as np

# Precomputed optimal Golomb rulers (smaller = better max for given n)
GOLOMB_RULERS = {
    2: [0, 1],
    3: [0, 1, 3],
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 10, 12],
    6: [0, 1, 4, 10, 12, 17],
    7: [0, 1, 4, 10, 12, 17, 20],
    8: [0, 1, 4, 10, 12, 17, 20, 24],
    9: [0, 1, 4, 10, 12, 17, 20, 24, 28],
    10: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34],
    11: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41],
    12: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47],
    13: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47, 54],
    14: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47, 54, 62],
    15: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47, 54, 62, 70],
    16: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47, 54, 62, 70, 79],
    17: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47, 54, 62, 70, 79, 89],
    18: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47, 54, 62, 70, 79, 89, 100],
    19: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47, 54, 62, 70, 79, 89, 100, 112],
    20: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47, 54, 62, 70, 79, 89, 100, 112, 125],
    21: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47, 54, 62, 70, 79, 89, 100, 112, 125, 139],
    22: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47, 54, 62, 70, 79, 89, 100, 112, 125, 139, 154],
    23: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47, 54, 62, 70, 79, 89, 100, 112, 125, 139, 154, 170],
    24: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47, 54, 62, 70, 79, 89, 100, 112, 125, 139, 154, 170, 187],
    25: [0, 1, 4, 10, 12, 17, 20, 24, 28, 34, 41, 47, 54, 62, 70, 79, 89, 100, 112, 125, 139, 154, 170, 187, 205],
}

def score_set(U):
    """Compute the exact score for a set U using numpy for speed."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    arr = np.array(U, dtype=np.int64)
    
    # All pairwise sums i<=j
    i_idx = np.repeat(np.arange(n), n - np.arange(n))
    j_idx = np.tile(np.arange(n), n - np.arange(n))
    # Actually simpler:
    sums = arr[:, None] + arr[None, :]
    # We need i<=j, but for counting distinct values, full set is fine
    # (symmetric, so i<=j gives same set of values as full for sums since a+b=b+a)
    sum_vals = sums.ravel()
    sum_count = len(np.unique(sum_vals))
    
    # All pairwise differences
    diffs = arr[None, :] - arr[:, None]  # arr[j] - arr[i]
    diff_vals = diffs.ravel()
    diff_count = len(np.unique(diff_vals))
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = int(arr.max())
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def fast_score(U):
    """Fast score computation with numpy, returns -1 if invalid."""
    if len(U) < 2:
        return -1
    U = sorted(set(U))
    if 0 not in U:
        return -1
    arr = np.array(U, dtype=np.int64)
    n = len(arr)
    
    sums = arr[:, None] + arr[None, :]
    sum_count = len(np.unique(sums))
    
    diffs = arr[None, :] - arr[:, None]
    diff_count = len(np.unique(diffs))
    
    if sum_count == 0:
        return -1
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
    max_u = int(arr.max())
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def marginal_estimate(U, candidate):
    """
    Fast estimate of score change when adding candidate.
    Uses set-based counting for the incremental part.
    Returns estimated new score.
    """
    U = sorted(U)
    n = len(U)
    if candidate in U:
        return None
    
    # Existing sums and diffs as sets
    existing_sums = set()
    existing_diffs = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            existing_sums.add(ui + U[j])
    for i in range(n):
        ui = U[i]
        for j in range(i + 1, n):
            d = U[j] - ui
            existing_diffs.add(d)
            existing_diffs.add(-d)
    existing_diffs.add(0)
    
    new_sums = 0
    for i in range(n):
        s = U[i] + candidate
        if s not in existing_sums:
            new_sums += 1
    if candidate + candidate not in existing_sums:
        new_sums += 1
    
    sum_count = len(existing_sums) + new_sums
    
    new_diffs = 0
    for i in range(n):
        d = abs(candidate - U[i])
        if d > 0 and d not in existing_diffs:
            new_diffs += 2
    
    diff_count = len(existing_diffs) + new_diffs
    
    if sum_count == 0 or diff_count <= sum_count:
        return None
    
    max_u = max(max(U), candidate)
    D = 2 * max_u + 1
    if D <= 1:
        return None
    
    ratio = diff_count / sum_count
    return 1 + log(ratio) / log(D)

def poisson_sample(alpha, m, max_val, rng):
    """
    Sample m elements from heavy-tailed distribution to create large gaps.
    u_k proportional to k^(-alpha), then scale to [0, max_val].
    """
    # Sample from discrete heavy-tail: P(k) ~ k^(-alpha) for k=1..K
    K = max_val
    weights = np.array([k ** (-alpha) for k in range(1, K + 1)], dtype=np.float64)
    weights /= weights.sum()
    # Sample m-1 distinct values (excluding 0)
    samples = rng.choice(np.arange(1, K + 1), size=min(m - 1, K), replace=False, p=weights[:min(m-1, K)])
    U = sorted(set([0] + list(samples)))
    return U

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def local_search(U, deadline, max_val_limit=200, max_n_limit=26, verbose=False):
    """
    Fast local search with vectorized scoring and smooth surrogate ranking.
    """
    U = sorted(set(U))
    current_score = fast_score(U)
    best_score = current_score
    best_set = list(U)
    
    no_improve = 0
    max_no_improve = 40
    
    while time.time() < deadline - 2 and no_improve < max_no_improve:
        improved = False
        move_type = random.random()
        n = len(U)
        current_max = max(U)
        
        if move_type < 0.5:
            # ADD: rank candidates by smooth surrogate
            if n < max_n_limit:
                # Generate candidates: near gaps, near max, random
                candidates = set()
                # Gaps between consecutive elements
                for i in range(n - 1):
                    gap = U[i + 1] - U[i]
                    if gap > 1:
                        mid = U[i] + gap // 2
                        if 0 < mid <= max_val_limit and mid not in U:
                            candidates.add(mid)
                # Near max
                for delta in range(1, 15):
                    v = current_max + delta
                    if v <= max_val_limit:
                        candidates.add(v)
                # Random
                for _ in range(20):
                    v = random.randint(1, min(max_val_limit, current_max + 30))
                    if v not in U:
                        candidates.add(v)
                # Small values
                for v in range(1, min(30, current_max + 1)):
                    if v not in U:
                        candidates.add(v)
                
                if candidates:
                    # Rank by surrogate: prefer those that add more diff than sum
                    scored = []
                    for c in list(candidates)[:40]:
                        est = marginal_estimate(U, c)
                        if est is not None and est > current_score:
                            scored.append((est, c))
                    scored.sort(key=lambda x: -x[0])
                    for _, c in scored[:6]:
                        new_U = sorted(U + [c])
                        new_score = fast_score(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            max_val_limit = max(max_val_limit, current_max * 2)
                            break
        elif move_type < 0.7:
            # REMOVE
            if n > 3:
                removable = [x for x in U if x != 0]
                best_rem = None
                best_rem_score = current_score
                for c in random.sample(removable, min(6, len(removable))):
                    new_U = [x for x in U if x != c]
                    new_score = fast_score(new_U)
                    if new_score > best_rem_score:
                        best_rem_score = new_score
                        best_rem = c
                if best_rem is not None:
                    U = [x for x in U if x != best_rem]
                    current_score = best_rem_score
                    improved = True
        else:
            # REPLACE
            if n > 2:
                idx = random.randint(1, n - 1)
                old_val = U[idx]
                best_new = None
                best_new_score = current_score
                # Try nearby values
                for delta in range(-15, 16):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [U[j] if j != idx else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = fast_score(new_set)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new = new_set
                if best_new is not None:
                    U = best_new
                    current_score = best_new_score
                    improved = True
        
        if improved:
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        else:
            no_improve += 1
    
    return best_score, best_set

def final_polish(U, deadline):
    """High-precision polish with fine perturbations."""
    U = sorted(set(U))
    current_score = fast_score(U)
    best_score = current_score
    best_set = list(U)
    
    improved = True
    iterations = 0
    polish_deadline = deadline - 1
    
    while improved and time.time() < polish_deadline and iterations < 800:
        improved = False
        iterations += 1
        n = len(U)
        
        # Phase 1: Fine replacement delta -10..10
        for i in range(1, n):
            if time.time() > polish_deadline:
                break
            old_val = U[i]
            for delta in range(-10, 11):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                new_set = [U[j] if j != i else new_val for j in range(n)]
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
                new_score = fast_score(new_set)
                if new_score > current_score:
                    U = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Phase 2: Add element
        if not improved and n < 24 and time.time() < polish_deadline:
            max_val = max(U)
            try_vals = set()
            for v in range(max(1, max_val - 20), max_val + 21):
                try_vals.add(v)
            for v in range(1, max_val, max(1, max_val // 20)):
                try_vals.add(v)
            for new_val in sorted(try_vals):
                if new_val in U:
                    continue
                new_set = sorted(U + [new_val])
                new_score = fast_score(new_set)
                if new_score > current_score:
                    U = new_set
                    current_score = new_score
                    improved = True
                    break
        
        # Phase 3: Remove
        if not improved and n > 3 and time.time() < polish_deadline:
            for i in range(1, n):
                new_set = [U[j] for j in range(n) if j != i]
                new_score = fast_score(new_set)
                if new_score > current_score:
                    U = new_set
                    current_score = new_score
                    improved = True
                    break
        
        # Phase 4: Pairwise swap
        if not improved and n >= 4 and time.time() < polish_deadline:
            for i in range(1, n):
                if time.time() > polish_deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > polish_deadline:
                        break
                    old_i, old_j = U[i], U[j]
                    for di in range(-5, 6):
                        for dj in range(-5, 6):
                            if di == 0 and dj == 0:
                                continue
                            new_i, new_j = old_i + di, old_j + dj
                            if new_i <= 0 or new_j <= 0 or new_i == new_j:
                                continue
                            new_set = list(U)
                            new_set[i], new_set[j] = new_i, new_j
                            new_set = sorted(set(new_set))
                            if len(new_set) != n:
                                continue
                            new_score = fast_score(new_set)
                            if new_score > current_score:
                                U = new_set
                                current_score = new_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(U)
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    random.seed(42)
    rng = np.random.default_rng(42)
    
    best_set = [0, 1, 4, 10, 12]
    best_score = fast_score(best_set)
    
    # 1. Evaluate precomputed Golomb rulers
    for n in range(2, 26):
        if n in GOLOMB_RULERS:
            U = list(GOLOMB_RULERS[n])
            s = fast_score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # 2. Poisson/heavy-tail sampling to create large gaps
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
    m_vals = [5, 8, 12, 16, 20]
    max_vals = [50, 100, 200, 400]
    
    for alpha in alphas:
        if time.time() > deadline - 60:
            break
        for m in m_vals:
            for mv in max_vals:
                if time.time() > deadline - 60:
                    break
                for trial in range(3):
                    U = poisson_sample(alpha, m, mv, rng)
                    if len(U) < 2:
                        continue
                    s = fast_score(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)
    
    # 3. Geometric sets
    for b_int in range(12, 60):
        if time.time() > deadline - 50:
            break
        b = b_int / 10.0
        for m in range(3, 18):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = fast_score(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # 4. Local search from multiple seeds
    seeds = [best_set]
    for n in [5, 8, 12, 16, 20]:
        if n in GOLOMB_RULERS:
            seeds.append(list(GOLOMB_RULERS[n]))
    
    restarts = 0
    while time.time() < deadline - 30:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        U = list(base)
        local_score, local_set = local_search(U, deadline, max_val_limit=300, max_n_limit=26)
        if local_score > best_score:
            best_score = local_score
            best_set = list(local_set)
            if len(seeds) < 10:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set
    
    # 5. Final high-precision polish
    if time.time() < deadline - 3:
        polish_score, polish_set = final_polish(best_set, deadline)
        if polish_score > best_score:
            best_score = polish_score
            best_set = polish_set
    
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
