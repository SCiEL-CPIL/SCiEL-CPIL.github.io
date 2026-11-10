import time
import random
import numpy as np
from scipy.optimize import linprog

def _max_abs_ap_sum(signs, threshold=None):
    """
    Compute the maximum absolute sum over any arithmetic progression
    contained in [1, n].
    signs: list or numpy array of length n, values -1 or 1.
    threshold: if provided, early-exit and return threshold+1 if exceeded.
    Returns the max absolute sum.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_sum = 0
    
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                continue
            idx = np.arange(s, n, d)
            cumsum = np.cumsum(signs_arr[idx])
            current_max = int(np.max(cumsum) - np.min(cumsum))
            if current_max > max_sum:
                max_sum = current_max
                if threshold is not None and max_sum > threshold:
                    return max_sum
    
    return max_sum

def _get_tight_ap_info(signs, top_k=40):
    """
    Find indices that appear in APs with sums close to the maximum.
    Returns a list of candidate indices to flip, weighted by tightness.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    
    ap_sums = []
    max_sum = 0
    
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                continue
            idx = np.arange(s, n, d)
            cumsum = np.cumsum(signs_arr[idx])
            current_max = int(np.max(cumsum) - np.min(cumsum))
            if current_max > 0:
                ap_sums.append((current_max, s, d))
            if current_max > max_sum:
                max_sum = current_max
    
    if max_sum == 0:
        return list(range(n))
    
    tight_threshold = max_sum - 1
    tight_aps = [(s, d) for (val, s, d) in ap_sums if val >= tight_threshold]
    
    idx_count = np.zeros(n, dtype=np.int32)
    for s, d in tight_aps:
        idx = s
        while idx < n:
            idx_count[idx] += 1
            idx += d
    
    candidates = np.argsort(-idx_count)[:top_k]
    return candidates.tolist()

def _solve_lp_relaxation(n, min_ap_len=4, deadline=None):
    """
    Solve LP relaxation: minimize D s.t. -D <= sum_{i in AP} s_i <= D
    for all APs of length >= min_ap_len, with -1 <= s_i <= 1.
    Returns the optimal continuous solution and D value.
    """
    # Collect all APs of length >= min_ap_len
    ap_indices = []
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                continue
            idx = np.arange(s, n, d)
            if len(idx) >= min_ap_len:
                ap_indices.append(idx)
    
    if not ap_indices:
        return None, None
    
    # LP: minimize D
    # Variables: s_0, s_1, ..., s_{n-1}, D
    # Constraints: for each AP: sum(s_i for i in AP) <= D
    #              for each AP: -sum(s_i for i in AP) <= D
    # Bounds: -1 <= s_i <= 1, D >= 0
    
    m = len(ap_indices)
    # c: objective = [0, 0, ..., 0, 1] (minimize D)
    c = np.zeros(n + 1)
    c[-1] = 1.0
    
    # A_ub: each row is one constraint
    # For AP with indices idx: sum_{i in idx} s_i - D <= 0
    #                            -sum_{i in idx} s_i - D <= 0
    A_ub = np.zeros((2 * m, n + 1))
    b_ub = np.zeros(2 * m)
    
    for k, idx in enumerate(ap_indices):
        # sum s_i - D <= 0
        A_ub[2*k, idx] = 1.0
        A_ub[2*k, -1] = -1.0
        # -sum s_i - D <= 0
        A_ub[2*k+1, idx] = -1.0
        A_ub[2*k+1, -1] = -1.0
    
    # Bounds: -1 <= s_i <= 1, D >= 0
    bounds = [(-1, 1)] * n + [(0, None)]
    
    # Check deadline
    if deadline is not None and time.time() > deadline:
        return None, None
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            s_cont = result.x[:n]
            D_opt = result.x[-1]
            return s_cont, D_opt
        else:
            return None, None
    except Exception:
        return None, None

def _round_continuous_to_signs(s_cont, rng):
    """
    Convert continuous solution to discrete signs using randomized rounding.
    """
    n = len(s_cont)
    signs = []
    for i in range(n):
        val = s_cont[i]
        # Probabilistic rounding: P(sign=+1) = (val+1)/2
        p = (val + 1) / 2.0
        p = max(0.0, min(1.0, p))
        if rng.random() < p:
            signs.append(1)
        else:
            signs.append(-1)
    return signs

def _local_search(signs, current_score, deadline, rng, top_k=40):
    """
    Local search to improve the discrete solution.
    Returns improved signs and score.
    """
    n = len(signs)
    best_signs = signs[:]
    best_score = current_score
    no_improve_count = 0
    max_no_improve = 30
    iteration = 0
    
    candidates = _get_tight_ap_info(best_signs, top_k=top_k)
    
    while time.time() < deadline:
        iteration += 1
        
        if no_improve_count > max_no_improve:
            # Restart with perturbation
            for _ in range(rng.randint(3, 8)):
                idx = rng.randint(0, n - 1)
                best_signs[idx] = -best_signs[idx]
            current_score = _max_abs_ap_sum(best_signs)
            if current_score < best_score:
                best_score = current_score
            no_improve_count = 0
            candidates = _get_tight_ap_info(best_signs, top_k=top_k)
            continue
        
        if iteration % 20 == 0:
            candidates = _get_tight_ap_info(best_signs, top_k=top_k)
            if not candidates:
                candidates = list(range(n))
        
        # Try single flip from tight indices
        if candidates:
            idx = candidates[rng.randint(0, len(candidates) - 1)]
        else:
            idx = rng.randint(0, n - 1)
        
        old_val = best_signs[idx]
        best_signs[idx] = -old_val
        new_score = _max_abs_ap_sum(best_signs, threshold=best_score)
        
        if new_score < best_score:
            best_score = new_score
            no_improve_count = 0
        else:
            best_signs[idx] = old_val
            no_improve_count += 1
    
    return best_signs, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s kill
    
    n = 256
    rng = random.Random(42)
    
    # Phase 1: LP relaxation to get principled initialization
    lp_solution, lp_D = _solve_lp_relaxation(n, min_ap_len=4, deadline=start_time + 20.0)
    
    initializations = []
    
    if lp_solution is not None:
        # Multiple random roundings of LP solution
        for _ in range(3):
            initializations.append(_round_continuous_to_signs(lp_solution, rng))
        # Also try threshold rounding
        initializations.append([1 if v >= 0 else -1 for v in lp_solution])
    else:
        # Fallback initializations
        initializations.append([1 if i % 2 == 0 else -1 for i in range(n)])
        tm = []
        for i in range(n):
            tm.append(1 if bin(i).count('1') % 2 == 0 else -1)
        initializations.append(tm)
        for _ in range(3):
            initializations.append([rng.choice((-1, 1)) for _ in range(n)])
    
    # Add some heuristic initializations as backup
    initializations.append([1 if i % 2 == 0 else -1 for i in range(n)])
    initializations.append([1 if (i // 2) % 2 == 0 else -1 for i in range(n)])
    
    # Evaluate all initializations
    best_signs = None
    best_score = float('inf')
    
    for init in initializations:
        if time.time() > deadline:
            break
        score = _max_abs_ap_sum(init)
        if score < best_score:
            best_score = score
            best_signs = init[:]
    
    # Phase 2: Local search from best initialization
    current_deadline = start_time + 48.0  # Leave 2s margin
    best_signs, best_score = _local_search(best_signs, best_score, current_deadline, rng, top_k=40)
    
    # Phase 3: If time remains, try a few more LP-guided restarts
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Try different LP parameters
        for min_len in [3, 5]:
            if time.time() > deadline - 2.0:
                break
            lp_sol2, _ = _solve_lp_relaxation(n, min_ap_len=min_len, deadline=time.time() + 10.0)
            if lp_sol2 is not None:
                for _ in range(2):
                    if time.time() > deadline - 2.0:
                        break
                    cand = _round_continuous_to_signs(lp_sol2, rng)
                    score = _max_abs_ap_sum(cand)
                    if score < best_score:
                        best_score = score
                        best_signs = cand[:]
        
        # Final local search polish
        if time.time() < deadline - 3.0:
            best_signs, best_score = _local_search(best_signs, best_score, deadline - 1.0, rng, top_k=40)
    
    return best_signs
