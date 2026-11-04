"""QAP solver using MCMC with block-reversal moves (Gibbs/Metropolis-Hastings)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute exact QAP cost for assignment p (facility i at location p[i])."""
    # p is a list/array of length n where p[i] = location of facility i
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Efficient: form B_p = b[np.ix_(p, p)] then cost = sum(a * B_p)
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _delta_cost_block_reverse(a, b, p, i, j):
    """
    Compute the change in QAP cost when we reverse the sub-sequence p[i..j].
    That is, facilities at positions i, i+1, ..., j in the permutation get their
    locations swapped/reversed.
    
    Actually, p[i] is the location assigned to facility i. Reversing a block of
    facilities means: for k in [i, j], facility k gets the location that facility (i+j-k) had.
    
    Let's think carefully. p is an array where p[fac] = loc.
    Reversing facilities i..j means: for t in 0..(j-i), the facility (i+t) gets location p[j-t].
    
    The cost is sum_{f1,f2} a[f1,f2] * b[p[f1], p[f2]].
    
    When we reverse block [i,j], only the locations of facilities in [i,j] change.
    Let S = set of facilities in [i,j].
    
    New cost - old cost = 
      sum over (f1 in S, f2 in S) of a[f1,f2]*(b[p_new[f1], p_new[f2]] - b[p_old[f1], p_old[f2]])
      + sum over (f1 in S, f2 not in S) of a[f1,f2]*(b[p_new[f1], p[f2]] - b[p_old[f1], p[f2]])
      + sum over (f1 not in S, f2 in S) of a[f1,f2]*(b[p[f1], p_new[f2]] - b[p[f1], p_old[f2]])
    
    This is O(|S| * n) per evaluation which for n=256 and |S| up to ~50 is ~12800 ops, fine.
    """
    length = j - i + 1
    if length <= 1:
        return 0
    
    # Build the mapping: old_loc[f] = p[f], new_loc[f] = p[i+j-f] for f in [i,j]
    # For f not in [i,j], new_loc[f] = old_loc[f]
    
    delta = 0.0
    
    # We can compute this more efficiently by noting:
    # For f1 in S, f2 in S:
    #   contribution change: a[f1,f2] * (b[new_loc[f1], new_loc[f2]] - b[old_loc[f1], old_loc[f2]])
    # For f1 in S, f2 not in S:
    #   contribution change: a[f1,f2] * (b[new_loc[f1], old_loc[f2]] - b[old_loc[f1], old_loc[f2]])
    # For f1 not in S, f2 in S:
    #   contribution change: a[f1,f2] * (b[old_loc[f1], new_loc[f2]] - b[old_loc[f1], old_loc[f2]])
    
    S = list(range(i, j + 1))
    
    # Precompute old and new locations for S
    old_locs = np.array([p[f] for f in S], dtype=np.int64)
    new_locs = np.array([p[i + j - f] for f in S], dtype=np.int64)
    
    # Indices into S array
    S_arr = np.array(S, dtype=np.int64)
    
    # Part 1: f1 in S, f2 in S
    # a[S, S] * (b[new_locs[:,None], new_locs[None,:]] - b[old_locs[:,None], old_locs[None,:]])
    A_SS = a[S_arr[:, None], S_arr[None, :]]
    B_old_SS = b[old_locs[:, None], old_locs[None, :]]
    B_new_SS = b[new_locs[:, None], new_locs[None, :]]
    delta += np.sum(A_SS * (B_new_SS - B_old_SS))
    
    # Part 2: f1 in S, f2 not in S
    # a[S, ~S] * (b[new_locs[:,None], p[~S][None,:]] - b[old_locs[:,None], p[~S][None,:]])
    not_S_mask = np.ones(len(p), dtype=bool)
    not_S_mask[S_arr] = False
    not_S = np.where(not_S_mask)[0]
    
    if len(not_S) > 0:
        A_S_notS = a[S_arr[:, None], not_S[None, :]]
        p_notS = p[not_S]
        B_old_S_notS = b[old_locs[:, None], p_notS[None, :]]
        B_new_S_notS = b[new_locs[:, None], p_notS[None, :]]
        delta += np.sum(A_S_notS * (B_new_S_notS - B_old_S_notS))
        
        # Part 3: f1 not in S, f2 in S
        A_notS_S = a[not_S[:, None], S_arr[None, :]]
        B_old_notS_S = b[p_notS[:, None], old_locs[None, :]]
        B_new_notS_S = b[p_notS[:, None], new_locs[None, :]]
        delta += np.sum(A_notS_S * (B_new_notS_S - B_old_notS_S))
    
    return delta


def solve(a, b):
    n = len(a)
    
    # Initial solution: rank-matching heuristic
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    # MCMC parameters
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Temperature schedule: start high, cool down
    # Estimate a reasonable initial temperature from typical delta costs
    # Try a few random moves to estimate typical |delta|
    temp = 1.0
    # Quick calibration
    for _ in range(10):
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, min(i + 20, n - 1))
        d = _delta_cost_block_reverse(a, b, p, i, j)
        if d != 0:
            temp = max(temp, abs(d) / 5.0)
    
    # Also try larger blocks
    for _ in range(10):
        length = random.randint(10, 50)
        i = random.randint(0, n - length - 1)
        j = i + length
        d = _delta_cost_block_reverse(a, b, p, i, j)
        if d != 0:
            temp = max(temp, abs(d) / 10.0)
    
    # Cap temperature
    temp = min(temp, 1e6)
    
    cooling_rate = 0.9999  # Slow cooling
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        # Choose block length: mix of small and large
        if rng.random() < 0.3:
            # Small block: 2-10
            length = int(rng.integers(2, 11))
        elif rng.random() < 0.7:
            # Medium block: 10-50
            length = int(rng.integers(10, 51))
        else:
            # Large block: 50-128
            length = int(rng.integers(50, min(129, n)))
        
        i = int(rng.integers(0, n - length + 1))
        j = i + length - 1
        
        delta = _delta_cost_block_reverse(a, b, p, i, j)
        
        # Metropolis criterion
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            # Accept: reverse the block
            p[i:j+1] = p[i:j+1][::-1]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        temp *= cooling_rate
        if temp < 1e-6:
            temp = 1e-6
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
