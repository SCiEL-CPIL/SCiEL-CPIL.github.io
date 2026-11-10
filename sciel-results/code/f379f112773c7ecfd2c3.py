"""
Strategy: Linear Programming (LP) Relaxation for Global Discrepancy Balancing.

The previous strategies relied on heuristic local search (Hill Climbing, Simulated Annealing) 
starting from algebraic or random initializations. These methods are prone to getting stuck 
in local minima because the objective function (max absolute sum over APs) is non-smooth and 
highly rugged.

This new approach reformulates the problem as a Linear Programming (LP) problem:
Minimize D
Subject to:
  -sum_ap <= D  for all APs
   sum_ap <= D  for all APs
  -1 <= x_i <= 1 for all i

Here, x_i represents the sign at position i (relaxed to continuous [-1, 1]).
The variable D represents the maximum absolute discrepancy.

Why this might work:
1. The LP solver (HiGHS via scipy) finds a globally optimal continuous solution.
2. This solution inherently balances the "pressure" from all APs, avoiding the 
   "greedy" bias of local search.
3. We then round the continuous solution to integers {-1, 1}. 
4. We perform a short local search refinement on this rounded solution to ensure 
   integrality and potentially reduce the discrepancy further if the rounding 
   introduced slight imbalances.

The LP has 256 variables (x_i) + 1 variable (D) = 257 variables.
The number of constraints is 2 * (number of APs).
Number of APs for n=256:
For step k, there are roughly n/k APs of length > 1? No, we count all APs.
Actually, the number of APs is sum_{k=1}^{n} (n-k+1) which is O(n^2).
For n=256, sum_{k=1}^{256} (256-k+1) = 256*257/2 = 32896 APs.
Constraints: 2 * 32896 = 65792 constraints.
Variables: 257.
This is a sparse LP. HiGHS should solve this quickly (likely < 1s).

We construct the constraint matrix A_ub and b_ub.
Each constraint is of the form:
  sum_{i in AP} x_i <= D  =>  sum_{i in AP} x_i - D <= 0
  -sum_{i in AP} x_i <= D => -sum_{i in AP} x_i - D <= 0

We also have bounds -1 <= x_i <= 1.

After solving, we round x_i to sign(x_i). If x_i is very close to 0, we break ties 
randomly or based on the residual. Then we run a fast local search to polish.
"""

import time
import numpy as np
from scipy.optimize import linprog

def _build_ap_structure(n):
    """
    Pre-compute all arithmetic progressions and map indices to the progressions they belong to.
    Returns:
    ap_indices: list of numpy arrays of indices for each AP
    index_to_aps: list of lists of (ap_id, pos) for each index
    """
    ap_indices = []
    index_to_aps = [[] for _ in range(n)]
    
    ap_id = 0
    for step in range(1, n + 1):
        for start in range(n):
            indices = np.arange(start, n, step, dtype=np.int32)
            if len(indices) == 0:
                continue
            
            ap_indices.append(indices)
            
            for pos, idx in enumerate(indices):
                index_to_aps[idx].append((ap_id, pos))
            
            ap_id += 1
            
    return ap_indices, index_to_aps

def _solve_lp(n, ap_indices, deadline):
    """
    Solve the LP relaxation.
    Returns the continuous solution x* and the optimal D*.
    """
    num_vars = n + 1  # x_0 ... x_{n-1}, D
    x_bounds = [(-1, 1)] * n + [(0, None)]  # D >= 0
    
    # We will build the constraint matrix A_ub and vector b_ub
    # Constraints:
    # 1. For each AP: sum(x_i) - D <= 0
    # 2. For each AP: -sum(x_i) - D <= 0
    
    num_aps = len(ap_indices)
    num_constraints = 2 * num_aps
    
    # Using a dense matrix might be slow/memory heavy if n is large, but n=256 is small.
    # However, A_ub is 65k x 257. That's ~16 million floats. 64MB. Acceptable.
    # But building it in Python loops is slow. We can use sparse matrices.
    from scipy.sparse import lil_matrix, csr_matrix
    
    A = lil_matrix((num_constraints, num_vars), dtype=np.float64)
    b = np.zeros(num_constraints, dtype=np.float64)
    
    for k, indices in enumerate(ap_indices):
        # Constraint 1: sum(x_i) - D <= 0
        # Indices in x: indices
        # Index in D: n
        for idx in indices:
            A[2*k, idx] = 1.0
        A[2*k, n] = -1.0
        b[2*k] = 0.0
        
        # Constraint 2: -sum(x_i) - D <= 0
        for idx in indices:
            A[2*k+1, idx] = -1.0
        A[2*k+1, n] = -1.0
        b[2*k+1] = 0.0
        
        if time.time() > deadline - 5:
            # Fallback to random if LP takes too long to build/solve
            return None, None

    A = A.tocsr()
    
    # Objective: Minimize D => c = [0, ..., 0, 1]
    c = np.zeros(num_vars)
    c[n] = 1.0
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')
        if res.status == 0:
            return res.x[:n], res.x[n]
        else:
            return None, None
    except Exception:
        return None, None

def _local_search(signs, ap_indices, index_to_aps, deadline):
    """
    Fast local search to polish the integer solution.
    """
    n = len(signs)
    signs = list(signs)
    
    # Compute initial prefix sums and max disc
    prefix_sums = []
    signs_arr = np.array(signs, dtype=np.int64)
    for indices in ap_indices:
        subseq = signs_arr[indices]
        ps = np.cumsum(subseq)
        prefix_sums.append(ps)
        
    ap_max_disc = np.array([np.max(np.abs(ps)) for ps in prefix_sums], dtype=np.int64)
    current_global_max = np.max(ap_max_disc)
    
    best_signs = list(signs)
    best_global_max = current_global_max
    
    max_ls_iters = 20
    ls_iter = 0
    
    # Use a fixed seed for reproducibility in polishing
    rng = np.random.default_rng(42)
    
    while ls_iter < max_ls_iters and time.time() < deadline - 2:
        improved = False
        indices_to_try = list(range(n))
        rng.shuffle(indices_to_try)
        
        for i in indices_to_try:
            if time.time() > deadline - 2:
                break
                
            old_val = signs[i]
            delta = -2 * old_val
            affected_aps = index_to_aps[i]
            
            # Evaluate the flip
            max_new_aff = 0
            for ap_id, pos in affected_aps:
                ps = prefix_sums[ap_id]
                if pos > 0:
                    max_pref = np.max(np.abs(ps[:pos]))
                else:
                    max_pref = 0
                max_suff = np.max(np.abs(ps[pos:] + delta))
                new_ap_max = max(max_pref, max_suff)
                if new_ap_max > max_new_aff:
                    max_new_aff = new_ap_max
                    
            if max_new_aff > current_global_max:
                continue
                
            # Apply flip tentatively
            signs[i] = -old_val
            for ap_id, pos in affected_aps:
                prefix_sums[ap_id][pos:] += delta
                new_m = np.max(np.abs(prefix_sums[ap_id]))
                ap_max_disc[ap_id] = new_m
                
            new_global_max = np.max(ap_max_disc)
            
            if new_global_max < current_global_max:
                current_global_max = new_global_max
                improved = True
                if current_global_max < best_global_max:
                    best_global_max = current_global_max
                    best_signs = list(signs)
            else:
                # Revert
                signs[i] = old_val
                for ap_id, pos in affected_aps:
                    prefix_sums[ap_id][pos:] -= delta
                    ap_max_disc[ap_id] = np.max(np.abs(prefix_sums[ap_id]))
                    
        ls_iter += 1
        if not improved:
            break
            
    return best_signs, best_global_max

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Pre-compute structure
    ap_indices, index_to_aps = _build_ap_structure(n)
    
    # 1. Try LP Relaxation
    x_lp, D_lp = _solve_lp(n, ap_indices, deadline)
    
    best_signs = None
    best_score = float('inf')
    
    if x_lp is not None:
        # Round the continuous solution to integers
        # If x_i > 0, sign is 1; if x_i < 0, sign is -1.
        # If x_i is close to 0, we need a tie-breaking strategy.
        # We can use the sign of the residual or random.
        # Let's start with sign(x_i).
        signs_int = [1 if xi > 0 else -1 for xi in x_lp]
        
        # Polish with local search
        polished_signs, polished_score = _local_search(signs_int, ap_indices, index_to_aps, deadline)
        
        if polished_score < best_score:
            best_score = polished_score
            best_signs = polished_signs
            
        # Also try a variant where we flip signs where |x_i| is smallest
        # This might escape local minima caused by rounding
        if time.time() < deadline - 10:
            # Get indices sorted by |x_i|
            abs_x = np.abs(x_lp)
            sorted_indices = np.argsort(abs_x)
            
            # Flip the first few smallest |x_i|
            # Try flipping 0, 1, 2, 4 smallest
            for num_flips in [0, 1, 2, 4]:
                signs_variant = list(signs_int)
                for idx in sorted_indices[:num_flips]:
                    signs_variant[idx] = -signs_variant[idx]
                
                polished_signs_v, polished_score_v = _local_search(signs_variant, ap_indices, index_to_aps, deadline)
                if polished_score_v < best_score:
                    best_score = polished_score_v
                    best_signs = polished_signs_v
                
                if time.time() > deadline - 5:
                    break
            
    # 2. Fallback / Multi-start with Heuristics if LP failed or time remains
    if time.time() < deadline - 15:
        # Use Legendre symbol as a strong heuristic start
        def _generate_legendre_sequence(n):
            def is_prime(k):
                if k < 2: return False
                for i in range(2, int(np.sqrt(k)) + 1):
                    if k % i == 0: return False
                return True
            p = n
            while not is_prime(p): p += 1
            seq = []
            for i in range(1, n + 1):
                ls = pow(i, (p - 1) // 2, p)
                if ls == 0: seq.append(1)
                else: seq.append(1 if ls == 1 else -1)
            return seq
        
        leg_seq = _generate_legendre_sequence(n)
        polished_signs_leg, polished_score_leg = _local_search(leg_seq, ap_indices, index_to_aps, deadline)
        if polished_score_leg < best_score:
            best_score = polished_score_leg
            best_signs = polished_signs_leg

    # 3. Random fallback if nothing else worked
    if best_signs is None:
        # Random walk or just random
        rng = np.random.default_rng(0)
        best_signs = list(rng.choice([-1, 1], size=n))
        # Do a quick local search
        polished_signs_r, polished_score_r = _local_search(best_signs, ap_indices, index_to_aps, deadline)
        best_signs = polished_signs_r
        best_score = polished_score_r

    return best_signs

# EVOLVE-BLOCK-END
