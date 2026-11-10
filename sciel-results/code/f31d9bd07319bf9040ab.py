"""
Strategy: Simulated Annealing with Smooth Surrogate Objective and Algebraic Initialization.

The current program uses greedy local search, which is prone to getting stuck in local minima
on the rugged max-discrepancy landscape. This approach posits that by using a smooth surrogate
objective (log-sum-exp of squared prefix sums, approximating the L_infinity norm), we can
navigate the landscape more effectively with Simulated Annealing. The SA process allows
accepting worse moves with a temperature-dependent probability, enabling escape from local
minima.

We retain the algebraic initialization (Legendre symbols) as it provides a strong starting point.
The SA loop uses vectorized NumPy operations to efficiently compute the surrogate objective.
After SA converges, we perform a final exact local search (polish) to minimize the true max
discrepancy, ensuring we don't settle for a suboptimal solution due to the surrogate.

Key improvements:
1. Smooth surrogate (log-sum-exp) for the SA phase to guide the search.
2. Simulated Annealing to escape local minima.
3. Final exact local search polish to refine the solution.
4. Algebraic initialization for a strong start.
"""

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _build_ap_structure(n):
    """
    Pre-compute all arithmetic progressions and map indices to the progressions they belong to.
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

def _init_prefix_sums(signs, ap_indices):
    """
    Compute initial prefix sums for all APs.
    """
    prefix_sums = []
    signs_arr = np.array(signs, dtype=np.int64)
    for indices in ap_indices:
        subseq = signs_arr[indices]
        ps = np.cumsum(subseq)
        prefix_sums.append(ps)
    return prefix_sums

def _legendre_symbol(n, p):
    """
    Compute the Legendre symbol (n/p) using Euler's criterion.
    Returns 0 if n % p == 0, 1 if n is a quadratic residue mod p, -1 otherwise.
    """
    if n % p == 0:
        return 0
    return pow(n, (p - 1) // 2, p)

def _generate_legendre_sequence(n):
    """
    Generate a sequence of signs based on the Legendre symbol for a prime near n.
    We find the smallest prime >= n.
    """
    def is_prime(k):
        if k < 2:
            return False
        for i in range(2, int(np.sqrt(k)) + 1):
            if k % i == 0:
                return False
        return True

    # Find prime p >= n
    p = n
    while not is_prime(p):
        p += 1

    seq = []
    for i in range(1, n + 1):
        ls = _legendre_symbol(i, p)
        if ls == 0:
            # Should not happen for i < p, but just in case
            seq.append(1)
        else:
            seq.append(ls)
    return seq

def _compute_surrogate(prefix_sums):
    """
    Compute the smooth surrogate objective: log-sum-exp of squared prefix sums.
    This approximates the L_infinity norm of the prefix sums.
    """
    # Concatenate all prefix sums
    all_ps = np.concatenate(prefix_sums)
    # Square them
    sq = all_ps ** 2
    # Log-sum-exp for stability
    max_sq = np.max(sq)
    if max_sq == 0:
        return 0.0
    log_sum_exp = max_sq + np.log(np.sum(np.exp(sq - max_sq)))
    return log_sum_exp

def _simulated_annealing(signs, ap_indices, index_to_aps, deadline, rng):
    """
    Perform Simulated Annealing using a smooth surrogate objective.
    Returns the best signs found.
    """
    n = len(signs)
    signs = list(signs)
    prefix_sums = _init_prefix_sums(signs, ap_indices)
    
    # Initial surrogate value
    current_surrogate = _compute_surrogate(prefix_sums)
    best_signs = list(signs)
    best_surrogate = current_surrogate
    
    # Temperature schedule
    T0 = 10.0
    T_end = 0.1
    cooling_rate = 0.995
    
    T = T0
    iter_count = 0
    max_iter = 50000
    
    while T > T_end and iter_count < max_iter and time.time() < deadline - 5:
        # Pick a random index to flip
        i = rng.randint(0, n - 1)
        old_val = signs[i]
        delta = -2 * old_val
        affected_aps = index_to_aps[i]
        
        # Compute new surrogate for affected APs
        # We only need to recompute the surrogate for the affected APs
        # but for simplicity and speed, we update the prefix sums and recompute
        # the full surrogate. This is O(N_ap * L) which might be slow.
        
        # Optimization: We can compute the change in the surrogate more efficiently.
        # But let's stick to the simple approach first and see if it's fast enough.
        
        # Tentatively apply flip
        signs[i] = -old_val
        for ap_id, pos in affected_aps:
            prefix_sums[ap_id][pos:] += delta
        
        new_surrogate = _compute_surrogate(prefix_sums)
        
        # Accept or reject
        if new_surrogate < current_surrogate:
            # Always accept improvement
            current_surrogate = new_surrogate
            if new_surrogate < best_surrogate:
                best_surrogate = new_surrogate
                best_signs = list(signs)
        else:
            # Accept with probability exp(-(new - current) / T)
            delta_e = new_surrogate - current_surrogate
            if rng.random() < np.exp(-delta_e / T):
                current_surrogate = new_surrogate
            else:
                # Revert
                signs[i] = old_val
                for ap_id, pos in affected_aps:
                    prefix_sums[ap_id][pos:] -= delta
        
        T *= cooling_rate
        iter_count += 1
        
    return best_signs

def _local_search_polish(signs, ap_indices, index_to_aps, deadline, rng):
    """
    Perform exact local search polish to minimize the true max discrepancy.
    """
    n = len(signs)
    signs = list(signs)
    prefix_sums = _init_prefix_sums(signs, ap_indices)
    ap_max_disc = np.array([np.max(np.abs(ps)) for ps in prefix_sums], dtype=np.int64)
    current_global_max = np.max(ap_max_disc)
    
    best_signs = list(signs)
    best_global_max = current_global_max
    
    max_ls_iters = 5000
    ls_iter = 0
    
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
                    
            # If the new max for affected APs is worse, skip
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
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Pre-compute structure
    ap_indices, index_to_aps = _build_ap_structure(n)
    
    rng = random.Random(42)
    
    best_signs_overall = None
    best_global_max_overall = float('inf')
    
    # Define starting configurations
    starts = []
    
    # 1. Legendre Symbol Sequence
    starts.append(_generate_legendre_sequence(n))
    
    # 2. Negated Legendre Symbol Sequence
    starts.append([-x for x in starts[0]])
    
    # 3. Shifted Legendre (shift by 1)
    leg_seq = _generate_legendre_sequence(n)
    starts.append([leg_seq[(i+1) % n] for i in range(n)])
    
    # 4. A few high-quality random starts
    for _ in range(2):
        starts.append([rng.choice((-1, 1)) for _ in range(n)])
    
    # Run SA + Polish from each start
    for start_signs in starts:
        if time.time() > deadline - 15:
            break
        
        # Phase 1: Simulated Annealing
        sa_signs = _simulated_annealing(start_signs, ap_indices, index_to_aps, deadline, rng)
        
        # Phase 2: Exact Local Search Polish
        polished_signs, polished_max = _local_search_polish(sa_signs, ap_indices, index_to_aps, deadline, rng)
        
        if polished_max < best_global_max_overall:
            best_global_max_overall = polished_max
            best_signs_overall = polished_signs
        
        # If we reach the theoretical lower bound, stop
        if best_global_max_overall == 1:
            break

    if best_signs_overall is None:
        best_signs_overall = [1] * n
        
    return best_signs_overall
# EVOLVE-BLOCK-END
