"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix using a Dynamic Programming approach
    over discretized budget levels. This models the "optimal" prize collectable
    from each node given a remaining budget, providing a global structural view
    of the problem rather than a local greedy lookahead.
    """
    start_time = time.time()
    deadline = start_time + 290.0

    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)

    # Discretize the budget into M bins
    M = 20
    # Time bins: t_m = (m / (M - 1)) * max_len
    t_bins = np.linspace(0, max_len, M)
    
    # DP[j, m] = max prize collectable starting at j with total remaining tour length budget t_m
    # We iterate m from 0 to M-1
    # DP[j, m] depends on DP[k, m'] where t_m' < t_m
    DP = np.zeros((n, M), dtype=np.float64)
    
    # Precompute distance to depot for quick checks
    d0 = distance_matrix[:, 0]
    
    # Fill DP table
    for m in range(M):
        t = t_bins[m]
        for j in range(n):
            # Option 1: Return immediately (prize 0)
            best_val = 0.0
            # Option 2: Visit some node k next
            # Must satisfy: d[j, k] + d[k, 0] <= t  (so we can return from k)
            # And we need to find the appropriate m' for k such that t_m' <= t - d[j, k]
            
            # Vectorized check for feasibility of moving j -> k
            # cost_jk = distance_matrix[j, :]
            # feasible_k = (distance_matrix[j, :] + d0) <= t
            # If feasible, new_budget_for_k = t - distance_matrix[j, k]
            # We need DP[k, m_k] where t_m_k is the largest bin <= new_budget_for_k
            
            # To keep it fast, we can just loop k or use vectorized ops
            # Given N=50, loop is fine.
            for k in range(n):
                if k == j:
                    continue
                d_jk = distance_matrix[j, k]
                if d_jk + d0[k] > t + 1e-9:
                    continue
                
                # Remaining budget for k
                rem_t = t - d_jk
                # Find index m_k such that t_bins[m_k] <= rem_t
                # Since t_bins is sorted, we can use searchsorted
                # We want the largest m_k such that t_bins[m_k] <= rem_t
                # If rem_t < t_bins[0], no valid state? But t_bins[0]=0.
                # DP[k, 0] corresponds to budget 0.
                # If rem_t >= 0, we can use some state.
                # Actually, if rem_t is small, we use low m.
                m_k = np.searchsorted(t_bins, rem_t, side='right') - 1
                if m_k < 0:
                    m_k = 0
                
                val = prizes[k] + DP[k, m_k]
                if val > best_val:
                    best_val = val
            
            DP[j, m] = best_val
    
    # Compute heuristic matrix
    # H[i, j] = DP[j, m_rem] where m_rem corresponds to budget max_len - d[i, j]
    # Note: The budget for the sub-tour starting at j is max_len - d[i, j] (assuming we start at i with 0 prior cost? 
    # Actually, the ant has traveled some distance. But H is static. 
    # Using max_len - d[i, j] is an optimistic estimate of available budget for j's sub-tour.
    
    heuristic = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_ij = distance_matrix[i, j]
            # Budget remaining for sub-tour starting at j
            # This is the total max_len minus the cost to get to j.
            # Since we don't know prior cost, we assume the "cost to get to j" is just d[i, j] 
            # relative to the edge itself? 
            # A more standard static heuristic for OPTW is prize[j] / (d[i, j] + d[j, 0]).
            # Our DP value DP[j, t] already incorporates prize and future potential.
            # We map the available budget to a bin index.
            avail_budget = max_len - d_ij
            if avail_budget < 0:
                heuristic[i, j] = 1e-9
                continue
            
            m_idx = np.searchsorted(t_bins, avail_budget, side='right') - 1
            if m_idx < 0:
                m_idx = 0
            
            heuristic[i, j] = DP[j, m_idx]
            
    # Normalize and clean up
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple heuristic if DP took too long (unlikely for N=50, M=20)
        # But just in case, return a simple valid matrix
        simple_h = np.ones((n, n)) * 1e-9
        return simple_h

    return heuristic
# EVOLVE-BLOCK-END
