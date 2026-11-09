import numpy as np
import time

# Precomputation cache
_POLICY_CACHE = {}

def _precompute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid/ask depths for all steps and inventory levels.
    Returns:
        bid_depths: array of shape (steps, 2*Q+1)
        ask_depths: array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    q_min = -Q
    q_max = Q
    n_q = 2 * Q + 1
    
    # Grid of depths
    # Small depths for trading, large depths for avoiding fills
    d_grid = np.unique(np.concatenate([
        np.linspace(0, 1.0, 12), 
        [1.5, 2.0, 3.0, 5.0, 10.0]
    ]))
    # Calculate probabilities for this grid
    # p = 1 - exp(-A * exp(-k * d) * dt)
    p_grid = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)
    
    bid_depths = np.zeros((steps, n_q))
    ask_depths = np.zeros((steps, n_q))
    
    # V_next[q_idx]
    V_next = np.zeros(n_q)
    
    # Precompute q values
    q_vals = np.arange(q_min, q_max + 1)
    q_sq = q_vals ** 2
    
    for n in range(steps - 1, -1, -1):
        # Penalty term C_q
        C_q = (gamma / 2.0) * (sigma ** 2) * q_sq * dt
        
        # Compute deltas for all q
        # delta_minus[q] = V(q-1) - V(q)
        # delta_plus[q] = V(q+1) - V(q)
        # Indices: q_idx = q - q_min
        # q-1 is q_idx - 1
        # q+1 is q_idx + 1
        
        # Handle boundaries
        # For q = Q (last index), q+1 is out of bounds.
        # If q=Q, we cannot ask (X_a=0). So p_a must be 0.
        # If q=-Q, we cannot bid (X_b=0). So p_b must be 0.
        
        # We will handle this in the optimization loop.
        
        V_curr = np.zeros(n_q)
        
        # Loop over candidate ask depths (j)
        # We vectorize over q (axis 0) and bid depths (axis 1)
        
        # p_grid shape: (M,)
        # d_grid shape: (M,)
        
        # We need to find max over i (bid) and j (ask)
        # Score(i, j, q) = p_i (d_i + delta_minus[q]) + p_j (d_j + delta_plus[q]) - p_i p_j (delta_minus[q] + delta_plus[q])
        
        # Let's precompute terms that don't depend on i or j?
        # delta_minus shape: (n_q,)
        # delta_plus shape: (n_q,)
        
        # delta_minus[q_idx] = V_next[q_idx-1] - V_next[q_idx]
        delta_minus = np.empty(n_q)
        delta_plus = np.empty(n_q)
        
        # q_idx 0 (q=-Q): q-1 is invalid. But if q=-Q, we can't bid, so p_i=0.
        # We can set delta_minus[0] to 0, it won't matter if p_i=0.
        delta_minus[1:] = V_next[:-1] - V_next[1:]
        delta_minus[0] = 0.0 # Placeholder
        
        # q_idx last (q=Q): q+1 is invalid.
        delta_plus[:-1] = V_next[1:] - V_next[:-1]
        delta_plus[-1] = 0.0 # Placeholder
        
        K = delta_minus + delta_plus
        
        # We want to maximize over i, j.
        # Let's iterate over j (ask depth index)
        # For each j, we compute the best i for all q.
        
        best_total_score = np.full(n_q, -np.inf)
        best_i_idx = np.zeros(n_q, dtype=int)
        best_j_idx = np.zeros(n_q, dtype=int)
        
        # p_grid and d_grid are 1D arrays of length M
        # We need to compute:
        # TermI(i, q) = p_grid[i] * (d_grid[i] + delta_minus[q])
        # TermJ(j, q) = p_grid[j] * (d_grid[j] + delta_plus[q])
        # Coupling(i, j, q) = - p_grid[i] * p_grid[j] * K[q]
        
        # Score(i, j, q) = TermI(i, q) + TermJ(j, q) + Coupling(i, j, q)
        
        # For fixed j, Score is linear in p_grid[i]? No, p_grid[i] is a constant for the grid point.
        # Score(i, j, q) = p_grid[i] * (d_grid[i] + delta_minus[q] - p_grid[j] * K[q]) + p_grid[j] * (d_grid[j] + delta_plus[q])
        
        # So for fixed j and q, we maximize over i:
        # p_grid[i] * (d_grid[i] + delta_minus[q] - p_grid[j] * K[q])
        
        # Let A_i_q = d_grid[i] + delta_minus[q] - p_grid[j] * K[q]
        # We want max_i p_grid[i] * A_i_q
        
        # We can compute this for all i, q simultaneously using broadcasting.
        
        for j in range(len(d_grid)):
            p_j = p_grid[j]
            d_j = d_grid[j]
            
            # TermJ for this j, for all q
            # Note: if q = -Q, we cannot ask. So we should force p_j=0 or set score to -inf for this j?
            # Actually, if q=-Q, delta_plus is 0, but we must not ask.
            # The constraint is: if q=-Q, X_a=0.
            # This means we must choose a depth d_a such that p_a=0.
            # In our grid, the largest depth gives smallest p.
            # If we set d_a to max grid value, p_a is small but not 0.
            # To be safe, we should explicitly set p_j=0 if q=-Q?
            # Or better: after finding optimal, override.
            # Let's just let the DP run, and then override the actions at boundaries.
            # However, the DP value might be wrong if it thinks it can fill.
            # If q=-Q, and it chooses an ask fill, it goes to -Q+1.
            # But the problem says "no ask fill at q = -Q".
            # So the transition probability for X_a=1 must be 0.
            # So we must enforce p_a=0 for q=-Q.
            
            # Similarly, p_b=0 for q=Q.
            
            # Let's create masks
            # mask_no_ask: q = -Q (index 0)
            # mask_no_bid: q = Q (index n_q-1)
            
            # For q=0, we must have p_j=0.
            # For q=n_q-1, we must have p_i=0.
            
            # We can handle this by modifying the effective p_grid or scores.
            
            # Compute A_i_q for all i, q
            # d_grid[:, None] + delta_minus[None, :] - p_j * K[None, :]
            A_i_q = d_grid[:, None] + delta_minus[None, :] - p_j * K[None, :]
            
            # Score_i_q = p_grid[:, None] * A_i_q
            Score_i_q = p_grid[:, None] * A_i_q
            
            # Enforce no bid at q=Q (last index)
            # If q=Q, we must choose i such that p_grid[i]=0.
            # Our grid has p > 0 for all i.
            # So we set Score_i_q[:, -1] = -np.inf, except for the largest depth?
            # Actually, if we set it to -inf, max will fail or pick 0.
            # We should pick the depth that gives p=0.
            # Since we don't have p=0 in grid, we pick the largest depth (smallest p).
            # Let's set all other scores to -inf for q=Q.
            # Find index of max depth
            max_depth_idx = np.argmax(d_grid)
            
            # For q = Q (last column), only allow max_depth_idx
            Score_i_q[:, -1] = -np.inf
            Score_i_q[max_depth_idx, -1] = 0.0 # p is small, reward is small.
            
            # Similarly, for q = -Q (first column), we are in loop j.
            # If q = -Q, we must have p_j=0.
            # So for j loop, if we are at q=0, this j is invalid unless p_j=0.
            # We can set TermJ to -inf for q=0?
            # Or simply, after computing best i for each q, we check.
            
            # Find best i for each q
            best_i_score = np.max(Score_i_q, axis=0)
            best_i_idx_curr = np.argmax(Score_i_q, axis=0)
            
            # TermJ for this j
            TermJ = p_j * (d_j + delta_plus)
            
            # Total Score for this j
            TotalScore = best_i_score + TermJ
            
            # Enforce no ask at q=-Q (first index)
            # If q=0, we must have p_j=0.
            # So TotalScore[0] should be -inf for this j, unless p_j is effectively 0.
            # Since p_j > 0, we set TotalScore[0] = -np.inf
            TotalScore[0] = -np.inf
            
            # Update best
            mask = TotalScore > best_total_score
            best_total_score[mask] = TotalScore[mask]
            best_i_idx[mask] = best_i_idx_curr[mask]
            best_j_idx[mask] = j
            
        # After loop over j, we have best_total_score, best_i_idx, best_j_idx
        # V_curr[q] = V_next[q] - C_q[q] + best_total_score[q]
        
        # Handle cases where no valid action found (should not happen)
        valid = np.isfinite(best_total_score)
        V_curr[valid] = V_next[valid] - C_q[valid] + best_total_score[valid]
        # If invalid, set to 0 or previous?
        # Should not happen.
        
        # Store optimal depths
        # best_i_idx is index in d_grid
        # best_j_idx is index in d_grid
        bid_depths[n, :] = d_grid[best_i_idx]
        ask_depths[n, :] = d_grid[best_j_idx]
        
        # Override boundaries explicitly to ensure no fills
        # q = Q (last index): bid depth should be large
        bid_depths[n, -1] = d_grid[-1]
        # q = -Q (first index): ask depth should be large
        ask_depths[n, 0] = d_grid[-1]
        
        V_next = V_curr
        
    return bid_depths, ask_depths

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma)
    
    if key not in _POLICY_CACHE:
        # Check time
        start_time = time.time()
        bid_depths, ask_depths = _precompute_policy(sigma, A, k, gamma, T, steps, Q)
        _POLICY_CACHE[key] = (bid_depths, ask_depths)
        
        # If precomputation took too long, we are in trouble, but it should be fast.
        elapsed = time.time() - start_time
        if elapsed > 10:
            # Fallback? No, just proceed.
            pass
            
    bid_depths, ask_depths = _POLICY_CACHE[key]
    
    # Determine step n
    dt = T / steps
    n = int(round(t / dt))
    if n >= steps: n = steps - 1
    if n < 0: n = 0
    
    # Get depths for this step
    # inventory is array of ints
    # Map to indices: idx = inventory + Q
    idx = inventory + Q
    
    # Clip indices just in case
    idx = np.clip(idx, 0, 2*Q)
    
    b_depths = bid_depths[n, idx]
    a_depths = ask_depths[n, idx]
    
    return b_depths, a_depths
