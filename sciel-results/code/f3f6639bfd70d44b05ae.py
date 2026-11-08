"""Optimal market making with inventory risk – precomputed HJB policy tables."""
import numpy as np
import time

# Module-level cache for policy tables keyed by regime params
_policy_cache = {}
_DEADLINE = 300.0  # seconds


def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths for all (time_step, inventory) states
    via backward induction on the discrete HJB equation.
    
    Returns:
        bid_table: array of shape (steps+1, 2*Q+1) with optimal bid depths
        ask_table: array of shape (steps+1, 2*Q+1) with optimal ask depths
    """
    n_steps = steps
    dt = T / n_steps
    n_inv = 2 * Q + 1
    inv_range = np.arange(-Q, Q + 1)  # -Q .. Q

    # Candidate depths: we'll search over a grid of depth values
    # Depth range: from very small (aggressive) to large (conservative)
    # The probability of fill is 1 - exp(-A * exp(-k*d) * dt)
    # For d very small, exp(-k*d) ~ 1, so prob ~ 1 - exp(-A*dt)
    # For d very large, exp(-k*d) ~ 0, so prob ~ 0
    
    # Let's use a logarithmic grid for depth candidates
    # Minimum depth: 1e-4 (very aggressive)
    # Maximum depth: 20/k (very conservative, fill prob ~ exp(-A*exp(-20)*dt) ~ 0)
    d_min = 1e-4
    d_max = max(20.0 / k, 1.0)
    n_depths = 200  # number of candidate depth values
    depth_grid = np.linspace(d_min, d_max, n_depths)
    
    # Precompute fill probabilities for each depth
    # fill_prob[d] = 1 - exp(-A * exp(-k*d) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Value function: V[n][q] = optimal expected reward from step n with inventory q
    # q index: 0 corresponds to q=-Q, Q corresponds to q=0, 2Q corresponds to q=+Q
    # V[n][q_idx]
    
    # Terminal condition: no terminal charge, V[steps][q] = 0 for all q
    V = np.zeros((n_steps + 1, n_inv), dtype=np.float64)
    
    # Optimal actions
    bid_table = np.full((n_steps + 1, n_inv), np.inf, dtype=np.float64)
    ask_table = np.full((n_steps + 1, n_inv), np.inf, dtype=np.float64)
    
    # Inventory cost per step: (gamma/2) * sigma^2 * q^2 * dt
    inv_cost = (gamma / 2.0) * sigma**2 * inv_range**2 * dt
    
    # Backward induction
    for n in range(n_steps - 1, -1, -1):
        # V[n][q] = max over (bid_d, ask_d) of:
        #   - inv_cost[q] 
        #   + p_bid * (bid_d + V[n+1][q+1]) 
        #   + p_ask * (ask_d + V[n+1][q-1])
        #   + (1-p_bid)(1-p_ask) * V[n+1][q]
        #   + p_bid*p_ask * V[n+1][q]  (both fill, net inventory change 0)
        # Wait, let me reconsider.
        # If bid fills: earn bid_d, inventory increases by 1 (q -> q+1)
        # If ask fills: earn ask_d, inventory decreases by 1 (q -> q-1)
        # Both fill: earn bid_d + ask_d, inventory unchanged (q -> q)
        # Neither fill: earn 0, inventory unchanged (q -> q)
        
        # So:
        # V[n][q] = -inv_cost[q] + max_{d_b, d_a} [
        #   p_b * (d_b + V[n+1][q+1]) 
        #   + p_a * (d_a + V[n+1][q-1])
        #   + (1-p_b)(1-p_a) * V[n+1][q]
        #   + p_b*p_a * V[n+1][q]
        # ]
        # = -inv_cost[q] + max_{d_b, d_a} [
        #   p_b * (d_b + V[n+1][q+1]) 
        #   + p_a * (d_a + V[n+1][q-1])
        #   + (1 - p_b - p_a + p_b*p_a) * V[n+1][q]
        #   + p_b*p_a * V[n+1][q]
        # ]
        # = -inv_cost[q] + max_{d_b, d_a} [
        #   p_b * (d_b + V[n+1][q+1]) 
        #   + p_a * (d_a + V[n+1][q-1])
        #   + (1 - p_b - p_a) * V[n+1][q]
        # ]
        # Because (1-p_b)(1-p_a) + p_b*p_a = 1 - p_b - p_a + 2*p_b*p_a... 
        # Let me redo: 
        # Neither: (1-p_b)(1-p_a) * V[q]
        # Both: p_b*p_a * V[q]
        # Sum: [(1-p_b)(1-p_a) + p_b*p_a] * V[q] = [1 - p_b - p_a + 2 p_b p_a] * V[q]
        # Hmm, that's not right either. Let me be more careful.
        
        # Actually, the four cases are mutually exclusive:
        # 1. Bid only fills: prob p_b*(1-p_a), reward d_b + V[n+1][q+1]
        # 2. Ask only fills: prob (1-p_b)*p_a, reward d_a + V[n+1][q-1]
        # 3. Both fill: prob p_b*p_a, reward d_b + d_a + V[n+1][q]
        # 4. Neither: prob (1-p_b)*(1-p_a), reward V[n+1][q]
        
        # Expected reward = p_b*(1-p_a)*(d_b + V[q+1]) + (1-p_b)*p_a*(d_a + V[q-1]) 
        #                  + p_b*p_a*(d_b + d_a + V[q]) + (1-p_b)*(1-p_a)*V[q]
        
        # = p_b*(1-p_a)*d_b + (1-p_b)*p_a*d_a + p_b*p_a*(d_b+d_a) 
        #   + [p_b*(1-p_a)*V[q+1] + (1-p_b)*p_a*V[q-1] + (p_b*p_a + (1-p_b)*(1-p_a))*V[q]]
        
        # Note: p_b*p_a + (1-p_b)*(1-p_a) = 1 - p_b - p_a + 2*p_b*p_a
        
        # For fixed q, we can separate the optimization:
        # The d_b terms: d_b * [p_b*(1-p_a) + p_b*p_a] = d_b * p_b
        # The d_a terms: d_a * [(1-p_b)*p_a + p_b*p_a] = d_a * p_a
        
        # So the expression becomes:
        # p_b * d_b + p_a * d_a + [p_b*(1-p_a)*V[q+1] + (1-p_b)*p_a*V[q-1] + (1-p_b-p_a+2*p_b*p_a)*V[q]]
        
        # Since p_b depends only on d_b and p_a depends only on d_a, and the cross term 
        # (1-p_b-p_a+2*p_b*p_a) couples them, we can't fully separate.
        
        # However, we can do a 2D grid search over (d_b, d_a).
        
        # For efficiency, let's precompute for each depth candidate:
        # p_b[d] and d_b * p_b[d]
        
        V_next = V[n + 1]
        
        for q_idx in range(n_inv):
            q = inv_range[q_idx]
            
            # Handle boundary conditions
            q_plus_idx = q_idx + 1 if q_idx < n_inv - 1 else None
            q_minus_idx = q_idx - 1 if q_idx > 0 else None
            
            V_q_plus = V_next[q_plus_idx] if q_plus_idx is not None else -np.inf
            V_q_minus = V_next[q_minus_idx] if q_minus_idx is not None else -np.inf
            V_q = V_next[q_idx]
            
            best_val = -np.inf
            best_db = np.inf
            best_da = np.inf
            
            # If at boundary Q (max inventory), no bid fill possible
            # If at boundary -Q (min inventory), no ask fill possible
            
            if q == Q:
                # No bid fill, only ask
                # Expected reward = p_a * d_a + (1-p_a) * V[q-1] + p_a * V[q-1]... 
                # Wait, if no bid fill, then:
                # Ask fills: prob p_a, reward d_a + V[q-1]
                # No fill: prob 1-p_a, reward V[q]
                # Expected = p_a*(d_a + V[q-1]) + (1-p_a)*V[q]
                # = p_a*d_a + p_a*V[q-1] + (1-p_a)*V[q]
                # Optimize over d_a: p_a*d_a + p_a*V[q-1] + (1-p_a)*V[q]
                # = p_a*(d_a + V[q-1]) + (1-p_a)*V[q]
                for i in range(n_depths):
                    pa = fill_probs[i]
                    da = depth_grid[i]
                    val = pa * (da + V_q_minus) + (1.0 - pa) * V_q
                    if val > best_val:
                        best_val = val
                        best_db = np.inf  # no bid quote
                        best_da = da
            elif q == -Q:
                # No ask fill, only bid
                # Expected = p_b*(d_b + V[q+1]) + (1-p_b)*V[q]
                for i in range(n_depths):
                    pb = fill_probs[i]
                    db = depth_grid[i]
                    val = pb * (db + V_q_plus) + (1.0 - pb) * V_q
                    if val > best_val:
                        best_val = val
                        best_db = db
                        best_da = np.inf  # no ask quote
            else:
                # Both sides possible
                # We need to optimize over (d_b, d_a)
                # To keep it tractable, we'll do a nested loop but with early termination
                for i_b in range(n_depths):
                    pb = fill_probs[i_b]
                    db = depth_grid[i_b]
                    # For each d_b, optimize over d_a
                    # Expected = pb*(1-pa)*(db + V[q+1]) + (1-pb)*pa*(da + V[q-1]) 
                    #           + pb*pa*(db + da + V[q]) + (1-pb)*(1-pa)*V[q]
                    # = pb*db + pa*da + pb*(1-pa)*V[q+1] + (1-pb)*pa*V[q-1] 
                    #   + (pb*pa + (1-pb)*(1-pa))*V[q]
                    
                    # Let's compute the terms that don't depend on da:
                    base_b = pb * db + pb * (1.0) * V_q_plus  # will adjust for (1-pa)
                    # Actually let's just do the full computation
                    for i_a in range(n_depths):
                        pa = fill_probs[i_a]
                        da = depth_grid[i_a]
                        # Compute expected value
                        # Term 1: bid only
                        t1 = pb * (1.0 - pa) * (db + V_q_plus)
                        # Term 2: ask only
                        t2 = (1.0 - pb) * pa * (da + V_q_minus)
                        # Term 3: both
                        t3 = pb * pa * (db + da + V_q)
                        # Term 4: neither
                        t4 = (1.0 - pb) * (1.0 - pa) * V_q
                        
                        val = t1 + t2 + t3 + t4
                        if val > best_val:
                            best_val = val
                            best_db = db
                            best_da = da
            
            V[n, q_idx] = best_val - inv_cost[q_idx]
            bid_table[n, q_idx] = best_db
            ask_table[n, q_idx] = best_da
    
    return bid_table, ask_table


def _get_policy_table(params):
    """Get or compute the policy table for the given parameters."""
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_policy_table(
            params["sigma"], params["A"], params["k"], 
            params["gamma"], params["T"], params["steps"], params["Q"]
        )
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Quote optimal bid/ask depths based on precomputed HJB policy table.
    
    Returns:
        (bid_depths, ask_depths): tuple of two float arrays of shape (4000,)
    """
    start_time = time.time()
    
    # Determine time step index
    dt = params["T"] / params["steps"]
    n_step = int(round(t / dt))
    n_step = max(0, min(n_step, params["steps"] - 1))
    
    # Get policy table
    bid_table, ask_table = _get_policy_table(params)
    
    # Map inventory to index: q in [-Q, Q] -> idx in [0, 2Q]
    Q = params["Q"]
    inv_idx = inventory + Q  # shift to [0, 2Q]
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Look up optimal depths
    bid_depths = bid_table[n_step, inv_idx].copy()
    ask_depths = ask_table[n_step, inv_idx].copy()
    
    # Ensure validity: depths must be >= -3/k (but they should be positive)
    # Replace any invalid values
    min_depth = -3.0 / params["k"]
    bid_depths[bid_depths < min_depth] = np.inf
    ask_depths[ask_depths < min_depth] = np.inf
    bid_depths[~np.isfinite(bid_depths) & (bid_depths != np.inf)] = np.inf
    ask_depths[~np.isfinite(ask_depths) & (ask_depths != np.inf)] = np.inf
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 5.0:
        # Return a safe fallback
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    return bid_depths, ask_depths
