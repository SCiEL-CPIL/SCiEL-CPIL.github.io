import numpy as np
import time

_start_time = None
_dp_cache = {}

def _get_dp_tables(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid and ask depths for all t and q using DP.
    Returns:
        bid_depths: array of shape (steps, 2*Q+1)
        ask_depths: array of shape (steps, 2*Q+1)
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _dp_cache:
        return _dp_cache[key]
    
    dt = T / steps
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)
    
    # V_next[q_index]
    V_next = np.zeros(n_states, dtype=np.float64)
    
    bid_depths = np.zeros((steps, n_states), dtype=np.float64)
    ask_depths = np.zeros((steps, n_states), dtype=np.float64)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_states, dtype=np.float64)
        
        for i, q in enumerate(q_values):
            # Inventory cost
            cost = 0.5 * gamma * sigma * sigma * q * q * dt
            
            # Calculate optimal depths
            # d_b = max(0, 1/k - (V_next[q+1] - V_next[q]))
            # d_a = max(0, 1/k - (V_next[q-1] - V_next[q]))
            
            # Handle boundaries
            if q == Q:
                d_b = np.inf
                # Ask is still possible
                # Delta_minus = V_next[q-1] - V_next[q]
                delta_minus = V_next[i-1] - V_next[i]
                d_a = max(0.0, (1.0 / k) - delta_minus)
            elif q == -Q:
                d_a = np.inf
                # Bid is still possible
                # Delta_plus = V_next[q+1] - V_next[q]
                delta_plus = V_next[i+1] - V_next[i]
                d_b = max(0.0, (1.0 / k) - delta_plus)
            else:
                delta_plus = V_next[i+1] - V_next[i]
                delta_minus = V_next[i-1] - V_next[i]
                d_b = max(0.0, (1.0 / k) - delta_plus)
                d_a = max(0.0, (1.0 / k) - delta_minus)
            
            bid_depths[n, i] = d_b
            ask_depths[n, i] = d_a
            
            # Calculate fill probabilities
            if np.isinf(d_b):
                p_b = 0.0
            else:
                # p_b = 1 - exp(-A * exp(-k * d_b) * dt)
                # For numerical stability, if d_b is large, exp(-k*d_b) is small
                lam_b = A * np.exp(-k * d_b) if d_b < 1000/k else 0.0
                p_b = 1.0 - np.exp(-lam_b * dt)
            
            if np.isinf(d_a):
                p_a = 0.0
            else:
                lam_a = A * np.exp(-k * d_a) if d_a < 1000/k else 0.0
                p_a = 1.0 - np.exp(-lam_a * dt)
            
            # Expected future value
            # Transitions:
            # q -> q+1 with prob p_b * (1-p_a)  (Bid fill, No Ask fill)
            # q -> q-1 with prob (1-p_b) * p_a  (No Bid fill, Ask fill)
            # q -> q   with prob (1-p_b)*(1-p_a) + p_b*p_a (No fills or both fill)
            # Note: "independently".
            
            # If q=Q, p_b=0, so q->Q+1 is 0.
            # If q=-Q, p_a=0, so q->Q-1 is 0.
            
            V_q = V_next[i]
            
            if q == Q:
                # Only q->Q (no bid fill) and q->Q-1 (ask fill)
                # Wait, if q=Q, bid doesn't fill.
                # Ask can fill: q->Q-1
                # Prob ask fill: p_a
                # Prob no ask fill: 1-p_a
                # E[V] = p_a * V_next[i-1] + (1-p_a) * V_next[i]
                exp_V = p_a * V_next[i-1] + (1.0 - p_a) * V_next[i]
            elif q == -Q:
                # Only q->-Q (no ask fill) and q->-Q+1 (bid fill)
                # Prob bid fill: p_b
                # Prob no bid fill: 1-p_b
                # E[V] = p_b * V_next[i+1] + (1-p_b) * V_next[i]
                exp_V = p_b * V_next[i+1] + (1.0 - p_b) * V_next[i]
            else:
                # All transitions possible
                p_b_fill_a_no = p_b * (1.0 - p_a)
                p_a_fill_b_no = (1.0 - p_b) * p_a
                p_no_fill = (1.0 - p_b) * (1.0 - p_a)
                p_both_fill = p_b * p_a
                
                # If both fill, q stays q? 
                # Bid fill: q+1. Ask fill: q-1. Net: q.
                # So both fill leads to q.
                # No fill leads to q.
                # So prob(q) = p_no_fill + p_both_fill
                
                exp_V = p_b_fill_a_no * V_next[i+1] + \
                        p_a_fill_b_no * V_next[i-1] + \
                        (p_no_fill + p_both_fill) * V_next[i]
            
            # Immediate reward
            # d_b * p_b + d_a * p_a
            # If d is inf, reward is 0 (no fill)
            reward_b = 0.0 if np.isinf(d_b) else d_b * p_b
            reward_a = 0.0 if np.isinf(d_a) else d_a * p_a
            
            V_curr[i] = -cost + reward_b + reward_a + exp_V
        
        V_next = V_curr
    
    _dp_cache[key] = (bid_depths, ask_depths)
    return _dp_cache[key]

def quote(t, inventory, mid, params):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard
    if time.time() - _start_time > 295:
        k = params['k']
        base = 1.0 / k
        return np.full(len(inventory), base), np.full(len(inventory), base)

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n = int(t / (T / steps))
    if n >= steps:
        n = steps - 1
        
    bid_depths_table, ask_depths_table = _get_dp_tables(sigma, A, k, gamma, T, steps, Q)
    
    # Map inventory to index: q in [-Q, Q] -> idx in [0, 2Q]
    # idx = q + Q
    q = np.asarray(inventory, dtype=np.int32)
    idx = q + Q
    
    # Fetch depths
    bid_d = bid_depths_table[n, idx]
    ask_d = ask_depths_table[n, idx]
    
    # Ensure types are float
    bid_d = bid_d.astype(np.float64)
    ask_d = ask_d.astype(np.float64)
    
    # Safety: check for invalid values
    min_depth = -3.0 / k + 0.1
    max_depth = 100.0 / k # Cap large finite depths
    
    # Replace inf with a large value? No, np.inf is allowed for "no quote".
    # But we must ensure finite values are within range.
    
    # Clip finite values
    finite_mask_bid = np.isfinite(bid_d)
    bid_d[finite_mask_bid] = np.clip(bid_d[finite_mask_bid], min_depth, max_depth)
    
    finite_mask_ask = np.isfinite(ask_d)
    ask_d[finite_mask_ask] = np.clip(ask_d[finite_mask_ask], min_depth, max_depth)
    
    # Handle NaNs just in case
    bid_d[np.isnan(bid_d)] = min_depth
    ask_d[np.isnan(ask_d)] = min_depth
    
    return bid_d, ask_d
