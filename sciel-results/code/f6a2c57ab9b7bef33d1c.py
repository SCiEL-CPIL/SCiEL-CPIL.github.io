"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np

_PARAMS_CACHE = {}

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid/ask depths for each (step, inventory) pair
    using backward induction on the discrete-time HJB equation.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1) with optimal bid depths
        ask_table: array of shape (steps, 2*Q+1) with optimal ask depths
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # Value function: V_next[q_idx] = optimal expected objective from next step with inventory q
    # V at terminal = 0 (no terminal charge)
    V_next = np.zeros(n_inv)
    
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # Depth grid: use a logarithmic grid from small to large depths
    min_depth = 0.001
    max_depth = 10.0 / k  # Increased max depth to ensure coverage
    n_depths = 100  # Increased resolution
    depth_grid = np.geomspace(min_depth, max_depth, n_depths)
    
    # Pre-compute fill probabilities for each depth
    # p[d] = 1 - exp(-A * exp(-k*d) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Inventory risk cost coefficient
    risk_coeff = 0.5 * gamma * sigma**2 * dt
    
    # Pre-compute inventory values
    q_vals = np.arange(n_inv) - Q
    
    # Prepare arrays for vectorization over inventory
    # Shapes: (1, 1, n_inv)
    V_q = np.zeros((1, 1, n_inv))
    V_q1 = np.zeros((1, 1, n_inv))
    V_qm1 = np.zeros((1, 1, n_inv))
    risk_cost = np.zeros((1, 1, n_inv))
    
    # Depth arrays for broadcasting
    # Shapes: (n_depths, 1, 1) and (1, n_depths, 1)
    PB = fill_probs[:, np.newaxis, np.newaxis]
    PA = fill_probs[np.newaxis, :, np.newaxis]
    B = depth_grid[:, np.newaxis, np.newaxis]
    Aa = depth_grid[np.newaxis, :, np.newaxis]
    
    for n in range(steps - 1, -1, -1):
        # Update V arrays for current step's V_next
        V_q[0, 0, :] = V_next
        # V_q1: V_next[q+1] if q < Q else 0
        V_q1[0, 0, :-1] = V_next[1:]
        V_q1[0, 0, -1] = 0.0
        # V_qm1: V_next[q-1] if q > -Q else 0
        V_qm1[0, 0, 1:] = V_next[:-1]
        V_qm1[0, 0, 0] = 0.0
        
        risk_cost[0, 0, :] = risk_coeff * (q_vals**2)
        
        # Compute E for both sides active
        # E = -risk_cost + PB*PA*(B+Aa+V_q) + PB*(1-PA)*(B+V_q1) + (1-PB)*PA*(Aa+V_qm1) + (1-PB)*(1-PA)*V_q
        E_both = (-risk_cost 
                  + PB * PA * (B + Aa + V_q) 
                  + PB * (1 - PA) * (B + V_q1) 
                  + (1 - PB) * PA * (Aa + V_qm1) 
                  + (1 - PB) * (1 - PA) * V_q)
        
        # Find optimal depths for both active
        # E_both shape: (n_depths, n_depths, n_inv)
        # We want argmax over first two axes for each inventory
        max_idx_flat = np.argmax(E_both, axis=(0, 1))
        # max_idx_flat shape: (n_inv,)
        
        bi_best = max_idx_flat // n_depths
        ai_best = max_idx_flat % n_depths
        
        bid_opt = depth_grid[bi_best]
        ask_opt = depth_grid[ai_best]
        V_curr = E_both[bi_best, ai_best, np.arange(n_inv)]
        
        # Handle boundaries: q = Q (index n_inv-1) -> no bid
        if n_inv > 1:
            # For q = Q
            # E_ask = -risk_cost + PA*(Aa + V_qm1) + (1-PA)*V_q
            # Note: V_q1 is 0, but term PB*(1-PA)*(B+V_q1) vanishes if we just calculate ask side.
            # Actually, if no bid, PB is effectively 0.
            # E = -risk_cost + PA*(Aa + V_qm1) + (1-PA)*V_q
            
            # Extract specific values for q=Q
            V_q_Q = V_next[-1]
            V_qm1_Q = V_next[-2]
            risk_Q = risk_coeff * (Q**2)
            
            # V_q and V_qm1 for q=Q are scalars in this context
            # PA shape (1, n_depths, 1) -> need to broadcast to (1, n_depths)
            # Let's just compute a 1D array for ask optimization
            E_ask_Q = (-risk_Q 
                       + PA[0, :, 0] * (Aa[0, :, 0] + V_qm1_Q) 
                       + (1 - PA[0, :, 0]) * V_q_Q)
            
            ai_best_Q = np.argmax(E_ask_Q)
            ask_opt[-1] = depth_grid[ai_best_Q]
            bid_opt[-1] = np.inf
            V_curr[-1] = E_ask_Q[ai_best_Q]
            
            # For q = -Q (index 0) -> no ask
            V_q_mQ = V_next[0]
            V_q1_mQ = V_next[1]
            risk_mQ = risk_coeff * ((-Q)**2)
            
            E_bid_mQ = (-risk_mQ 
                        + PB[:, 0, 0] * (B[:, 0, 0] + V_q1_mQ) 
                        + (1 - PB[:, 0, 0]) * V_q_mQ)
            
            bi_best_mQ = np.argmax(E_bid_mQ)
            bid_opt[0] = depth_grid[bi_best_mQ]
            ask_opt[0] = np.inf
            V_curr[0] = E_bid_mQ[bi_best_mQ]
            
            # For intermediate q, use the computed bid_opt/ask_opt/V_curr
            # But we must ensure that for q=Q and q=-Q, the "both active" calculation 
            # didn't incorrectly influence the neighbors? No, the vectorized calc is independent per q.
            # However, V_curr for intermediate q is already correct.
            # We just overwrite the boundary indices.
            
        # Store results
        bid_table[n, :] = bid_opt
        ask_table[n, :] = ask_opt
        V_next = V_curr
    
    return bid_table, ask_table

def _get_policy_table(params):
    """Get or compute the policy table for the given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _PARAMS_CACHE:
        _PARAMS_CACHE[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], 
            params['gamma'], params['T'], params['steps'], params['Q']
        )
    return _PARAMS_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Quote optimal bid/ask depths based on pre-computed policy table.
    
    Args:
        t: current time
        inventory: array of shape (4000,) with inventory levels
        mid: array of shape (4000,) with mid prices
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths): tuple of two arrays of shape (4000,)
    """
    bid_table, ask_table = _get_policy_table(params)
    
    steps = params['steps']
    Q = params['Q']
    
    # Determine current step
    T = params['T']
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Map inventory to table indices
    inv_idx = np.clip(inventory, -Q, Q) + Q
    inv_idx = inv_idx.astype(int)
    
    # Get depths from table
    bid_depths = bid_table[n, inv_idx].copy()
    ask_depths = ask_table[n, inv_idx].copy()
    
    # Ensure no bid at q=Q and no ask at q=-Q
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    # Ensure depths are valid (positive, not too small, not NaN)
    bid_depths = np.where(np.isfinite(bid_depths), np.maximum(bid_depths, 1e-6), bid_depths)
    ask_depths = np.where(np.isfinite(ask_depths), np.maximum(ask_depths, 1e-6), ask_depths)
    
    return bid_depths, ask_depths
