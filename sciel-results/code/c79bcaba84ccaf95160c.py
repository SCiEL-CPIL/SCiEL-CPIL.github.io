"""Optimal market making with inventory risk using discrete-time HJB."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar
from functools import lru_cache

# Precompute HJB value functions and optimal quotes for each regime
# We'll compute these lazily on first call for each (sigma, A, k, gamma) combination

_hjb_cache = {}

def _compute_hjb(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the discrete-time HJB equation backwards to get optimal bid/ask depths
    for each inventory level and time step.
    
    Returns:
        bid_depths: shape (steps+1, 2*Q+1) - optimal bid depth for each (t, q)
        ask_depths: shape (steps+1, 2*Q+1) - optimal ask depth for each (t, q)
    """
    dt = T / steps
    n_states = 2 * Q + 1
    
    # Value function V[t][q] where t is step index (0..steps), q is inventory offset
    # V[steps][q] = 0 (no terminal charge)
    V = np.zeros((steps + 1, n_states))
    
    # Optimal depths
    bid_depths = np.zeros((steps + 1, n_states))
    ask_depths = np.zeros((steps + 1, n_states))
    
    # Precompute mid-move std
    sigma_sqrt_dt = sigma * np.sqrt(dt)
    
    # We need to optimize over bid_depth b >= 0 and ask_depth a >= 0
    # Objective for step t, inventory q:
    #   - (gamma/2) * sigma^2 * q^2 * dt  (holding cost)
    #   + (1 - exp(-A * exp(-k*b) * dt)) * (b - sigma^2 * dt * q)  ... wait
    #   
    # Let me re-derive. The objective per path is:
    #   sum over steps of [depths earned] - sum over steps of (gamma/2) sigma^2 q^2 dt
    #
    # At step t with inventory q (before fills):
    #   - Bid hit prob: p_b = 1 - exp(-A * exp(-k*b) * dt)
    #   - Ask hit prob: p_a = 1 - exp(-A * exp(-k*a) * dt)
    #   - If bid hit: earn b, inventory becomes q-1
    #   - If ask hit: earn a, inventory becomes q+1
    #   - Both can't happen? The problem says "independently" so both can happen.
    #   - If both: earn b + a, inventory unchanged (q-1+1 = q)
    #   - Holding cost: -(gamma/2) * sigma^2 * q^2 * dt
    #
    # Expected continuation:
    #   p_b * (1-p_a) * V[t+1][q-1] + (1-p_b) * p_a * V[t+1][q+1] 
    #   + p_b * p_a * V[t+1][q] + (1-p_b)*(1-p_a)*V[t+1][q]
    #
    # But we also need to account for the mid movement? The mid moves but since
    # we're quoting relative to mid, the absolute mid doesn't affect the relative
    # depths. The inventory cost depends on q, not mid. So mid movement doesn't
    # affect the HJB in this formulation.
    
    for t in range(steps - 1, -1, -1):
        for q_idx in range(n_states):
            q = q_idx - Q  # actual inventory
            
            # Holding cost
            hold_cost = -(gamma / 2.0) * sigma**2 * q**2 * dt
            
            # If q == Q, no bid fill possible; if q == -Q, no ask fill possible
            bid_allowed = (q < Q)
            ask_allowed = (q > -Q)
            
            def objective(ba_tuple):
                b, a = ba_tuple
                if b < 0 or a < 0:
                    return 0.0  # invalid
                
                # Compute probabilities
                if bid_allowed:
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                else:
                    p_b = 0.0
                
                if ask_allowed:
                    p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                else:
                    p_a = 0.0
                
                # Immediate reward
                reward = p_b * b + p_a * a + hold_cost
                
                # Continuation value
                # States: q-1 (bid only), q+1 (ask only), q (both or neither)
                # Index mapping: q -> q + Q
                val_q = V[t + 1, q_idx]
                
                if bid_allowed:
                    val_qm1 = V[t + 1, q_idx - 1]
                else:
                    val_qm1 = 0.0
                
                if ask_allowed:
                    val_qp1 = V[t + 1, q_idx + 1]
                else:
                    val_qp1 = 0.0
                
                # Expected continuation
                # P(bid only) = p_b * (1 - p_a)
                # P(ask only) = (1 - p_b) * p_a
                # P(both) = p_b * p_a
                # P(neither) = (1 - p_b) * (1 - p_a)
                
                p_bid_only = p_b * (1.0 - p_a)
                p_ask_only = (1.0 - p_b) * p_a
                p_both = p_b * p_a
                p_neither = (1.0 - p_b) * (1.0 - p_a)
                
                cont = p_bid_only * val_qm1 + p_ask_only * val_qp1 + p_both * val_q + p_neither * val_q
                cont = p_bid_only * val_qm1 + p_ask_only * val_qp1 + (p_both + p_neither) * val_q
                
                total = reward + cont
                return -total  # minimize negative for maximization
            
            # Optimize over b and a
            # Use nested optimization: for each b, optimize a, then optimize b
            # Or use 2D optimization. Let's try a grid + local refinement.
            
            # For efficiency, let's use a reasonable search range
            # Optimal depth is typically O(1/k)
            max_depth = 5.0 / k if k > 0 else 5.0
            max_depth = min(max_depth, 10.0)
            
            # Coarse grid
            n_grid = 20
            b_grid = np.linspace(0, max_depth, n_grid)
            a_grid = np.linspace(0, max_depth, n_grid)
            
            best_val = -np.inf
            best_b = 0.0
            best_a = 0.0
            
            for b in b_grid:
                # For this b, find optimal a
                def obj_a(a_val):
                    return objective((b, a_val))
                
                res = minimize_scalar(obj_a, bounds=(0, max_depth), method='bounded',
                                       options={'xatol': 1e-4})
                a_opt = res.x
                val = -res.fun  # objective returns -total, so total = -res.fun
                
                if val > best_val:
                    best_val = val
                    best_b = b
                    best_a = a_opt
            
            # Refine around best_b
            b_lo = max(0, best_b - max_depth / n_grid)
            b_hi = min(max_depth, best_b + max_depth / n_grid)
            
            def obj_b(b_val):
                def obj_a_inner(a_val):
                    return objective((b_val, a_val))
                res = minimize_scalar(obj_a_inner, bounds=(0, max_depth), method='bounded',
                                       options={'xatol': 1e-4})
                return res.fun
            
            res_b = minimize_scalar(obj_b, bounds=(b_lo, b_hi), method='bounded',
                                     options={'xatol': 1e-4})
            best_b = res_b.x
            
            # Get optimal a for best_b
            def obj_a_final(a_val):
                return objective((best_b, a_val))
            res_a = minimize_scalar(obj_a_final, bounds=(0, max_depth), method='bounded',
                                     options={'xatol': 1e-4})
            best_a = res_a.x
            
            bid_depths[t, q_idx] = best_b
            ask_depths[t, q_idx] = best_a
            V[t, q_idx] = -objective((best_b, best_a))
    
    return bid_depths, ask_depths


def _get_hjb_tables(params):
    """Get or compute HJB tables for given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    if key not in _hjb_cache:
        _hjb_cache[key] = _compute_hjb(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _hjb_cache[key]


def quote(t, inventory, mid, params):
    """
    Quote optimal bid/ask depths based on precomputed HJB solution.
    
    Returns:
        (bid_depths, ask_depths): two float arrays of shape (N,)
    """
    N = len(inventory)
    Q = params['Q']
    steps = params['steps']
    
    # Determine current step index
    dt = params['T'] / steps
    step_idx = int(round(t / dt))
    step_idx = min(step_idx, steps - 1)
    
    # Get precomputed tables
    bid_table, ask_table = _get_hjb_tables(params)
    
    # Map inventory to state index
    # inventory is in [-Q, Q], state index = q + Q
    inv_indices = np.clip(inventory, -Q, Q) + Q
    
    # Get depths for each path
    bid_depths = bid_table[step_idx, inv_indices].astype(np.float64)
    ask_depths = ask_table[step_idx, inv_indices].astype(np.float64)
    
    # Enforce constraints: no bid at q=Q, no ask at q=-Q
    # If inventory == Q, bid should be inf (no quote)
    # If inventory == -Q, ask should be inf (no quote)
    at_max_inv = (inventory == Q)
    at_min_inv = (inventory == -Q)
    
    bid_depths[at_max_inv] = np.inf
    ask_depths[at_min_inv] = np.inf
    
    # Ensure all values are valid (non-negative, finite or inf)
    # Clip small negative values to 0
    bid_depths = np.where(bid_depths < 0, 0.0, bid_depths)
    ask_depths = np.where(ask_depths < 0, 0.0, ask_depths)
    
    # Replace NaN with inf (no quote)
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
