"""
Optimal market making via pre-computed DP policy lookup.
The state space is (step, inventory) with 1000 steps and 61 inventory levels (-30..30).
We solve the Bellman equation backwards to find optimal bid/ask depths for each state.
Results are cached per regime so only the first call pays the DP cost.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 325.0  # safety margin before 330s kill


def _compute_optimal_policy(params):
    """
    Compute optimal bid/ask depths for all (step, inventory) states via backward DP.
    Vectorized across inventory levels for speed.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1) - optimal bid depth for each state
        ask_table: array of shape (steps, 2*Q+1) - optimal ask depth for each state
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    T = params['T']
    n_steps = params.get('steps', 1000)
    dt = T / n_steps
    
    n_inv = 2 * Q + 1
    inv_offset = Q  # inventory index: -Q -> 0, 0 -> Q, Q -> 2Q
    
    # Depth grid for search - log-spaced for better resolution near zero
    max_d = 10.0 / k
    n_depths = 300
    eps_d = 1e-6
    depth_grid = np.concatenate([[0.0], np.geomspace(eps_d, max_d, n_depths - 1)])
    # Also include np.inf as "no quote" option
    all_depths = np.concatenate([depth_grid, [np.inf]])
    n_opts = len(all_depths)
    
    # Precompute fill probabilities for all depths
    # p_fill(d) = 1 - exp(-A * exp(-k*d) * dt)
    # For d = inf, p_fill = 0
    p_fill = np.zeros(n_opts)
    finite_mask = np.isfinite(all_depths)
    if np.any(finite_mask):
        exp_term = A * np.exp(-k * all_depths[finite_mask]) * dt
        p_fill[finite_mask] = 1.0 - np.exp(-exp_term)
    
    # Expected gain from quoting at depth d: d * p_fill(d)
    # For d = inf, gain = 0
    gain = np.zeros(n_opts)
    gain[finite_mask] = all_depths[finite_mask] * p_fill[finite_mask]
    
    # DP tables
    V = np.zeros((n_steps + 1, n_inv))  # value function
    bid_table = np.zeros((n_steps, n_inv))
    ask_table = np.zeros((n_steps, n_inv))
    
    # Precompute inventory values and penalty
    inv_vals = np.arange(n_inv) - inv_offset  # -Q, ..., Q
    penalty = (gamma / 2.0) * sigma * sigma * inv_vals * inv_vals * dt
    
    # Boundary masks: which inventory levels can bid/ask
    can_bid = np.ones(n_inv, dtype=bool)
    can_bid[-1] = False  # q = Q, no bid
    can_ask = np.ones(n_inv, dtype=bool)
    can_ask[0] = False    # q = -Q, no ask
    
    # Backward induction
    for n in range(n_steps - 1, -1, -1):
        # Check time limit
        if time.time() - _START_TIME > TIME_LIMIT - 5.0:
            # Return what we have so far (remaining steps use heuristic)
            d0 = 1.0 / k
            for nn in range(n - 1, -1, -1):
                bid_table[nn, :] = d0
                ask_table[nn, :] = d0
            # Fix boundary conditions
            bid_table[:, 2 * Q] = np.inf  # q = Q, no bid
            ask_table[:, 0] = np.inf       # q = -Q, no ask
            return bid_table, ask_table
        
        V_next = V[n + 1]
        
        # --- Vectorized bid optimization across all inventory levels ---
        # For inventory ii (0..n_inv-1), q_val = ii - inv_offset
        # Bid fill moves inventory from ii to ii+1 (if ii < n_inv-1)
        # V_next[ii+1] is the value after bid fill; V_next[ii] is value if no fill
        # p_b * V_next[ii+1] + (1-p_b) * V_next[ii]
        
        # Shape: (n_inv, n_opts) for future values
        # V_next[ii+1] for each ii: shift V_next by 1
        V_next_shift_up = np.zeros(n_inv)
        V_next_shift_up[:-1] = V_next[1:]
        V_next_shift_up[-1] = V_next[-1]  # boundary: stays at Q
        
        # future_val_b[ii, j] = p_fill[j] * V_next_shift_up[ii] + (1-p_fill[j]) * V_next[ii]
        # = V_next[ii] + p_fill[j] * (V_next_shift_up[ii] - V_next[ii])
        # Shape: (n_inv, 1) + (1, n_opts) -> (n_inv, n_opts)
        delta_V_b = (V_next_shift_up - V_next).reshape(n_inv, 1)  # (n_inv, 1)
        future_val_b = V_next.reshape(n_inv, 1) + p_fill.reshape(1, n_opts) * delta_V_b
        total_b = future_val_b + gain.reshape(1, n_opts)
        
        best_b_idx = np.argmax(total_b, axis=1)  # (n_inv,)
        bid_table[n, :] = all_depths[best_b_idx]
        bid_gain_arr = gain[best_b_idx]  # (n_inv,)
        bid_prob_arr = p_fill[best_b_idx]  # (n_inv,)
        
        # Apply boundary: no bid at q=Q
        bid_table[n, 2 * Q] = np.inf
        bid_gain_arr = bid_gain_arr.copy()
        bid_prob_arr = bid_prob_arr.copy()
        bid_gain_arr[2 * Q] = 0.0
        bid_prob_arr[2 * Q] = 0.0
        
        # --- Vectorized ask optimization across all inventory levels ---
        # Ask fill moves inventory from ii to ii-1 (if ii > 0)
        V_next_shift_down = np.zeros(n_inv)
        V_next_shift_down[1:] = V_next[:-1]
        V_next_shift_down[0] = V_next[0]  # boundary: stays at -Q
        
        delta_V_a = (V_next_shift_down - V_next).reshape(n_inv, 1)
        future_val_a = V_next.reshape(n_inv, 1) + p_fill.reshape(1, n_opts) * delta_V_a
        total_a = future_val_a + gain.reshape(1, n_opts)
        
        best_a_idx = np.argmax(total_a, axis=1)
        ask_table[n, :] = all_depths[best_a_idx]
        ask_gain_arr = gain[best_a_idx]
        ask_prob_arr = p_fill[best_a_idx]
        
        # Apply boundary: no ask at q=-Q
        ask_table[n, 0] = np.inf
        ask_gain_arr = ask_gain_arr.copy()
        ask_prob_arr = ask_prob_arr.copy()
        ask_gain_arr[0] = 0.0
        ask_prob_arr[0] = 0.0
        
        # --- Compute V[n, :] from joint outcomes ---
        # p_both = p_b * p_a, p_bid_only = p_b*(1-p_a), p_ask_only = (1-p_b)*p_a, p_neither = (1-p_b)*(1-p_a)
        p_b = bid_prob_arr  # (n_inv,)
        p_a = ask_prob_arr  # (n_inv,)
        
        p_both = p_b * p_a
        p_bid_only = p_b * (1.0 - p_a)
        p_ask_only = (1.0 - p_b) * p_a
        p_neither = (1.0 - p_b) * (1.0 - p_a)
        
        # Expected future value:
        # fv_both = V_next[ii] (both fill: net change 0)
        # fv_bid = V_next[ii+1] (bid only: +1)
        # fv_ask = V_next[ii-1] (ask only: -1)
        # fv_none = V_next[ii] (neither: 0)
        
        fv_both = V_next  # (n_inv,)
        fv_bid = V_next_shift_up  # (n_inv,)
        fv_ask = V_next_shift_down  # (n_inv,)
        fv_none = V_next  # (n_inv,)
        
        exp_future = p_both * fv_both + p_bid_only * fv_bid + p_ask_only * fv_ask + p_neither * fv_none
        
        # Current reward: bid_gain + ask_gain - penalty
        current_reward = bid_gain_arr + ask_gain_arr - penalty
        
        V[n, :] = current_reward + exp_future
    
    # Fix boundary conditions
    bid_table[:, 2 * Q] = np.inf  # q = Q
    ask_table[:, 0] = np.inf       # q = -Q
    
    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """
    Return optimal bid/ask depths using pre-computed DP policy.
    """
    T = params["T"]
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    Q = params["Q"]
    n_steps = params.get("steps", 1000)
    dt = T / n_steps
    
    # Create cache key
    cache_key = (sigma, A, k, gamma, Q, T, n_steps)
    
    if cache_key not in _policy_cache:
        bid_table, ask_table = _compute_optimal_policy(params)
        _policy_cache[cache_key] = (bid_table, ask_table)
    else:
        bid_table, ask_table = _policy_cache[cache_key]
    
    # Determine current step
    n = int(round(t / dt))
    n = np.clip(n, 0, n_steps - 1)
    
    # Map inventory to index: -Q -> 0, 0 -> Q, Q -> 2Q
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[n, inv_idx].astype(np.float64)
    ask_depths = ask_table[n, inv_idx].astype(np.float64)
    
    # Ensure valid values
    min_valid = -3.0 / k
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= min_valid), ask_depths, np.inf)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
