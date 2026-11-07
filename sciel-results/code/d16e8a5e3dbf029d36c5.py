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
    
    # Depth grid for search (vectorized)
    # Use a log-spaced grid for better resolution near zero where optimal depths tend to be
    max_d = 10.0 / k
    n_depths = 250
    # Log-spaced from a small positive value to max_d, plus 0
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
    
    # Terminal condition: V[n_steps, :] = 0 (no terminal charge)
    
    # Precompute inventory values and penalty
    inv_vals = np.arange(n_inv) - inv_offset  # -Q, ..., Q
    penalty = (gamma / 2.0) * sigma * sigma * inv_vals * inv_vals * dt
    
    # Backward induction
    for n in range(n_steps - 1, -1, -1):
        # Check time limit
        if time.time() - _START_TIME > TIME_LIMIT - 5.0:
            # Return what we have so far (remaining steps use heuristic)
            # Fill remaining with a simple heuristic
            for nn in range(n - 1, -1, -1):
                for ii in range(n_inv):
                    q_val = ii - inv_offset
                    d0 = 1.0 / k
                    bid_table[nn, ii] = d0
                    ask_table[nn, ii] = d0
            # Fix boundary conditions
            bid_table[:, 2 * Q] = np.inf  # q = Q, no bid
            ask_table[:, 0] = np.inf       # q = -Q, no ask
            return bid_table, ask_table
        
        # Vectorized computation across all inventory states simultaneously
        # For bid: states where q < Q can bid (indices 0 to 2Q-1)
        # For ask: states where q > -Q can ask (indices 1 to 2Q)
        
        # Compute optimal bid depths for all states at once
        # For each state ii, we maximize over depths: gain[d] + p_fill[d] * V[n+1, ii+1] + (1-p_fill[d]) * V[n+1, ii]
        # This equals: gain[d] + V[n+1, ii] + p_fill[d] * (V[n+1, ii+1] - V[n+1, ii])
        
        # Bid: valid for ii in 0..2Q-1 (q < Q)
        if 2 * Q > 0:
            # V_diff_bid[i] = V[n+1, i+1] - V[n+1, i] for i in 0..2Q-1
            V_diff_bid = V[n + 1, 1:] - V[n + 1, :-1]  # shape (2Q,)
            # For each state i, compute total over all depths
            # total_b[i, d] = gain[d] + V[n+1, i] + p_fill[d] * V_diff_bid[i]
            # = gain[d] + V[n+1, i] + p_fill[d] * V_diff_bid[i]
            
            # Vectorize: create matrix (2Q, n_opts)
            # p_fill is (n_opts,), V_diff_bid is (2Q,)
            # total_b[i, d] = gain[d] + V[n+1, i] + p_fill[d] * V_diff_bid[i]
            
            # Use broadcasting: p_fill[None, :] * V_diff_bid[:, None] gives (2Q, n_opts)
            p_term_bid = p_fill[None, :] * V_diff_bid[:, None]  # (2Q, n_opts)
            V_base_bid = V[n + 1, :-1][:, None]  # (2Q, 1)
            gain_row = gain[None, :]  # (1, n_opts)
            
            total_b = gain_row + V_base_bid + p_term_bid  # (2Q, n_opts)
            best_b_idx = np.argmax(total_b, axis=1)  # (2Q,)
            
            # Extract optimal depths
            bid_table[n, :2*Q] = all_depths[best_b_idx]
            
            # Compute optimal bid gain and prob for V update
            bid_gain_opt = gain[best_b_idx]  # (2Q,)
            bid_prob_opt = p_fill[best_b_idx]  # (2Q,)
        else:
            bid_gain_opt = np.zeros(0)
            bid_prob_opt = np.zeros(0)
        
        # Ask: valid for ii in 1..2Q (q > -Q)
        if 2 * Q > 0:
            # V_diff_ask[i] = V[n+1, i-1] - V[n+1, i] for i in 1..2Q
            V_diff_ask = V[n + 1, :-1] - V[n + 1, 1:]  # shape (2Q,)
            
            # total_a[i, d] = gain[d] + V[n+1, i] + p_fill[d] * V_diff_ask[i]
            # where i indexes into 1..2Q, so V[n+1, i] is V[n+1, 1:]
            
            p_term_ask = p_fill[None, :] * V_diff_ask[:, None]  # (2Q, n_opts)
            V_base_ask = V[n + 1, 1:][:, None]  # (2Q, 1)
            
            total_a = gain_row + V_base_ask + p_term_ask  # (2Q, n_opts)
            best_a_idx = np.argmax(total_a, axis=1)  # (2Q,)
            
            ask_table[n, 1:] = all_depths[best_a_idx]
            
            ask_gain_opt = gain[best_a_idx]  # (2Q,)
            ask_prob_opt = p_fill[best_a_idx]  # (2Q,)
        else:
            ask_gain_opt = np.zeros(0)
            ask_prob_opt = np.zeros(0)
        
        # Boundary conditions
        bid_table[n, 2*Q] = np.inf  # q = Q, no bid
        ask_table[n, 0] = np.inf     # q = -Q, no ask
        
        # Now compute V[n, ii] for all ii
        # V[n, ii] = bid_gain + ask_gain - penalty + exp_future
        
        # For interior states (1 <= ii <= 2Q-1), both bid and ask are available
        # For ii = 0 (q = -Q): only ask, no bid
        # For ii = 2Q (q = Q): only bid, no ask
        
        # Let's compute the expected future value contribution
        # For state ii:
        # p_b = bid_prob_opt[ii-1] if ii > 0 else 0
        # p_a = ask_prob_opt[ii-1] if ii < 2Q else 0
        # Wait, let me re-index.
        
        # bid_gain_opt[i] corresponds to state ii = i (for i in 0..2Q-1)
        # ask_gain_opt[i] corresponds to state ii = i+1 (for i in 0..2Q-1)
        
        # So for state ii:
        # bid_gain = bid_gain_opt[ii] if ii < 2Q else 0
        # ask_gain = ask_gain_opt[ii-1] if ii > 0 else 0
        
        # Let's create arrays for all states
        bid_gain_all = np.zeros(n_inv)
        bid_prob_all = np.zeros(n_inv)
        ask_gain_all = np.zeros(n_inv)
        ask_prob_all = np.zeros(n_inv)
        
        if 2 * Q > 0:
            bid_gain_all[:2*Q] = bid_gain_opt
            bid_prob_all[:2*Q] = bid_prob_opt
            ask_gain_all[1:] = ask_gain_opt
            ask_prob_all[1:] = ask_prob_opt
        
        # Expected future value for each state
        # p_both = p_b * p_a
        # p_bid_only = p_b * (1 - p_a)
        # p_ask_only = (1 - p_b) * p_a
        # p_neither = (1 - p_b) * (1 - p_a)
        
        p_b = bid_prob_all
        p_a = ask_prob_all
        
        p_both = p_b * p_a
        p_bid_only = p_b * (1.0 - p_a)
        p_ask_only = (1.0 - p_b) * p_a
        p_neither = (1.0 - p_b) * (1.0 - p_a)
        
        # Future values:
        # fv_bid: V[n+1, ii+1] for states where ii+1 <= 2Q, else 0
        fv_bid = np.zeros(n_inv)
        fv_bid[:2*Q] = V[n + 1, 1:]  # ii+1 for ii in 0..2Q-1
        
        # fv_ask: V[n+1, ii-1] for states where ii-1 >= 0, else 0
        fv_ask = np.zeros(n_inv)
        fv_ask[1:] = V[n + 1, :-1]  # ii-1 for ii in 1..2Q
        
        # fv_none = V[n+1, ii]
        fv_none = V[n + 1, :]
        
        # fv_both = V[n+1, ii] (same as fv_none)
        
        exp_future = p_both * fv_none + p_bid_only * fv_bid + p_ask_only * fv_ask + p_neither * fv_none
        
        # Simplify: exp_future = (p_both + p_neither) * fv_none + p_bid_only * fv_bid + p_ask_only * fv_ask
        # p_both + p_neither = 1 - p_bid_only - p_ask_only
        
        current_reward = bid_gain_all + ask_gain_all - penalty
        
        V[n, :] = current_reward + exp_future
    
    # Fix boundary conditions (should already be handled, but ensure)
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
