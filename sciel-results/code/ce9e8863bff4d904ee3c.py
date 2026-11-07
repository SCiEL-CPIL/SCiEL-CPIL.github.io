"""
Optimal market making via pre-computed DP policy lookup with analytical depth optimization.
Uses the concavity of the Bellman objective in fill probability to solve for optimal
depths in closed form, eliminating the discrete grid search entirely.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 325.0  # safety margin before 330s kill


def _optimal_depth_analytical(delta, A, k, dt):
    """
    Solve for optimal depth d that maximizes:
        d * (1 - exp(-A * exp(-k*d) * dt)) + delta * (1 - exp(-A * exp(-k*d) * dt))
    
    Substituting p = 1 - exp(-A * exp(-k*d) * dt):
        d(p) = -ln(1-p)/k
        Objective: -ln(1-p)/k * p + delta * p = p * (delta - ln(1-p)/k)
    
    For delta > 0: interior solution exists
    For delta <= 0: p* = 0 (d = inf)
    
    Returns optimal depth (float), np.inf if no quote is optimal.
    """
    if delta <= 0:
        return np.inf
    
    # Let u = 1 - p, so p = 1 - u
    # Objective: (1-u) * (delta + ln(u)/k)
    # d/du: -1*(delta + ln(u)/k) + (1-u)/k = 0
    # => delta + ln(u)/k + (1-u)/k = 0
    # => k*delta + ln(u) + 1 - u = 0
    # => ln(u) - u = -k*delta - 1
    # => u - ln(u) = k*delta + 1
    
    # Let f(u) = u - ln(u). f'(u) = 1 - 1/u < 0 for u < 1, so f is decreasing on (0,1).
    # f(1) = 1, f(u) -> inf as u -> 0+.
    # We need f(u) = k*delta + 1.
    # If k*delta + 1 <= 1, i.e., delta <= 0, no solution (handled above).
    # For delta > 0, k*delta + 1 > 1, so there's a unique u in (0, 1).
    
    target = k * delta + 1.0
    
    # Solve u - ln(u) = target for u in (0, 1)
    # Use Newton's method
    # f(u) = u - ln(u) - target
    # f'(u) = 1 - 1/u
    
    # Initial guess: for small target, u is close to 1
    # u - ln(u) ≈ 1 + (u-1)^2/2 for u near 1
    # target ≈ 1 + (u-1)^2/2 => u ≈ 1 - sqrt(2*(target-1))
    # But for larger target, u is smaller
    
    # Use a robust approach: bisection combined with Newton
    lo, hi = 1e-15, 1.0 - 1e-15
    
    # Check if target is achievable
    # f(1) = 1, so if target <= 1, no solution (but delta > 0 ensures target > 1)
    # f(lo) is very large, so solution exists
    
    # Newton with bisection fallback
    u = max(0.5, 1.0 - np.sqrt(2.0 * (target - 1.0)))  # initial guess
    u = np.clip(u, lo, hi)
    
    for _ in range(100):
        f_val = u - np.log(u) - target
        f_prime = 1.0 - 1.0 / u
        
        if abs(f_val) < 1e-15:
            break
        
        # Newton step
        if abs(f_prime) > 1e-15:
            u_new = u - f_val / f_prime
            if u_new <= lo or u_new >= hi:
                # Bisection
                if f_val > 0:
                    hi = u
                else:
                    lo = u
                u = 0.5 * (lo + hi)
            else:
                u = u_new
        else:
            if f_val > 0:
                hi = u
            else:
                lo = u
            u = 0.5 * (lo + hi)
        
        u = np.clip(u, lo, hi)
    
    # p* = 1 - u
    p_star = 1.0 - u
    
    # d* = -ln(1 - p*)/k = -ln(u)/k
    if u <= 0:
        return np.inf
    
    d_star = -np.log(u) / k
    
    # Also check d = 0 (p = 1 - exp(-A*dt))
    p_zero = 1.0 - np.exp(-A * dt)
    obj_zero = 0.0 + delta * p_zero  # d=0, gain=0, future = delta * p_zero
    
    # Check interior
    obj_interior = p_star * (delta - np.log(u) / k)
    
    if obj_interior > obj_zero:
        return d_star
    else:
        return 0.0


def _compute_optimal_policy(params):
    """
    Compute optimal bid/ask depths for all (step, inventory) states via backward DP
    with analytical depth optimization.
    
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
    
    # DP tables
    V = np.zeros((n_steps + 1, n_inv))  # value function
    bid_table = np.full((n_steps, n_inv), np.inf)
    ask_table = np.full((n_steps, n_inv), np.inf)
    
    # Terminal condition: V[n_steps, :] = 0 (no terminal charge)
    
    # Precompute inventory values and penalty
    inv_vals = np.arange(n_inv) - inv_offset  # -Q, ..., Q
    penalty = (gamma / 2.0) * sigma * sigma * inv_vals * inv_vals * dt
    
    # Backward induction
    for n in range(n_steps - 1, -1, -1):
        # Check time limit
        if time.time() - _START_TIME > TIME_LIMIT - 5.0:
            # Return what we have so far (remaining steps use heuristic)
            for nn in range(n - 1, -1, -1):
                for ii in range(n_inv):
                    bid_table[nn, ii] = 1.0 / k
                    ask_table[nn, ii] = 1.0 / k
            bid_table[:, 2 * Q] = np.inf
            ask_table[:, 0] = np.inf
            return bid_table, ask_table
        
        # Compute V_diff for bid and ask
        # V_diff_bid[i] = V[n+1, i+1] - V[n+1, i] for i in 0..2Q-1
        # V_diff_ask[i] = V[n+1, i-1] - V[n+1, i] for i in 1..2Q (index i-1 in 0..2Q-1)
        
        if 2 * Q > 0:
            V_diff_bid = V[n + 1, 1:] - V[n + 1, :-1]  # shape (2Q,)
            V_diff_ask = V[n + 1, :-1] - V[n + 1, 1:]  # shape (2Q,)
            
            # Solve for optimal depths analytically
            bid_depths = np.empty(2 * Q)
            ask_depths = np.empty(2 * Q)
            
            for i in range(2 * Q):
                bid_depths[i] = _optimal_depth_analytical(V_diff_bid[i], A, k, dt)
                ask_depths[i] = _optimal_depth_analytical(V_diff_ask[i], A, k, dt)
            
            bid_table[n, :2*Q] = bid_depths
            ask_table[n, 1:] = ask_depths
            
            # Compute fill probabilities for V update
            bid_prob = np.zeros(2 * Q)
            ask_prob = np.zeros(2 * Q)
            
            finite_b = np.isfinite(bid_depths)
            if np.any(finite_b):
                bid_prob[finite_b] = 1.0 - np.exp(-A * np.exp(-k * bid_depths[finite_b]) * dt)
            
            finite_a = np.isfinite(ask_depths)
            if np.any(finite_a):
                ask_prob[finite_a] = 1.0 - np.exp(-A * np.exp(-k * ask_depths[finite_a]) * dt)
            
            # Gains
            bid_gain = np.zeros(2 * Q)
            ask_gain = np.zeros(2 * Q)
            bid_gain[finite_b] = bid_depths[finite_b] * bid_prob[finite_b]
            ask_gain[finite_a] = ask_depths[finite_a] * ask_prob[finite_a]
            
            # Build full arrays
            bid_gain_all = np.zeros(n_inv)
            bid_prob_all = np.zeros(n_inv)
            ask_gain_all = np.zeros(n_inv)
            ask_prob_all = np.zeros(n_inv)
            
            bid_gain_all[:2*Q] = bid_gain
            bid_prob_all[:2*Q] = bid_prob
            ask_gain_all[1:] = ask_gain
            ask_prob_all[1:] = ask_prob
        else:
            bid_gain_all = np.zeros(n_inv)
            bid_prob_all = np.zeros(n_inv)
            ask_gain_all = np.zeros(n_inv)
            ask_prob_all = np.zeros(n_inv)
        
        # Boundary conditions
        bid_table[n, 2*Q] = np.inf  # q = Q, no bid
        ask_table[n, 0] = np.inf     # q = -Q, no ask
        
        # Expected future value
        p_b = bid_prob_all
        p_a = ask_prob_all
        
        p_both = p_b * p_a
        p_bid_only = p_b * (1.0 - p_a)
        p_ask_only = (1.0 - p_b) * p_a
        p_neither = (1.0 - p_b) * (1.0 - p_a)
        
        fv_bid = np.zeros(n_inv)
        fv_ask = np.zeros(n_inv)
        fv_none = V[n + 1, :]
        
        if 2 * Q > 0:
            fv_bid[:2*Q] = V[n + 1, 1:]
            fv_ask[1:] = V[n + 1, :-1]
        
        exp_future = (p_both + p_neither) * fv_none + p_bid_only * fv_bid + p_ask_only * fv_ask
        
        current_reward = bid_gain_all + ask_gain_all - penalty
        
        V[n, :] = current_reward + exp_future
    
    # Fix boundary conditions
    bid_table[:, 2 * Q] = np.inf
    ask_table[:, 0] = np.inf
    
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
