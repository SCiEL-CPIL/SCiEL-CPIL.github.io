"""Optimal market making with inventory risk via parametric ansatz + grid search."""
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 285.0
_CACHE = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve for optimal depth policy using a parametric ansatz:
    The marginal value of inventory (gradient of value function) is approximated
    as a quadratic in q: dv(q) = alpha + beta*q + delta*q^2
    
    Then optimal depths are:
      b_opt(q, n) = 1/k - dv(q)  [clamped to valid range]
      a_opt(q, n) = 1/k + dv(q)  [clamped to valid range]
    
    We optimize (alpha, beta, delta) via grid search to maximize a proxy objective
    that captures the key trade-offs.
    """
    dt = T / steps
    n_levels = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Pre-compute the time-dependent scaling factor
    # The inventory penalty accumulates as (gamma/2)*sigma^2*q^2*dt per step
    # Over n steps remaining, total penalty ≈ (gamma/2)*sigma^2*q^2*dt*n
    # This suggests the marginal value should scale with sqrt(n) or n
    
    # Parametric ansatz: dv(q, n) = alpha(n) + beta(n)*q + delta(n)*q^2
    # where alpha(n), beta(n), delta(n) are functions of time remaining n
    
    # For simplicity, assume:
    #   alpha(n) = 0  (symmetric problem)
    #   beta(n) = B * sqrt(n)  (linear in inventory, time-dependent)
    #   delta(n) = D * n  (quadratic in inventory, time-dependent)
    
    # Optimize B and D via grid search
    
    # Proxy objective: simulate the policy for a few representative paths
    # and compute the objective value
    
    best_B = 0.0
    best_D = 0.0
    best_score = -np.inf
    
    # Grid for B and D
    # B should be small: the marginal value per unit inventory
    # D should be negative: holding more inventory is costly
    
    # Estimate scales:
    # Inventory penalty per step: (gamma/2)*sigma^2*q^2*dt
    # For q=1: ~ gamma*sigma^2*dt/2
    # Fill probability: A*exp(-k*b)*dt ~ A*dt for small b
    # Expected fill gain: ~ b*A*dt
    
    # Let's use a coarse grid first
    B_range = np.linspace(-5.0, 5.0, 20)
    D_range = np.linspace(-0.5, 0.0, 10)
    
    # For each (B, D), compute a proxy score
    # Use a simplified single-path simulation with q=0,1,-1 as representative states
    
    for B in B_range:
        for D in D_range:
            # Compute proxy objective for a few representative scenarios
            total_obj = 0.0
            
            # Test with a few initial inventories
            for q_init in [0, 5, -5, 10, -10]:
                q = q_init
                obj = 0.0
                
                # Simulate a simplified version: assume mid is constant, 
                # and use expected fill probabilities
                for n in range(steps):
                    n_rem = steps - n
                    
                    # Compute depths
                    dv = B * np.sqrt(n_rem) + D * n_rem * q
                    b = 1.0 / k - dv
                    a = 1.0 / k + dv
                    
                    # Clamp
                    b = max(1e-6, min(b, 10.0 / k))
                    a = max(1e-6, min(a, 10.0 / k))
                    
                    # Fill probabilities
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    
                    # Expected immediate gain
                    exp_gain = p_b * b + p_a * a
                    
                    # Inventory penalty
                    penalty = 0.5 * gamma * sigma * sigma * q * q * dt
                    
                    obj += exp_gain - penalty
                    
                    # Expected inventory change
                    # Bid fill: q -> q+1 (if q < Q)
                    # Ask fill: q -> q-1 (if q > -Q)
                    dq_expected = 0.0
                    if q < Q:
                        dq_expected += p_b * 1.0
                    if q > -Q:
                        dq_expected -= p_a * 1.0
                    
                    q += dq_expected
                    q = max(-Q, min(Q, q))
                
                total_obj += obj
            
            score = total_obj / len([0, 5, -5, 10, -10])
            
            if score > best_score:
                best_score = score
                best_B = B
                best_D = D
    
    # Build the policy tables
    # For each step n (0 to steps-1) and each inventory q, store optimal depths
    opt_b = np.zeros((steps, n_levels))
    opt_a = np.zeros((steps, n_levels))
    
    for n in range(steps):
        n_rem = steps - n
        for q_idx in range(n_levels):
            q = q_idx - Q
            dv = best_B * np.sqrt(n_rem) + best_D * n_rem * q
            b = 1.0 / k - dv
            a = 1.0 / k + dv
            
            # Clamp to valid range
            b = max(1e-6, min(b, 10.0 / k))
            a = max(1e-6, min(a, 10.0 / k))
            
            opt_b[n, q_idx] = b
            opt_a[n, q_idx] = a
    
    return opt_b, opt_a


def quote(t, inventory, mid, params):
    """
    Compute optimal bid and ask depths for each path using pre-computed
    parametric policy.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a safe default: quote at 1/k depth
        n = len(inventory)
        bid_depths = np.full(n, 1.0 / k)
        ask_depths = np.full(n, 1.0 / k)
        return bid_depths, ask_depths
    
    # Cache key
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    
    if cache_key not in _CACHE:
        if time.time() > _DEADLINE - 30:
            # Not enough time for full solve, use simple heuristic
            n = len(inventory)
            bid_depths = np.full(n, 1.0 / k)
            ask_depths = np.full(n, 1.0 / k)
            return bid_depths, ask_depths
        
        opt_b, opt_a = _solve_regime(sigma, A, k, gamma, T, steps, Q)
        _CACHE[cache_key] = (opt_b, opt_a)
    
    opt_b, opt_a = _CACHE[cache_key]
    
    # Current step index
    n_current = int(round(t / dt))
    n_current = max(0, min(n_current, steps - 1))
    
    # Get inventory indices
    n_levels = 2 * Q + 1
    inv_idx = np.clip(inventory + Q, 0, n_levels - 1).astype(int)
    
    # Look up optimal depths
    bid_depths = opt_b[n_current, inv_idx]
    ask_depths = opt_a[n_current, inv_idx]
    
    # Ensure valid: no NaN, no -inf, values >= -3/k (should be positive)
    bid_depths = np.nan_to_num(bid_depths, nan=1.0/k, posinf=10.0/k, neginf=1.0/k)
    ask_depths = np.nan_to_num(ask_depths, nan=1.0/k, posinf=10.0/k, neginf=1.0/k)
    
    # Clamp to valid range
    min_depth = 1e-6
    max_depth = 10.0 / k
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    
    return bid_depths, ask_depths
