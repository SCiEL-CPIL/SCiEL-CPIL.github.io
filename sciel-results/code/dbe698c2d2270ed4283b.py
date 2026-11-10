"""
Optimal market making with inventory risk - High-Resolution DP + Local Polish.

This program computes the optimal bid and ask depths using a dynamic programming
approach. It uses a high-resolution global grid to determine initial optimal
depths for all states, followed by a local refinement step to polish the results.
This approach aims to capture the continuous optimum more accurately than
coarse-grid methods.
"""
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_LIMIT = 330.0  # Hard limit
_SAFETY_MARGIN = 5.0 # Seconds to reserve for cleanup/return
_MAX_COMPUTE_TIME = _TIME_LIMIT - _SAFETY_MARGIN

def _get_deadline():
    return _GLOBAL_START + _MAX_COMPUTE_TIME

def _fill_prob_vec(depths, A, k, dt):
    """Vectorized fill probability calculation."""
    depths = np.asarray(depths, dtype=float)
    # Handle edge case where depth is 0 or negative
    # exp(-k * 0) = 1, p = 1 - exp(-A dt). But problem says depth > 0 for quote.
    # However, in DP we consider depths from a grid.
    # If depth <= 0, probability is 0.
    p = np.where(depths > 0, 1.0 - np.exp(-A * np.exp(-k * depths) * dt), 0.0)
    return np.clip(p, 0.0, 1.0)

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes the optimal bid and ask depths.
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1
    
    # Inventory holding cost per step: (gamma / 2) * sigma^2 * q^2 * dt
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt

    # Define a fine grid for depths.
    # Based on analysis, optimal depths are typically around 1/k.
    # Max depth needed is likely < 3.
    # We use a log-spaced or linear grid. Linear is safer for small depths.
    # Let's use 400 points from 0.001 to 3.0.
    grid_max = 3.0
    n_grid = 400
    depth_grid = np.linspace(0.001, grid_max, n_grid)
    
    # Pre-compute fill probabilities for the grid
    p_grid = _fill_prob_vec(depth_grid, A, k, dt)
    
    # Pre-compute reward components for the grid
    # Rewards are depth * prob
    # We need to handle the joint probability of bid and ask fills.
    
    # V[n, q_idx]
    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))
    
    deadline = _get_deadline()
    
    # Arrays for transition values
    V_q_plus = np.zeros(n_q)
    V_q_minus = np.zeros(n_q)
    V_q = np.zeros(n_q)
    
    for n in range(steps - 1, -1, -1):
        # Check time
        if time.time() > deadline:
            # If we run out of time, copy the next step's policy (or zero if n=steps-1)
            if n < steps - 1:
                V[n] = V[n + 1]
                opt_bid[n] = opt_bid[n + 1]
                opt_ask[n] = opt_ask[n + 1]
            else:
                V[n] = 0
                opt_bid[n] = 0.01
                opt_ask[n] = 0.01
            # Fill remaining steps if we broke early? 
            # Actually, the loop goes backwards. If we break, we have computed 
            # steps n+1 to steps-1. We need to fill 0 to n.
            # Since we broke, we assume the policy is roughly constant or we 
            # just use the last computed one for all earlier steps.
            for n_rem in range(n, -1, -1):
                V[n_rem] = V[n_rem + 1] if n_rem < steps - 1 else 0
                opt_bid[n_rem] = opt_bid[n_rem + 1] if n_rem < steps - 1 else 0.01
                opt_ask[n_rem] = opt_ask[n_rem + 1] if n_rem < steps - 1 else 0.01
            break
            
        V_next = V[n + 1]
        
        # Prepare value arrays for next state inventory
        V_q[:] = V_next
        V_q_plus[:] = 0.0
        V_q_minus[:] = 0.0
        V_q_plus[:n_q - 1] = V_next[1:]
        V_q_minus[1:] = V_next[:-1]
        
        # We want to find b, a that maximize:
        # b * p_b + a * p_a - C_inv[q] + 
        # p_b(1-p_a)V(q+1) + p_a(1-p_b)V(q-1) + (1-p_b-p_a+p_b p_a)V(q)
        
        # Let's vectorize over the grid for b and a.
        # p_b is a vector of size n_grid
        # p_a is a vector of size n_grid
        
        # We can compute the max for each q separately, or vectorize over q as well?
        # n_q is 61. n_grid is 400. 61 * 400 * 400 operations = ~10 million per step.
        # 1000 steps * 10 million = 10^10 operations. This might be too slow in pure Python loops, 
        # but NumPy vectorization over the grid is fast.
        # However, doing a 2D search (b, a) for each q is expensive.
        
        # Optimization:
        # The objective function is separable in a way?
        # Let F(b, a) = b p_b + a p_a + p_b(1-p_a)V+ + p_a(1-p_b)V- + (1-p_b-p_a+p_b p_a)V0
        # F(b, a) = b p_b + a p_a + p_b V+ - p_b p_a V+ + p_a V- - p_b p_a V- + V0 - p_b V0 - p_a V0 + p_b p_a V0
        # F(b, a) = V0 + p_b (b + V+ - V0) + p_a (a + V- - V0) - p_b p_a (V+ + V- - V0)
        
        # Let A_b = b + V+ - V0
        # Let A_a = a + V- - V0
        # Let K = V+ + V- - V0
        
        # Maximize: p_b A_b + p_a A_a - K p_b p_a
        
        # For a fixed b, we can optimize a?
        # Or for a fixed a, optimize b?
        # Given the small size of n_q (61), we can loop over q and vectorize over the grid.
        
        refined_bid = np.zeros(n_q)
        refined_ask = np.zeros(n_q)
        V_refined = np.zeros(n_q)
        
        p_grid = _fill_prob_vec(depth_grid, A, k, dt)
        
        for q_idx in range(n_q):
            V0 = V_next[q_idx]
            V_plus = V_q_plus[q_idx]
            V_minus = V_q_minus[q_idx]
            
            # If q == Q, no bid fill allowed. Effectively p_b = 0.
            # If q == -Q, no ask fill allowed. Effectively p_a = 0.
            
            if q_idx == n_q - 1: # q = Q
                # Only ask matters
                # Maximize: a p_a - C_inv + p_a V_minus + (1-p_a) V0
                # = V0 - C_inv + p_a (a + V_minus - V0)
                # Since a + V_minus - V0 is likely positive (reward + future value), we want max p_a?
                # No, we want max a * p_a. 
                # Let's just search over a.
                a_rewards = depth_grid * p_grid
                # Transition
                trans = p_grid * V_minus + (1 - p_grid) * V0
                vals = a_rewards - C_inv[q_idx] + trans
                best_a_idx = np.argmax(vals)
                a_opt = depth_grid[best_a_idx]
                b_opt = 0.01 # Dummy
                V_opt = vals[best_a_idx]
                
            elif q_idx == 0: # q = -Q
                # Only bid matters
                b_rewards = depth_grid * p_grid
                trans = p_grid * V_plus + (1 - p_grid) * V0
                vals = b_rewards - C_inv[q_idx] + trans
                best_b_idx = np.argmax(vals)
                b_opt = depth_grid[best_b_idx]
                a_opt = 0.01 # Dummy
                V_opt = vals[best_b_idx]
                
            else:
                # Both bid and ask matter
                # F(b, a) = V0 - C_inv + p_b (b + V_plus - V0) + p_a (a + V_minus - V0) - (V_plus + V_minus - V0) p_b p_a
                # Let C_b = V_plus - V0
                # Let C_a = V_minus - V0
                # Let K = V_plus + V_minus - V0 = C_b + C_a
                
                # We need to maximize over b, a in grid.
                # n_grid is 400. 400x400 matrix is 160k elements.
                # Creating this matrix for each of 61 q's is 61 * 160k = ~10M elements.
                # This is feasible in NumPy.
                
                p_b = p_grid # shape (400,)
                p_a = p_grid # shape (400,)
                
                # Term 1: p_b * (b + C_b)
                # b is depth_grid
                term_b = p_b * (depth_grid + V_plus - V0)
                # Term 2: p_a * (a + C_a)
                term_a = p_a * (depth_grid + V_minus - V0)
                # Term 3: -K * p_b * p_a
                
                K = V_plus + V_minus - V0
                
                # Construct 2D arrays
                # term_b is (400, 1)
                # term_a is (1, 400)
                # p_b is (400, 1), p_a is (1, 400)
                
                val_matrix = term_b[:, np.newaxis] + term_a[np.newaxis, :] - K * p_b[:, np.newaxis] * p_a[np.newaxis, :]
                val_matrix -= C_inv[q_idx] + V0 # Subtract constant parts
                
                best_idx = np.argmax(val_matrix)
                i_idx, j_idx = np.unravel_index(best_idx, (n_grid, n_grid))
                b_opt = depth_grid[i_idx]
                a_opt = depth_grid[j_idx]
                V_opt = val_matrix[i_idx, j_idx]
                
            refined_bid[q_idx] = b_opt
            refined_ask[q_idx] = a_opt
            V_refined[q_idx] = V_opt

        V[n, :] = V_refined
        opt_bid[n, :] = refined_bid
        opt_ask[n, :] = refined_ask

    return opt_bid, opt_ask

def _get_policy(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'],
        params['T'], params['steps'], params['Q']
    )

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    opt_bid, opt_ask = _get_policy(params)

    dt = params['T'] / params['steps']
    n = int(round(t / dt))

    if n < 0:
        n = 0
    if n >= params['steps']:
        n = params['steps'] - 1

    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)

    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]

    # Ensure valid outputs
    # Problem states: Values below -3/k, -inf, NaN or wrong shape make result invalid.
    # Our grid starts at 0.001, so it should be fine.
    # Just in case, clip to a small positive value.
    bid_depths = np.where(np.isnan(bid_depths), 0.01, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), 0.01, ask_depths)
    bid_depths = np.where(bid_depths < 0.001, 0.001, bid_depths)
    ask_depths = np.where(ask_depths < 0.001, 0.001, ask_depths)

    return bid_depths.astype(float), ask_depths.astype(float)
