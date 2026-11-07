"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 10s safety margin before 330s kill

_policy_cache = {}

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth tables via backward induction.
    Returns (bid_table, ask_table) each of shape (steps, 2*Q+1).
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_indices = np.arange(-Q, Q + 1)  # q = -Q, ..., Q
    
    # Value function: V[i, j] where j is index into inv_indices
    V = np.zeros((steps + 1, n_inv))
    
    # Optimal depths
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # For optimization, we'll use a grid of depths and interpolate
    # Depth range: from very small to large
    # exp(-k*d) ranges from 1 (d=0) to ~0 (d large)
    # Fill prob = 1 - exp(-A * exp(-k*d) * dt)
    # For small A*dt*exp(-k*d), fill prob ≈ A*exp(-k*d)*dt
    # Optimal depth is around where marginal benefit = marginal cost
    
    # Use a reasonable grid of depths
    # The optimal depth is typically O(1/k) to O(10/k)
    max_depth = 20.0 / k
    n_depth = 200
    depth_grid = np.linspace(0.01, max_depth, n_depth)
    
    # Precompute fill probabilities for the grid
    # fill_prob[d] = 1 - exp(-A * exp(-k*d) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        remaining = steps - n  # number of steps remaining including this one
        # For each inventory level
        for j in range(n_inv):
            q = inv_indices[j]
            
            # If at Q, can't have bid fill (no buying at max inventory)
            # If at -Q, can't have ask fill (no selling at min inventory)
            
            best_val = -np.inf
            best_bid = 1.0 / k
            best_ask = 1.0 / k
            
            # We need to jointly optimize bid and ask depths
            # This is a 2D optimization. Let's use a coarse grid then refine.
            
            # For efficiency, let's try a smaller grid first
            n_coarse = 40
            coarse_grid = np.linspace(0.01, max_depth, n_coarse)
            coarse_fill = 1.0 - np.exp(-A * np.exp(-k * coarse_grid) * dt)
            
            for i_bid in range(n_coarse):
                bid_d = coarse_grid[i_bid]
                bid_p = coarse_fill[i_bid]
                
                # If q == Q, bid fill is not allowed
                if q >= Q:
                    bid_p = 0.0
                
                for i_ask in range(n_coarse):
                    ask_d = coarse_grid[i_ask]
                    ask_p = coarse_fill[i_ask]
                    
                    # If q == -Q, ask fill is not allowed
                    if q <= -Q:
                        ask_p = 0.0
                    
                    # Expected value
                    # No fill: prob (1-bid_p)(1-ask_p)
                    # Bid fill only: bid_p * (1-ask_p)
                    # Ask fill only: (1-bid_p) * ask_p
                    # Both fill: bid_p * ask_p
                    
                    # Inventory transitions:
                    # No fill: q
                    # Bid fill: q+1 (but capped at Q)
                    # Ask fill: q-1 (but floored at -Q)
                    # Both: q (net zero)
                    
                    q_no = q
                    q_bid = min(q + 1, Q)
                    q_ask = max(q - 1, -Q)
                    q_both = q  # net zero
                    
                    j_no = q_no + Q
                    j_bid = q_bid + Q
                    j_ask = q_ask + Q
                    j_both = q_both + Q
                    
                    # Immediate reward
                    reward_no = 0.0
                    reward_bid = bid_d
                    reward_ask = ask_d
                    reward_both = bid_d + ask_d
                    
                    # Inventory cost for this step
                    inv_cost = (gamma / 2.0) * sigma * sigma * q * q * dt
                    
                    # Continuation value
                    val_no = V[n + 1, j_no]
                    val_bid = V[n + 1, j_bid]
                    val_ask = V[n + 1, j_ask]
                    val_both = V[n + 1, j_both]
                    
                    total = (
                        (1 - bid_p) * (1 - ask_p) * (reward_no + val_no) +
                        bid_p * (1 - ask_p) * (reward_bid + val_bid) +
                        (1 - bid_p) * ask_p * (reward_ask + val_ask) +
                        bid_p * ask_p * (reward_both + val_both)
                    ) - inv_cost
                    
                    if total > best_val:
                        best_val = total
                        best_bid = bid_d
                        best_ask = ask_d
            
            # Refine around the best point with a finer grid
            if best_val > -np.inf:
                # Fine grid around the best point
                margin = max_depth / n_coarse * 3
                lo = max(0.01, best_bid - margin)
                hi = min(max_depth, best_bid + margin)
                fine_bid = np.linspace(lo, hi, 30)
                fine_bid_p = 1.0 - np.exp(-A * np.exp(-k * fine_bid) * dt)
                
                lo_a = max(0.01, best_ask - margin)
                hi_a = min(max_depth, best_ask + margin)
                fine_ask = np.linspace(lo_a, hi_a, 30)
                fine_ask_p = 1.0 - np.exp(-A * np.exp(-k * fine_ask) * dt)
                
                if q >= Q:
                    fine_bid_p = np.zeros_like(fine_bid_p)
                if q <= -Q:
                    fine_ask_p = np.zeros_like(fine_ask_p)
                
                best_val_fine = -np.inf
                best_bid_fine = best_bid
                best_ask_fine = best_ask
                
                for i_bid in range(len(fine_bid)):
                    bid_d = fine_bid[i_bid]
                    bid_p = fine_bid_p[i_bid]
                    
                    for i_ask in range(len(fine_ask)):
                        ask_d = fine_ask[i_ask]
                        ask_p = fine_ask_p[i_ask]
                        
                        q_no = q
                        q_bid = min(q + 1, Q)
                        q_ask = max(q - 1, -Q)
                        q_both = q
                        
                        j_no = q_no + Q
                        j_bid = q_bid + Q
                        j_ask = q_ask + Q
                        j_both = q_both + Q
                        
                        reward_no = 0.0
                        reward_bid = bid_d
                        reward_ask = ask_d
                        reward_both = bid_d + ask_d
                        
                        inv_cost = (gamma / 2.0) * sigma * sigma * q * q * dt
                        
                        val_no = V[n + 1, j_no]
                        val_bid = V[n + 1, j_bid]
                        val_ask = V[n + 1, j_ask]
                        val_both = V[n + 1, j_both]
                        
                        total = (
                            (1 - bid_p) * (1 - ask_p) * (reward_no + val_no) +
                            bid_p * (1 - ask_p) * (reward_bid + val_bid) +
                            (1 - bid_p) * ask_p * (reward_ask + val_ask) +
                            bid_p * ask_p * (reward_both + val_both)
                        ) - inv_cost
                        
                        if total > best_val_fine:
                            best_val_fine = total
                            best_bid_fine = bid_d
                            best_ask_fine = ask_d
                
                best_val = best_val_fine
                best_bid = best_bid_fine
                best_ask = best_ask_fine
            
            V[n, j] = best_val
            bid_table[n, j] = best_bid
            ask_table[n, j] = best_ask
        
        # Check deadline
        if time.time() > _DEADLINE - 5:
            # Fill remaining with default values
            for n2 in range(n - 1, -1, -1):
                for j2 in range(n_inv):
                    V[n2, j2] = V[n2 + 1, j2]
                    bid_table[n2, j2] = 1.0 / k
                    ask_table[n2, j2] = 1.0 / k
            break
    
    return bid_table, ask_table


def _get_policy(params):
    """Get or compute the policy table for the given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        if time.time() > _DEADLINE - 10:
            # Not enough time, return default policy
            Q = params['Q']
            steps = params['steps']
            k = params['k']
            default_depth = np.full((steps, 2 * Q + 1), 1.0 / k)
            _policy_cache[key] = (default_depth, default_depth.copy())
        else:
            _policy_cache[key] = _compute_policy(
                params['sigma'], params['A'], params['k'], params['gamma'],
                params['T'], params['steps'], params['Q']
            )
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    
    Parameters:
    - t: current time
    - inventory: (4000,) array of inventory levels
    - mid: (4000,) array of mid prices
    - params: dict with sigma, A, k, gamma, T, steps, Q
    
    Returns:
    - (bid_depths, ask_depths): two arrays of shape (4000,)
    """
    steps = params['steps']
    Q = params['Q']
    k = params['k']
    
    # Determine current step index
    dt = params['T'] / steps
    step_idx = int(round(t / dt))
    step_idx = min(max(step_idx, 0), steps - 1)
    
    # Get policy tables
    bid_table, ask_table = _get_policy(params)
    
    # Map inventory to table index
    # inventory is in [-Q, Q]
    inv_clipped = np.clip(inventory, -Q, Q).astype(int)
    table_indices = inv_clipped + Q
    
    # Look up depths
    bid_depths = bid_table[step_idx, table_indices]
    ask_depths = ask_table[step_idx, table_indices]
    
    # Ensure valid values: clip to [1e-6, large]
    bid_depths = np.clip(bid_depths, 1e-6, 100.0)
    ask_depths = np.clip(ask_depths, 1e-6, 100.0)
    
    # Handle boundary conditions: no bid at Q, no ask at -Q
    # At q=Q, we should not quote bid (or quote very far away)
    # At q=-Q, we should not quote ask
    mask_no_bid = (inventory >= Q)
    mask_no_ask = (inventory <= -Q)
    
    if np.any(mask_no_bid):
        bid_depths[mask_no_bid] = np.inf
    if np.any(mask_no_ask):
        ask_depths[mask_no_ask] = np.inf
    
    return bid_depths, ask_depths
