"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np
from functools import lru_cache
import time

# Global cache for policy tables
_POLICY_CACHE = {}

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
    inv_range = np.arange(-Q, Q + 1)
    
    # Value function: V[n, q] = optimal expected objective from step n with inventory q
    # We work backwards from n = steps-1 to n = 0
    V = np.zeros(n_inv)  # Terminal value = 0 (no terminal charge)
    
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # Search range for depths: from very small to large
    # The optimal depth is typically around 1/k to a few/k
    d_min = 1e-6
    d_max = 10.0 / k
    
    # We'll use a grid search for the optimal depth, then refine
    # Number of grid points for initial search
    n_grid = 200
    
    for n in range(steps - 1, -1, -1):
        # For each inventory level, find optimal bid and ask depths
        for idx, q in enumerate(inv_range):
            # Inventory penalty for this step
            penalty = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            # No bid fill at q = Q, no ask fill at q = -Q
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            # Optimal bid depth
            if can_bid:
                # If we bid at depth d_bid, with probability p_bid we get a fill
                # Fill earns d_bid and moves inventory to q+1
                # With probability 1-p_bid, no fill, inventory stays q
                # Next step value is V[q] (for no fill) or V[q+1] (for fill)
                
                # We need to maximize: -penalty + p_bid * (d_bid + V_next[q+1]) + (1-p_bid) * V_next[q]
                # where p_bid = 1 - exp(-A * exp(-k * d_bid) * dt)
                
                # Let's define the objective as a function of d_bid
                V_next = V.copy()
                
                def bid_obj(d):
                    p = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                    # Expected value from bidding
                    # If fill: earn d, go to q+1
                    # If no fill: go to q
                    return p * (d + V_next[idx + 1]) + (1.0 - p) * V_next[idx]
                
                # Grid search
                grid = np.linspace(d_min, d_max, n_grid)
                objs = np.array([bid_obj(d) for d in grid])
                best_idx = np.argmax(objs)
                
                # Refine with golden section search
                a, b = grid[max(0, best_idx - 1)], grid[min(n_grid - 1, best_idx + 1)]
                if a == b:
                    best_d = a
                else:
                    # Golden section search
                    gr = (np.sqrt(5) - 1) / 2
                    c = b - gr * (b - a)
                    d = a + gr * (b - a)
                    fc = bid_obj(c)
                    fd = bid_obj(d)
                    for _ in range(50):
                        if fc < fd:
                            a = c
                            c = d
                            fc = fd
                            d = a + gr * (b - a)
                            fd = bid_obj(d)
                        else:
                            b = d
                            d = c
                            fd = fc
                            c = b - gr * (b - a)
                            fc = bid_obj(c)
                    best_d = (a + b) / 2.0
                
                bid_table[n, idx] = best_d
                # Update V for this state
                p_best = 1.0 - np.exp(-A * np.exp(-k * best_d) * dt)
                V[idx] = -penalty + p_best * (best_d + V_next[idx + 1]) + (1.0 - p_best) * V_next[idx]
            else:
                bid_table[n, idx] = np.inf
                V[idx] = -penalty + V[idx]
            
            # Optimal ask depth
            if can_ask:
                V_next = V.copy()
                
                def ask_obj(d):
                    p = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                    # If fill: earn d, go to q-1
                    # If no fill: go to q
                    return p * (d + V_next[idx - 1]) + (1.0 - p) * V_next[idx]
                
                # Grid search
                grid = np.linspace(d_min, d_max, n_grid)
                objs = np.array([ask_obj(d) for d in grid])
                best_idx = np.argmax(objs)
                
                # Refine with golden section search
                a, b = grid[max(0, best_idx - 1)], grid[min(n_grid - 1, best_idx + 1)]
                if a == b:
                    best_d = a
                else:
                    gr = (np.sqrt(5) - 1) / 2
                    c = b - gr * (b - a)
                    d = a + gr * (b - a)
                    fc = ask_obj(c)
                    fd = ask_obj(d)
                    for _ in range(50):
                        if fc < fd:
                            a = c
                            c = d
                            fc = fd
                            d = a + gr * (b - a)
                            fd = ask_obj(d)
                        else:
                            b = d
                            d = c
                            fd = fc
                            c = b - gr * (b - a)
                            fc = ask_obj(c)
                    best_d = (a + b) / 2.0
                
                ask_table[n, idx] = best_d
                # Update V for this state (ask fill moves inventory to q-1)
                p_best = 1.0 - np.exp(-A * np.exp(-k * best_d) * dt)
                V[idx] = p_best * (best_d + V_next[idx - 1]) + (1.0 - p_best) * V_next[idx]
            else:
                ask_table[n, idx] = np.inf
        
        # After processing all inventories for step n, V now represents V[n]
        # But we need to be careful: V was updated in-place. Let's redo this properly.
    
    # The above has a bug: V is updated in-place during the loop, which affects subsequent calculations
    # Let's rewrite with proper separation
    
    # Actually, let's restart with a cleaner implementation
    V_next = np.zeros(n_inv)  # Terminal value
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_inv)
        
        for idx, q in enumerate(inv_range):
            penalty = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            # Optimal bid depth
            best_bid_val = 0.0
            best_bid_d = np.inf
            if can_bid:
                def bid_obj(d):
                    p = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                    return p * (d + V_next[idx + 1]) + (1.0 - p) * V_next[idx]
                
                grid = np.linspace(d_min, d_max, n_grid)
                objs = np.array([bid_obj(d) for d in grid])
                best_idx = np.argmax(objs)
                
                a, b = grid[max(0, best_idx - 1)], grid[min(n_grid - 1, best_idx + 1)]
                if a == b:
                    best_bid_d = a
                else:
                    gr = (np.sqrt(5) - 1) / 2
                    c = b - gr * (b - a)
                    d = a + gr * (b - a)
                    fc = bid_obj(c)
                    fd = bid_obj(d)
                    for _ in range(40):
                        if fc < fd:
                            a = c
                            c = d
                            fc = fd
                            d = a + gr * (b - a)
                            fd = bid_obj(d)
                        else:
                            b = d
                            d = c
                            fd = fc
                            c = b - gr * (b - a)
                            fc = bid_obj(c)
                    best_bid_d = (a + b) / 2.0
                
                best_bid_val = bid_obj(best_bid_d)
            
            # Optimal ask depth
            best_ask_val = 0.0
            best_ask_d = np.inf
            if can_ask:
                def ask_obj(d):
                    p = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                    return p * (d + V_next[idx - 1]) + (1.0 - p) * V_next[idx]
                
                grid = np.linspace(d_min, d_max, n_grid)
                objs = np.array([ask_obj(d) for d in grid])
                best_idx = np.argmax(objs)
                
                a, b = grid[max(0, best_idx - 1)], grid[min(n_grid - 1, best_idx + 1)]
                if a == b:
                    best_ask_d = a
                else:
                    gr = (np.sqrt(5) - 1) / 2
                    c = b - gr * (b - a)
                    d = a + gr * (b - a)
                    fc = ask_obj(c)
                    fd = ask_obj(d)
                    for _ in range(40):
                        if fc < fd:
                            a = c
                            c = d
                            fc = fd
                            d = a + gr * (b - a)
                            fd = ask_obj(d)
                        else:
                            b = d
                            d = c
                            fd = fc
                            c = b - gr * (b - a)
                            fc = ask_obj(c)
                    best_ask_d = (a + b) / 2.0
                
                best_ask_val = ask_obj(best_ask_d)
            
            bid_table[n, idx] = best_bid_d
            ask_table[n, idx] = best_ask_d
            V_curr[idx] = -penalty + best_bid_val + best_ask_val
        
        V_next = V_curr
    
    return bid_table, ask_table


def _get_policy_table(params):
    """Get or compute the policy table for given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    steps = params['steps']
    Q = params['Q']
    n_paths = len(inventory)
    
    # Get step index
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(steps - 1, n))
    
    # Get policy table
    bid_table, ask_table = _get_policy_table(params)
    
    # Map inventory to index: q in [-Q, Q] -> idx in [0, 2Q]
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Get depths for each path
    bid_depths = bid_table[n, inv_idx]
    ask_depths = ask_table[n, inv_idx]
    
    # Ensure valid values: depths must be >= -3/k
    min_depth = -3.0 / params['k']
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Replace inf with a large but valid value if needed (shouldn't happen for valid inventory)
    # Actually, inf is allowed per the spec ("np.inf means no quote")
    
    return bid_depths, ask_depths
