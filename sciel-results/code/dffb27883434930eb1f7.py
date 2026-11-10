"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

sys.setrecursionlimit(100000)

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 320.0  # 10s safety margin before 330s limit

# Cache for pre-computed policies
_policy_cache = {}

def _compute_policy(params):
    """
    Compute optimal bid and ask depths for all steps and inventory levels.
    Uses vectorized DP with a fine hybrid depth grid.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_states = 2 * Q + 1
    q_min = -Q
    q_max = Q
    
    # V[t, q_idx] : Value function
    V = np.zeros((steps + 1, n_states))
    
    # Policy tables: optimal depth for each step and state
    # Initialize with a reasonable default (1/k)
    opt_bid = np.full((steps, n_states), 1.0 / k)
    opt_ask = np.full((steps, n_states), 1.0 / k)
    
    # Pre-compute depth grid
    # Hybrid grid: log-spaced for small depths (high sensitivity), linear for large
    # Range: [1e-4, 20/k]
    n_log = 400
    n_lin = 200
    # Ensure 1.0/k is well represented
    d_max = 20.0 / k
    
    # Log part: 1e-4 to 1.0
    d_log = np.geomspace(1e-4, 1.0, n_log)
    # Linear part: 1.0 to d_max
    d_lin = np.linspace(1.0, d_max, n_lin)
    
    # Combine and unique
    depth_grid = np.unique(np.concatenate([d_log, d_lin]))
    n_d = len(depth_grid)
    
    # Pre-compute probabilities P(d) = 1 - exp(-A * exp(-k*d) * dt)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        exp_term = np.exp(-k * depth_grid)
        P_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    # Clip probabilities to [0, 1] for numerical stability
    P_grid = np.clip(P_grid, 0.0, 1.0)
    
    # Pre-compute d * P(d) ? No, we need (d + delta) * P(d).
    # We will compute this in the loop as it depends on delta.
    
    # Inventory values for risk cost calculation
    q_vals = np.arange(q_min, q_max + 1, dtype=np.float64)
    risk_cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # DP Loop backwards in time
    for t in range(steps - 1, -1, -1):
        # Deadline check: if too slow, break and return current best (default 1/k)
        if time.time() > _DEADLINE:
            # Fill remaining steps with default policy
            for tt in range(t, steps):
                opt_bid[tt, :] = 1.0 / k
                opt_ask[tt, :] = 1.0 / k
            break
            
        V_next = V[t + 1]
        
        # Compute state value differences
        # Delta_plus[q] = V_next[q+1] - V_next[q]
        # For q < Q. For q = Q, no bid fill allowed, so we handle via masking.
        D_plus = np.zeros(n_states)
        D_minus = np.zeros(n_states)
        
        # D_plus[i] corresponds to state q = i - Q
        # V_next[i+1] - V_next[i] for i in 0..n_states-2
        D_plus[:-1] = V_next[1:] - V_next[:-1]
        # D_plus[-1] remains 0, but we will mask q=Q anyway
        
        # D_minus[i] = V_next[i-1] - V_next[i] for i in 1..n_states-1
        D_minus[1:] = V_next[:-1] - V_next[1:]
        # D_minus[0] remains 0, but we will mask q=-Q anyway
        
        # Vectorized Optimization
        # M_bid[i, j] = (depth_grid[j] + D_plus[i]) * P_grid[j]
        # We want max over j for each i.
        
        # Broadcasting: D_plus[:, None] + depth_grid[None, :]
        # Multiply by P_grid[None, :]
        
        # To avoid large memory allocation if n_states * n_d is huge, 
        # but 61 * 600 is small.
        
        M_bid = (D_plus[:, None] + depth_grid[None, :]) * P_grid[None, :]
        M_ask = (D_minus[:, None] + depth_grid[None, :]) * P_grid[None, :]
        
        # Find optimal depth index
        idx_bid = np.argmax(M_bid, axis=1)
        idx_ask = np.argmax(M_ask, axis=1)
        
        # Retrieve optimal depths
        d_bid_opt = depth_grid[idx_bid]
        d_ask_opt = depth_grid[idx_ask]
        
        # Retrieve max values for V update
        max_val_bid = M_bid[np.arange(n_states), idx_bid]
        max_val_ask = M_ask[np.arange(n_states), idx_ask]
        
        # Update Value Function
        # V[t] = max_bid + max_ask + V_next - risk_cost
        V[t] = max_val_bid + max_val_ask + V_next - risk_cost
        
        # Store optimal depths
        opt_bid[t] = d_bid_opt
        opt_ask[t] = d_ask_opt
        
        # Apply Boundary Constraints
        # If q == Q (index n_states-1), no bid fill. Set depth to inf.
        opt_bid[t, -1] = np.inf
        # If q == -Q (index 0), no ask fill. Set depth to inf.
        opt_ask[t, 0] = np.inf
        
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths for the given state.
    """
    # Deadline guard for the quote function itself (though policy is pre-computed)
    if time.time() > _DEADLINE:
        k = params["k"]
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Create a hashable key
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _policy_cache:
        _policy_cache[key] = _compute_policy(params)
    
    opt_bid, opt_ask = _policy_cache[key]
    
    # Determine step index
    dt = T / steps
    step_idx = int(round(t / dt))
    if step_idx >= steps:
        step_idx = steps - 1
    if step_idx < 0:
        step_idx = 0
        
    # Map inventory to index
    # inventory is an array of ints in [-Q, Q]
    inv_idx = inventory + Q
    
    # Safety clip for indices (though inputs should be valid)
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Retrieve depths
    bid_depths = opt_bid[step_idx, inv_idx]
    ask_depths = opt_ask[step_idx, inv_idx]
    
    # Ensure correct dtype and shape
    bid_depths = np.asarray(bid_depths, dtype=np.float64)
    ask_depths = np.asarray(ask_depths, dtype=np.float64)
    
    return (bid_depths, ask_depths)

# EVOLVE-BLOCK-END
