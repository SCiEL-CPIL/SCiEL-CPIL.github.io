"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for precomputed optimal policies
_POLICY_CACHE = {}
_START_TIME = time.time()
TIME_LIMIT = 300.0  # Safety margin before 330s kill

def _solve_mdp(params):
    """
    Solves the finite-horizon MDP for the given parameters.
    Uses a refined grid search for optimal depths.
    Returns arrays D_b and D_a of shape (steps, 2*Q + 1) containing optimal depths.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    c = A * dt  # Constant in the exponent for fill probability
    
    # Value function V[n, q_index]
    # q_index = q + Q
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal depths D_b[n, q_index], D_a[n, q_index]
    D_b = np.full((steps, 2 * Q + 1), np.inf)
    D_a = np.full((steps, 2 * Q + 1), np.inf)
    
    # Refined grid for depth search
    # The optimal depth is typically around 1/k.
    # We use a finer grid around this region to improve accuracy.
    # Range: [0, 5/k] is usually sufficient. 5/k gives prob ~ 1-exp(-A*dt*exp(-5)) which is small but non-zero.
    # Let's use 2000 points for better resolution.
    d_max = 5.0 / k
    d_grid = np.linspace(0, d_max, 2000)
    
    # Precompute probabilities for the grid
    # P(d) = 1 - exp(-c * exp(-k*d))
    exp_term = np.exp(-k * d_grid)
    P_grid = 1.0 - np.exp(-c * exp_term)
    
    # Precompute d_grid as float64 for speed
    d_grid = d_grid.astype(np.float64)
    P_grid = P_grid.astype(np.float64)
    
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > TIME_LIMIT - 10:
            # If we run out of time, break and return what we have (or fallback)
            # But since this is called once per regime, and we cache, 
            # it's better to just finish if we are close.
            # If we are way over, we might need a heuristic, but let's assume it fits.
            pass
        
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            
            # Cost
            cost = 0.5 * gamma * sigma * sigma * q * q * dt
            
            # Bid side
            if q < Q:
                M_b = V[n + 1, q + 1 + Q]  # V(n+1, q+1)
                # Maximize P_grid * (d_grid + M_b)
                # Vectorized operation
                obj_b = P_grid * (d_grid + M_b)
                idx_b = np.argmax(obj_b)
                d_b_opt = d_grid[idx_b]
                val_b = obj_b[idx_b]
            else:
                d_b_opt = np.inf
                val_b = 0.0
                
            # Ask side
            if q > -Q:
                M_a = V[n + 1, q - 1 + Q]  # V(n+1, q-1)
                # Maximize P_grid * (d_grid + M_a)
                obj_a = P_grid * (d_grid + M_a)
                idx_a = np.argmax(obj_a)
                d_a_opt = d_grid[idx_a]
                val_a = obj_a[idx_a]
            else:
                d_a_opt = np.inf
                val_a = 0.0
                
            D_b[n, q_idx] = d_b_opt
            D_a[n, q_idx] = d_a_opt
            
            V[n, q_idx] = val_b + val_a - cost
            
    return D_b, D_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Create a key for the cache
    key = (sigma, A, k, gamma, T, steps, Q)
    
    # Check if we have already solved this MDP
    if key not in _POLICY_CACHE:
        # Time guard
        # Solving 3 regimes * (1000 steps * 61 states * 2000 grid) 
        # ~ 3 * 122M operations. NumPy is fast, but let's be careful.
        # If we are already past 250s, we might not have time to solve all.
        # However, the first call usually happens early.
        _POLICY_CACHE[key] = _solve_mdp(params)
    
    D_b, D_a = _POLICY_CACHE[key]
    
    # Determine current step n
    dt = T / steps
    n = int(round(t / dt))
    # Clamp n to valid range [0, steps-1]
    if n < 0:
        n = 0
    elif n >= steps:
        n = steps - 1
        
    # Process inventory
    inv = np.asarray(inventory, dtype=np.int64)
    
    # Map inventory to index: q + Q
    # Ensure inventory is within [-Q, Q] for indexing
    # The problem guarantees [-Q, Q], but let's clip to be safe against indexing errors
    inv_clipped = np.clip(inv, -Q, Q)
    q_indices = inv_clipped + Q
    
    # Retrieve depths
    bid_depths = D_b[n, q_indices].astype(np.float64)
    ask_depths = D_a[n, q_indices].astype(np.float64)
    
    # Apply boundary constraints explicitly to be safe
    # No bid fill at q = Q
    bid_depths[inv >= Q] = np.inf
    # No ask fill at q = -Q
    ask_depths[inv <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
