"""Combined program: analytic inverse depth lookup + vectorized MDP backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s kill

_POLICY_CACHE = {}

def _build_depth_tables(A, k, dt):
    """
    Precompute a lookup table that, given a target fill probability p,
    returns the depth d such that 1 - exp(-A * exp(-k*d) * dt) = p.
    
    We invert analytically:
      p = 1 - exp(-c * exp(-k*d)) where c = A*dt
      exp(-c * exp(-k*d)) = 1 - p
      -c * exp(-k*d) = ln(1-p)
      exp(-k*d) = -ln(1-p) / c
      -k*d = ln(-ln(1-p) / c)
      d = -ln(-ln(1-p) / c) / k
    
    We build a table over p in (0, 1) and interpolate.
    """
    c = A * dt
    
    # We want p values that are achievable.
    # As d -> 0, p -> 1 - exp(-c)
    # As d -> inf, p -> 0
    # So p_max = 1 - exp(-c)
    # If c is very small, p_max is small.
    
    p_max = 1.0 - np.exp(-c)
    
    # If p_max is very small, the range of useful p is small.
    # Let's create a grid of p values from 0 to p_max.
    # But we need to handle the case where the optimal p might be outside [0, p_max]?
    # No, the fill probability is always in [0, p_max].
    # However, the objective function f(d) = p(d) * (d + M) might have an optimum
    # where we want p to be as small as possible (d -> inf) or as large as possible (d -> 0).
    # If M is very negative, we want p -> 0 (d -> inf).
    # If M is very positive, we want p -> p_max (d -> 0).
    
    # Let's create a grid of p values.
    # To be safe, let's include p=0 and p=p_max.
    # Number of points: 2000 should be enough for linear interpolation accuracy.
    n_p = 2000
    p_grid = np.linspace(0.0, p_max, n_p)
    
    # Compute corresponding d values
    # d(p) = -ln(-ln(1-p) / c) / k
    # Handle p=0: -ln(1) = 0, ln(0) is -inf. d -> inf.
    # Handle p=p_max: 1-p = exp(-c), -ln(1-p) = c, ln(c/c)=0, d=0.
    
    d_grid = np.full(n_p, np.inf)
    
    # For p > 0
    mask = p_grid > 0
    p_pos = p_grid[mask]
    
    # Compute -ln(1-p)
    # 1-p is in (1-p_max, 1] = [exp(-c), 1]
    # -ln(1-p) is in [0, c]
    
    log_term = -np.log(1.0 - p_pos)
    
    # Avoid division by zero if log_term is 0 (p=0, but we masked it)
    # log_term is 0 only if p=0.
    # If p is very small, log_term is very small.
    
    arg = log_term / c
    
    # arg is in (0, 1]
    # d = -ln(arg) / k
    
    # If arg == 1 (p=p_max), ln(1)=0, d=0.
    # If arg < 1, ln(arg) < 0, d > 0.
    
    d_vals = -np.log(arg) / k
    
    d_grid[mask] = d_vals
    
    # The grid is sorted by p ascending, so d is descending.
    # p: 0 -> p_max
    # d: inf -> 0
    
    return p_grid, d_grid

def _solve_mdp(params):
    """
    Solves the finite-horizon MDP for the given parameters.
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
    
    n_inv = 2 * Q + 1
    
    V = np.zeros((steps + 1, n_inv))
    D_b = np.full((steps, n_inv), np.inf)
    D_a = np.full((steps, n_inv), np.inf)
    
    inv_cost_coeff = (gamma / 2.0) * sigma**2 * dt
    
    # Precompute depth tables
    p_grid, d_grid = _build_depth_tables(A, k, dt)
    
    # Helper to find optimal depth for a given M (marginal value)
    # Maximize f(d) = p(d) * (d + M)
    # Let p = p(d). Then d = d(p).
    # We want to maximize g(p) = p * (d(p) + M) for p in [0, p_max].
    # We can evaluate g(p) on the p_grid and find the argmax.
    # This is O(n_p) per state.
    # 61k states * 2000 points = 122M operations.
    # This might be slow in Python loops, but we can vectorize over states if we batch them?
    # Or just accept the cost. 122M simple ops in NumPy is ~1s.
    # But we are doing this inside a Python loop for n and q.
    # 1000 * 61 = 61000 calls to a vectorized op of size 2000.
    # That is 61000 * 2000 = 122M flops.
    # Let's try to optimize by precomputing d_grid + M? No, M changes.
    
    # Alternative:
    # Note that g(p) = p * d(p) + p * M.
    # h(p) = p * d(p). This is independent of M!
    # So g(p) = h(p) + p * M.
    # We can precompute h(p) for all p in the grid.
    # Then for a given M, we just need to find argmax of h(p_grid) + p_grid * M.
    # This is still O(n_p).
    
    # Let's precompute h(p) = p * d(p).
    # d(p) is inf for p=0. 0 * inf is NaN.
    # Limit p->0: p * (-ln(-ln(1-p)/c)/k)
    # Let u = -ln(1-p) ~ p.
    # p ~ u.
    # d ~ -ln(u/c)/k.
    # p * d ~ u * -ln(u/c)/k.
    # As u->0, u*ln(u) -> 0.
    # So h(0) = 0.
    
    h_p = np.zeros_like(p_grid)
    mask = p_grid > 0
    h_p[mask] = p_grid[mask] * d_grid[mask]
    
    # Now for each state, we have M.
    # We want to maximize h_p + M * p_grid.
    # Let's store h_p and p_grid.
    
    for n in range(steps - 1, -1, -1):
        # Time check
        if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
            # Fallback: fill remaining with 1/k
            for q_idx in range(n_inv):
                D_b[n, q_idx] = 1.0 / k
                D_a[n, q_idx] = 1.0 / k
            break
        
        for q_idx in range(n_inv):
            q = q_idx - Q
            inv_cost = -inv_cost_coeff * q**2
            
            # Bid side
            if q < Q:
                M_b = V[n + 1, q_idx + 1]
                # Maximize h_p + M_b * p_grid
                obj_b = h_p + M_b * p_grid
                idx_b = np.argmax(obj_b)
                d_b_opt = d_grid[idx_b]
                val_b = obj_b[idx_b]
            else:
                d_b_opt = np.inf
                val_b = 0.0
                
            # Ask side
            if q > -Q:
                M_a = V[n + 1, q_idx - 1]
                obj_a = h_p + M_a * p_grid
                idx_a = np.argmax(obj_a)
                d_a_opt = d_grid[idx_a]
                val_a = obj_a[idx_a]
            else:
                d_a_opt = np.inf
                val_a = 0.0
                
            D_b[n, q_idx] = d_b_opt
            D_a[n, q_idx] = d_a_opt
            
            V[n, q_idx] = val_b + val_a + inv_cost
            
    return D_b, D_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _solve_mdp(params)
        
    D_b, D_a = _POLICY_CACHE[key]
    
    dt = T / steps
    n = int(round(t / dt))
    if n < 0:
        n = 0
    elif n >= steps:
        n = steps - 1
        
    inv = np.asarray(inventory, dtype=np.int64)
    q_indices = inv + Q
    
    bid_depths = D_b[n, q_indices].astype(np.float64)
    ask_depths = D_a[n, q_indices].astype(np.float64)
    
    # Boundary constraints
    bid_depths[inv >= Q] = np.inf
    ask_depths[inv <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
