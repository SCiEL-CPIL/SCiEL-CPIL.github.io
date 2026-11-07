"""Optimal market making with inventory risk - DP pre-computation."""
# EVOLVE-BLOCK-START
import numpy as np

_policy_cache = {}

def _precompute_policy(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _policy_cache:
        return _policy_cache[key]
        
    T = params['T']
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    N = params['steps']
    dt = T / N
    
    n_q = 2 * Q + 1
    # Grid for depths
    # Range: -3/k to 5/k
    grid_min = -3.0 / k
    grid_max = 5.0 / k
    n_grid = 100
    depths = np.linspace(grid_min, grid_max, n_grid)
    
    # Pre-compute probabilities for grid
    exponents = A * np.exp(-k * depths) * dt
    p_grid = 1.0 - np.exp(-exponents)
    p_grid = np.clip(p_grid, 0.0, 1.0)
    
    # V_next for all q
    V_next = np.zeros(n_q)
    
    opt_b = np.zeros((N, n_q))
    opt_a = np.zeros((N, n_q))
    
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    penalty = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # Pre-compute reward parts
    reward_b = depths * (1.0 - p_grid)
    reward_a = depths * (1.0 - p_grid)
    
    for n in range(N - 1, -1, -1):
        V_curr = np.zeros(n_q)
        
        # Shifts
        V_q = V_next
        V_q_plus = np.roll(V_next, -1)
        V_q_plus[-1] = V_next[-1]
        V_q_minus = np.roll(V_next, 1)
        V_q_minus[0] = V_next[0]
        
        # Shapes for broadcasting
        # V terms: (n_q, 1, 1)
        Vq = V_q[:, np.newaxis, np.newaxis]
        Vq_plus = V_q_plus[:, np.newaxis, np.newaxis]
        Vq_minus = V_q_minus[:, np.newaxis, np.newaxis]
        
        # P terms: (n_q, n_grid, 1) and (n_q, 1, n_grid)
        # Start with broadcasted (1, n_grid, 1) and (1, 1, n_grid)
        pb = p_grid[np.newaxis, :, np.newaxis]
        pa = p_grid[np.newaxis, np.newaxis, :]
        
        # Expand to (n_q, n_grid, 1) and (n_q, 1, n_grid)
        pb_full = np.broadcast_to(pb, (n_q, n_grid, 1))
        pa_full = np.broadcast_to(pa, (n_q, 1, n_grid))
        
        # Apply boundary constraints: p_b=0 at q=Q, p_a=0 at q=-Q
        pb_full[-1, :, 0] = 0.0
        pa_full[0, 0, :] = 0.0
        
        # Expectation
        exp_v = (1-pb_full)*(1-pa_full)*Vq + pb_full*(1-pa_full)*Vq_plus + (1-pb_full)*pa_full*Vq_minus + pb_full*pa_full*Vq
        
        # Immediate Reward
        # reward_b: (1, n_grid, 1) -> broadcast to (n_q, n_grid, 1)
        # reward_a: (1, 1, n_grid) -> broadcast to (n_q, 1, n_grid)
        # penalty: (n_q, 1, 1)
        
        J = reward_b[np.newaxis, :, :] + reward_a[np.newaxis, :, :] - penalty[:, np.newaxis, np.newaxis] + exp_v
        
        # Find max
        max_idx = np.argmax(J, axis=(1, 2))
        b_idx, a_idx = np.unravel_index(max_idx, (n_grid, n_grid))
        
        opt_b[n, :] = depths[b_idx]
        opt_a[n, :] = depths[a_idx]
        
        # Map max grid index to inf
        opt_b[n, b_idx == n_grid - 1] = np.inf
        opt_a[n, a_idx == n_grid - 1] = np.inf
        
        V_curr = J[np.arange(n_q), b_idx, a_idx]
        V_next = V_curr
        
    _policy_cache[key] = (opt_b, opt_a)
    return opt_b, opt_a

def quote(t, inventory, mid, params):
    opt_b, opt_a = _precompute_policy(params)
    
    T = params['T']
    steps = params['steps']
    dt = T / steps
    n = int(t / dt)
    if n >= steps: n = steps - 1
    if n < 0: n = 0
    
    Q = params['Q']
    q_idx = inventory + Q
    
    bid_depths = opt_b[n, q_idx]
    ask_depths = opt_a[n, q_idx]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
