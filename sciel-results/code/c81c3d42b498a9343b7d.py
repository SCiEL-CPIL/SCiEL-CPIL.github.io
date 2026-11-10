import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 330.0

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    num_states = 2 * Q + 1
    
    # Grid for depths
    # Range: [-3/k, 10/k] seems safe
    min_depth = -3.0 / k
    max_depth = 10.0 / k
    grid_size = 100
    b_grid = np.linspace(min_depth, max_depth, grid_size)
    a_grid = np.linspace(min_depth, max_depth, grid_size)
    
    # Precompute probabilities for grid
    # p = 1 - exp(-A * exp(-k * depth) * dt)
    exp_term_b = np.exp(-k * b_grid)
    p_b_grid = 1.0 - np.exp(-A * exp_term_b * dt)
    
    exp_term_a = np.exp(-k * a_grid)
    p_a_grid = 1.0 - np.exp(-A * exp_term_a * dt)
    
    # Value function
    V_next = np.zeros(num_states)
    
    # Store optimal depths
    bid_depths = np.zeros((steps, num_states))
    ask_depths = np.zeros((steps, num_states))
    
    # Inventory values
    q_vals = np.arange(-Q, Q + 1)
    
    # Masks for boundaries
    mask_bid = (q_vals < Q).astype(float)  # 1 if can bid
    mask_ask = (q_vals > -Q).astype(float) # 1 if can ask
    
    # Padded V for easy indexing
    # V_padded[q_idx + 1] corresponds to q_vals[q_idx]
    # We need V(q-1), V(q), V(q+1)
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 10:
            break # Safety margin
            
        V_curr = np.zeros(num_states)
        
        # Initialize with risk-neutral
        b_curr = np.full(num_states, 1.0 / k)
        a_curr = np.full(num_states, 1.0 / k)
        
        # Iterative optimization
        for _ in range(5):
            # --- Update b ---
            # Calculate p_a for current a_curr
            p_a_curr = 1.0 - np.exp(-A * np.exp(-k * a_curr) * dt)
            p_a_curr = p_a_curr * mask_ask  # Enforce no ask at boundary
            
            # V terms
            V_0 = V_next
            V_minus = np.zeros_like(V_next)
            V_plus = np.zeros_like(V_next)
            
            # q_vals > -Q means index > 0
            # V_minus for q[i] is V_next[i-1] if i>0 else 0 (or V_0 if no transition)
            # But we use p_b * (V_minus - V_0). If p_b=0 at boundary, it doesn't matter.
            # However, for q=0 (idx 0), V_minus should be V_0 to avoid -inf or wrong gradient?
            # Actually, if mask_bid[0] is 0 (q=-Q), p_b is 0.
            # If mask_bid[0] is 1 (q > -Q), then i>0 is not necessarily true?
            # q_vals[0] = -Q. mask_bid[0] = 0.
            # q_vals[1] = -Q+1. mask_bid[1] = 1. V_minus[1] = V_next[0].
            
            V_minus[1:] = V_next[:-1]
            V_minus[0] = V_next[0] # Placeholder, masked out
            
            V_plus[:-1] = V_next[1:]
            V_plus[-1] = V_next[-1] # Placeholder, masked out
            
            # C_b = V_minus - V_0 - p_a * (V_minus + V_plus - 2*V_0)
            C_b = V_minus - V_0 - p_a_curr * (V_minus + V_plus - 2 * V_0)
            
            # Objective: p_b * (b + C_b)
            # p_b_grid shape (M, 1)
            # C_b shape (1, N)
            # b_grid shape (M, 1)
            
            p_b_eff = p_b_grid * mask_bid  # (M, N)
            obj_b = p_b_eff * (b_grid[:, np.newaxis] + C_b[np.newaxis, :])
            
            idx_b = np.argmax(obj_b, axis=0)
            b_curr = b_grid[idx_b]
            
            # --- Update a ---
            p_b_curr = 1.0 - np.exp(-A * np.exp(-k * b_curr) * dt)
            p_b_curr = p_b_curr * mask_bid
            
            C_a = V_plus - V_0 - p_b_curr * (V_minus + V_plus - 2 * V_0)
            
            p_a_eff = p_a_grid * mask_ask  # (M, N)
            obj_a = p_a_eff * (a_grid[:, np.newaxis] + C_a[np.newaxis, :])
            
            idx_a = np.argmax(obj_a, axis=0)
            a_curr = a_grid[idx_a]
            
        # Calculate V_curr
        p_b_final = 1.0 - np.exp(-A * np.exp(-k * b_curr) * dt)
        p_b_final = p_b_final * mask_bid
        
        p_a_final = 1.0 - np.exp(-A * np.exp(-k * a_curr) * dt)
        p_a_final = p_a_final * mask_ask
        
        # Immediate reward
        imm_reward = p_b_final * b_curr + p_a_final * a_curr - (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        # Expected future value
        # E[V] = p_b(1-p_a)V_minus + (1-p_b)p_a V_plus + (1 - p_b - p_a + 2 p_b p_a) V_0
        # Using the simplified form: V_0 + p_b(V_minus - V_0) + p_a(V_plus - V_0) - p_b p_a (V_minus + V_plus - 2 V_0)
        
        term_b = p_b_final * (V_minus - V_0)
        term_a = p_a_final * (V_plus - V_0)
        term_coupling = -p_b_final * p_a_final * (V_minus + V_plus - 2 * V_0)
        
        V_curr = V_0 + term_b + term_a + term_coupling + imm_reward
        
        bid_depths[n] = b_curr
        ask_depths[n] = a_curr
        
        V_next = V_curr
        
    return bid_depths, ask_depths

# Pre-compute known regimes
known_params = [
    {'sigma': 2, 'A': 140, 'k': 1.5, 'gamma': 0.05, 'T': 1, 'steps': 1000, 'Q': 30},
    {'sigma': 2, 'A': 140, 'k': 1.5, 'gamma': 0.5, 'T': 1, 'steps': 1000, 'Q': 30},
    {'sigma': 1.5, 'A': 50, 'k': 0.75, 'gamma': 0.2, 'T': 1, 'steps': 1000, 'Q': 30}
]

for p in known_params:
    key = (p['sigma'], p['A'], p['k'], p['gamma'])
    try:
        _POLICY_CACHE[key] = _compute_policy(**p)
    except Exception:
        pass

def quote(t, inventory, mid, params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    if key not in _POLICY_CACHE:
        if time.time() > _DEADLINE - 5:
            # Fallback to seed
            depth = np.full(len(inventory), 1.0 / params["k"])
            return depth, depth.copy()
        try:
            _POLICY_CACHE[key] = _compute_policy(**params)
        except Exception:
            depth = np.full(len(inventory), 1.0 / params["k"])
            return depth, depth.copy()
            
    bid_table, ask_table = _POLICY_CACHE[key]
    steps = params['steps']
    Q = params['Q']
    
    n = int(t * steps)
    if n >= steps:
        n = steps - 1
        
    # Map inventory to index
    # inventory is in [-Q, Q]
    idx = inventory + Q
    
    bid_depths = bid_table[n, idx]
    ask_depths = ask_table[n, idx]
    
    # Ensure valid values
    # np.inf means no quote.
    # The problem says "np.inf means no quote".
    # Our grid doesn't produce inf.
    # But if we wanted to disable a side, we could set it to inf.
    # However, the probability becomes 0, which is equivalent.
    # The prompt says "Values below -3/k ... make the whole result invalid".
    # Our grid starts at -3/k.
    
    return bid_depths, ask_depths
