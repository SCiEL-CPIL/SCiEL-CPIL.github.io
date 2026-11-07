import numpy as np
import threading

# Global cache for precomputed strategies
_strategy_cache = {}
_cache_lock = threading.Lock()

def _compute_optimal_strategy(params):
    """
    Precompute the optimal bid and ask depths for all time steps and inventory levels.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Grid for depths
    # Max depth: when p is very small.
    # p = 1 - exp(-A * exp(-k*b) * dt)
    # If b=5, k=1.5, exp(-7.5) ~ 0.0005. A*dt ~ 0.14. p ~ 0.00007.
    # 5.0 is a safe upper bound.
    num_grid = 100
    b_grid = np.linspace(0.0, 5.0, num_grid)
    a_grid = np.linspace(0.0, 5.0, num_grid)
    
    # Precompute probabilities for grid points
    # p_b[i] = 1 - exp(-A * exp(-k * b_grid[i]) * dt)
    exp_term_b = np.exp(-k * b_grid)
    p_b_grid = 1.0 - np.exp(-A * exp_term_b * dt)
    
    exp_term_a = np.exp(-k * a_grid)
    p_a_grid = 1.0 - np.exp(-A * exp_term_a * dt)
    
    # Value function V[n][q]
    # q ranges from -Q to Q
    num_states = 2 * Q + 1
    V_next = np.zeros(num_states)
    
    # Store optimal depths
    opt_b = np.full((steps, num_states), np.inf)
    opt_a = np.full((steps, num_states), np.inf)
    
    # Index helper: q -> index
    def q_to_idx(q):
        return q + Q
        
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(num_states)
        
        for q in range(-Q, Q + 1):
            idx = q_to_idx(q)
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            if q == Q:
                # No bid fill. Optimize a only.
                # Objective: p_a * a + (1 - p_a) * V_next[idx] + p_a * V_next[idx - 1] - cost
                # V_next[idx] is V(q)
                # V_next[idx - 1] is V(q-1)
                v_q = V_next[idx]
                v_qm1 = V_next[idx - 1]
                
                # Vectorized over a_grid
                # p_a is p_a_grid
                # term1 = p_a * a_grid
                # term2 = (1 - p_a) * v_q
                # term3 = p_a * v_qm1
                # obj = term1 + term2 + term3 - cost
                obj_a = p_a_grid * a_grid + (1.0 - p_a_grid) * v_q + p_a_grid * v_qm1 - cost
                best_a_idx = np.argmax(obj_a)
                opt_a[n, idx] = a_grid[best_a_idx]
                opt_b[n, idx] = np.inf # No bid
                V_curr[idx] = obj_a[best_a_idx]
                
            elif q == -Q:
                # No ask fill. Optimize b only.
                v_q = V_next[idx]
                v_qp1 = V_next[idx + 1]
                
                obj_b = p_b_grid * b_grid + (1.0 - p_b_grid) * v_q + p_b_grid * v_qp1 - cost
                best_b_idx = np.argmax(obj_b)
                opt_b[n, idx] = b_grid[best_b_idx]
                opt_a[n, idx] = np.inf # No ask
                V_curr[idx] = obj_b[best_b_idx]
                
            else:
                # Interior state. Optimize b and a.
                v_q = V_next[idx]
                v_qp1 = V_next[idx + 1]
                v_qm1 = V_next[idx - 1]
                
                # Broadcast p_b and p_a to matrix
                # P_B shape (N, 1), P_A shape (1, M)
                P_B = p_b_grid[:, np.newaxis]
                P_A = p_a_grid[np.newaxis, :]
                B = b_grid[:, np.newaxis]
                A_grid = a_grid[np.newaxis, :]
                
                # Objective Matrix
                # Reward: P_B * B + P_A * A_grid
                # Continuation:
                # (1 - P_B * P_A) * v_q
                # P_B * (1 - P_A) * v_qp1
                # P_A * (1 - P_B) * v_qm1
                
                # Note: P_B * P_A is element-wise product of broadcasted arrays
                term_reward = P_B * B + P_A * A_grid
                term_cont_main = (1.0 - P_B * P_A) * v_q
                term_cont_bid = P_B * (1.0 - P_A) * v_qp1
                term_cont_ask = P_A * (1.0 - P_B) * v_qm1
                
                obj_matrix = term_reward + term_cont_main + term_cont_bid + term_cont_ask - cost
                
                best_idx = np.unravel_index(np.argmax(obj_matrix), obj_matrix.shape)
                opt_b[n, idx] = b_grid[best_idx[0]]
                opt_a[n, idx] = a_grid[best_idx[1]]
                V_curr[idx] = obj_matrix[best_idx]
                
        V_next = V_curr
        
    return opt_b, opt_a

def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths for the given state.
    """
    # Create a hashable key for the cache
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    with _cache_lock:
        if key not in _strategy_cache:
            _strategy_cache[key] = _compute_optimal_strategy(params)
        opt_b, opt_a = _strategy_cache[key]
    
    dt = params['T'] / params['steps']
    steps = params['steps']
    Q = params['Q']
    
    # Determine current step n
    # t is n * dt
    n = int(round(t / dt))
    n = np.clip(n, 0, steps - 1)
    
    # Map inventory to index
    indices = inventory + Q
    
    # Get depths
    bid_depths = opt_b[n, indices]
    ask_depths = opt_a[n, indices]
    
    return bid_depths, ask_depths
