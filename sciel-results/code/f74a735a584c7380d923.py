import numpy as np
import time

# Cache for DP tables
_dp_cache = {}

def _compute_dp_tables(params):
    """
    Computes the optimal bid and ask depths for all time steps and inventory levels.
    Returns a tuple (bid_depths, ask_depths) where each is a 2D array of shape (steps, 2*Q+1).
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    lambda_b = A * dt
    lambda_a = A * dt  # Symmetric
    
    # Inventory indices: 0..2Q corresponds to -Q..Q
    n_states = 2 * Q + 1
    
    # Initialize value function at T
    V_next = np.zeros(n_states)
    
    # Pre-allocate storage for optimal depths
    # We store the depths, not probabilities
    bid_depths = np.zeros((steps, n_states))
    ask_depths = np.zeros((steps, n_states))
    
    # Helper for log(1-p) stability
    eps = 1e-6
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_states)
        
        # Inventory array for current step
        q_vals = np.arange(-Q, Q + 1)
        
        # Get values for q, q+1, q-1
        # V_q
        V_q = V_next.copy()
        # V_q1 (q+1)
        V_q1 = np.zeros(n_states)
        V_q1[:-1] = V_next[1:]
        V_q1[-1] = V_next[-1] # Boundary: q=Q, q+1 is invalid, set to V_q or 0? 
                               # In the expectation, if q=Q, the term with V_q1 is multiplied by p_b which we will force to 0 or ignore.
                               # To be safe, set V_q1[-1] = V_next[-1] so it doesn't cause issues if mask fails.
        
        # V_qm1 (q-1)
        V_qm1 = np.zeros(n_states)
        V_qm1[1:] = V_next[:-1]
        V_qm1[0] = V_next[0] # Boundary: q=-Q, q-1 is invalid.
        
        # Initialize probabilities
        P_b = np.full(n_states, 0.5)
        P_a = np.full(n_states, 0.5)
        
        # Masks
        mask_q_max = np.zeros(n_states, dtype=bool)
        mask_q_max[-1] = True # q = Q
        
        mask_q_min = np.zeros(n_states, dtype=bool)
        mask_q_min[0] = True # q = -Q
        
        mask_interior = ~(mask_q_max | mask_q_min)
        
        # Gradient Ascent Optimization
        # We optimize over P_b and P_a
        # Step size
        alpha = 0.05
        max_iter = 50
        
        for _ in range(max_iter):
            # Compute gradients for all states
            # Term: -1/(k*lambda) * (ln(1-P) + P/(1-P))
            # Note: P is in (0, 1)
            
            # Safe log
            log_1_minus_Pb = np.log(np.maximum(1 - P_b, eps))
            log_1_minus_Pa = np.log(np.maximum(1 - P_a, eps))
            
            term_b_cost = (log_1_minus_Pb + P_b / np.maximum(1 - P_b, eps)) / (k * lambda_b)
            term_a_cost = (log_1_minus_Pa + P_a / np.maximum(1 - P_a, eps)) / (k * lambda_a)
            
            # Gradient w.r.t P_b
            g_b = V_q * (-1 + 2 * P_a) + V_q1 * (1 - P_a) - V_qm1 * P_a - term_b_cost
            # Gradient w.r.t P_a
            g_a = V_q * (-1 + 2 * P_b) + V_qm1 * (1 - P_b) - V_q1 * P_b - term_a_cost
            
            # Apply masks:
            # At q=Q, P_b is fixed at 0 (no bid). So g_b should be 0, and we don't update P_b.
            # At q=-Q, P_a is fixed at 0 (no ask). So g_a should be 0, and we don't update P_a.
            
            g_b[mask_q_max] = 0.0
            g_a[mask_q_min] = 0.0
            
            # Update
            P_b += alpha * g_b
            P_a += alpha * g_a
            
            # Clip
            P_b = np.clip(P_b, eps, 1 - eps)
            P_a = np.clip(P_a, eps, 1 - eps)
            
            # Force boundaries
            P_b[mask_q_max] = 0.0
            P_a[mask_q_min] = 0.0
            # For interior, P_b and P_a are free.
            # For q=Q, P_a is free.
            # For q=-Q, P_b is free.
            
            # Check convergence (optional, fixed iterations usually fine)
            # if np.max(np.abs(g_b[mask_interior])) < 1e-4 and np.max(np.abs(g_a[mask_interior])) < 1e-4:
            #     break

        # Compute optimal depths
        # b = -1/(k*lambda_b) * ln(1 - P_b)
        # a = -1/(k*lambda_a) * ln(1 - P_a)
        
        # Handle P=0 case (boundaries)
        # If P_b=0, b is undefined/inf? No, if P_b=0, we don't quote? 
        # The task says "np.inf means no quote".
        # If P_b=0, the optimal depth is effectively infinite (no quote) or 0?
        # If we don't quote, we don't get a fill.
        # If P_b=0 is optimal, it means we don't want to quote.
        # So we should return np.inf.
        
        # However, for q=Q, we *must* not quote a bid. So return np.inf.
        # For interior, if P_b is very small, b is large.
        
        bid_depths[n, :] = -np.log(np.maximum(1 - P_b, eps)) / (k * lambda_b)
        ask_depths[n, :] = -np.log(np.maximum(1 - P_a, eps)) / (k * lambda_a)
        
        # Set boundaries to inf
        bid_depths[n, mask_q_max] = np.inf
        ask_depths[n, mask_q_min] = np.inf
        
        # Compute V_curr
        # J = V_q (1 - P_b - P_a + 2 P_b P_a) + V_q1 P_b (1-P_a) + V_qm1 P_a (1-P_b) 
        #   + b P_b + a P_a
        #   - (gamma/2) sigma^2 q^2 dt
        
        # Note: b and a are the depths.
        # b = -1/(k*lambda_b) ln(1-P_b)
        
        b_vals = bid_depths[n, :]
        a_vals = ask_depths[n, :]
        
        # Replace inf with 0 for calculation (since P=0, term is 0)
        b_calc = np.where(np.isinf(b_vals), 0, b_vals)
        a_calc = np.where(np.isinf(a_vals), 0, a_vals)
        
        # Ensure P=0 where inf
        P_b_calc = P_b.copy()
        P_a_calc = P_a.copy()
        P_b_calc[np.isinf(b_vals)] = 0
        P_a_calc[np.isinf(a_vals)] = 0
        
        # Inventory penalty
        q_sq = q_vals ** 2
        penalty = 0.5 * gamma * sigma**2 * q_sq * dt
        
        V_curr = V_q * (1 - P_b_calc - P_a_calc + 2 * P_b_calc * P_a_calc) + \
                 V_q1 * P_b_calc * (1 - P_a_calc) + \
                 V_qm1 * P_a_calc * (1 - P_b_calc) + \
                 b_calc * P_b_calc + \
                 a_calc * P_a_calc - penalty
        
        V_next = V_curr
        
    return bid_depths, ask_depths

def quote(t, inventory, mid, params):
    # Cache key
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _dp_cache:
        # Compute and cache
        bid_depths, ask_depths = _compute_dp_tables(params)
        _dp_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _dp_cache[key]
    
    steps = params['steps']
    Q = params['Q']
    dt = params['T'] / steps
    
    # Determine step index
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    # Inventory to index
    # inventory is an array of ints in [-Q, Q]
    # We need to return an array of depths for each path
    # The DP table is indexed by [n, q_index]
    # inventory[i] is the inventory for path i
    # q_index = inventory[i] + Q
    
    q_indices = inventory + Q
    
    # Extract depths
    # bid_depths[n, q_indices] gives the optimal bid depth for each path
    bid_out = bid_depths[n, q_indices]
    ask_out = ask_depths[n, q_indices]
    
    # Ensure output types are float
    return bid_out.astype(float), ask_out.astype(float)
