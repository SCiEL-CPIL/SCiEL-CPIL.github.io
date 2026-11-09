import numpy as np
import time

# Cache for precomputed tables
_TABLE_CACHE = {}
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def _check_deadline():
    return time.time() >= _DEADLINE

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    N = steps
    n_states = 2 * Q + 1
    max_q = Q
    min_q = -Q
    
    # Precompute depth grid
    # Max depth where probability is non-negligible
    # p = 1 - exp(-A exp(-k b) dt)
    # If b is large, p ~ 0.
    # Let's use a grid up to 20/k to be safe, or dynamic.
    # For k=1.5, 20/1.5 = 13.3.
    # For k=0.75, 20/0.75 = 26.6.
    # Let's use a fixed fine grid to ensure vectorization consistency.
    # 256 points is fast.
    max_depth = 30.0 / k 
    depth_grid = np.linspace(0, max_depth, 256)
    
    # Precompute p_b for the grid
    p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    opt_bid = np.zeros((N, n_states))
    opt_ask = np.zeros((N, n_states))
    
    V_next = np.zeros(n_states)
    
    # Indices for q
    q_vals = np.arange(-Q, Q + 1)
    
    for n in range(N - 1, -1, -1):
        if _check_deadline():
            # Fill remaining with default
            depth_default = 1.0 / k
            for n_fill in range(n, N):
                opt_bid[n_fill, :] = depth_default
                opt_ask[n_fill, :] = depth_default
            return opt_bid, opt_ask
        
        V_curr = np.zeros(n_states)
        
        # Initialize depths with previous or default
        # We will do fixed point iterations
        # Start with 0 or previous step's value
        if n == N - 1:
            b_curr = np.zeros(n_states)
            a_curr = np.zeros(n_states)
        else:
            b_curr = opt_bid[n+1, :]
            a_curr = opt_ask[n+1, :]
            
        # Fixed point iterations
        # We need to solve for b given a, then a given b.
        # Vectorized update for b:
        # J_b(b) = p_b(b) * (b + A_q - B_q) + B_q - risk
        # A_q = (1-p_a) V_{q+1} - p_a V_{q-1} - (1-p_a) V_q
        # B_q = p_a V_{q-1} + (1-p_a) V_q
        
        # We need V_{q+1}, V_q, V_{q-1}
        # q_idx is 0..2Q
        # q_idx+1 is valid for q < Q
        # q_idx-1 is valid for q > -Q
        
        # Prepare V shifts
        V_q = V_next
        V_q_plus = np.roll(V_next, -1)
        V_q_minus = np.roll(V_next, 1)
        
        # Handle boundaries
        # For q=Q, no bid fill possible, so b is irrelevant for V update?
        # No, if q=Q, we cannot have bid fill. So p_b is effectively 0 in the transition?
        # The model says "no bid fill at q=Q".
        # So for q=Q, the term involving p_b should be 0.
        # Similarly for q=-Q, p_a = 0.
        
        # Let's create masks
        mask_bid_active = q_vals < max_q
        mask_ask_active = q_vals > min_q
        
        for _iter in range(10):
            # Update b
            # Calculate A_q and B_q based on current a_curr
            p_a_curr = 1.0 - np.exp(-A * np.exp(-k * a_curr) * dt)
            
            # A_q = (1-p_a) V_{q+1} - p_a V_{q-1} - (1-p_a) V_q
            A_q = (1.0 - p_a_curr) * V_q_plus - p_a_curr * V_q_minus - (1.0 - p_a_curr) * V_q
            # B_q = p_a V_{q-1} + (1-p_a) V_q
            B_q = p_a_curr * V_q_minus + (1.0 - p_a_curr) * V_q
            
            # Risk term is constant w.r.t b
            risk = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
            
            # Objective for each b in grid and each q
            # Shape: (256, n_states)
            # p_grid shape (256,)
            # depth_grid shape (256,)
            # A_q, B_q shape (n_states,)
            
            # J = p_grid[:, None] * (depth_grid[:, None] + A_q[None, :] - B_q[None, :]) + B_q[None, :] - risk[None, :]
            
            # Optimization:
            # J = p_grid[:, None] * depth_grid[:, None] + p_grid[:, None] * (A_q[None, :] - B_q[None, :]) + B_q[None, :] - risk[None, :]
            
            # We can compute this efficiently.
            term1 = p_grid[:, None] * depth_grid[:, None]
            term2 = p_grid[:, None] * (A_q[None, :] - B_q[None, :])
            term3 = B_q[None, :] - risk[None, :]
            
            J_b = term1 + term2 + term3
            
            # Mask out inactive bids
            # If mask_bid_active is False, J_b should be -inf so it's not chosen, 
            # or we just set b=0 for those.
            # Actually, if q=Q, we don't quote bid. So b should be 0 or inf?
            # "np.inf means no quote". But the return type is float arrays.
            # The task says "distances below... to quote".
            # If no quote, should it be 0 or inf?
            # "np.inf means no quote".
            # However, the objective function in DP assumes we pick a b.
            # If we are at Q, we cannot fill. So the choice of b doesn't affect V.
            # But we should return 0 or inf?
            # Usually, if you can't quote, you don't.
            # Let's check the seed program. It sets b_new=0.0 if q>=max_q.
            # So we will set b=0 for inactive.
            
            # To enforce this, we can set J_b[:, ~mask_bid_active] = -np.inf
            # But we need to be careful with broadcasting.
            J_b[:, ~mask_bid_active] = -np.inf
            
            # Find best b
            best_idx_b = np.argmax(J_b, axis=0)
            b_new = depth_grid[best_idx_b]
            
            # Update a
            p_b_new = 1.0 - np.exp(-A * np.exp(-k * b_new) * dt)
            
            # A'_q for ask optimization
            # J_a(a) = p_a(a) * a + p_a(a) C'_q + (1-p_a(a)) D'_q - risk
            # C'_q = (1-p_b) V_{q-1} - p_b V_{q+1} - (1-p_b) V_q  (Symmetric)
            # D'_q = p_b V_{q+1} + (1-p_b) V_q
            
            C_a = (1.0 - p_b_new) * V_q_minus - p_b_new * V_q_plus - (1.0 - p_b_new) * V_q
            D_a = p_b_new * V_q_plus + (1.0 - p_b_new) * V_q
            
            term1_a = p_grid[:, None] * depth_grid[:, None]
            term2_a = p_grid[:, None] * (C_a[None, :] - D_a[None, :])
            term3_a = D_a[None, :] - risk[None, :]
            
            J_a = term1_a + term2_a + term3_a
            
            J_a[:, ~mask_ask_active] = -np.inf
            
            best_idx_a = np.argmax(J_a, axis=0)
            a_new = depth_grid[best_idx_a]
            
            # Check convergence
            if np.allclose(b_new, b_curr, atol=1e-4) and np.allclose(a_new, a_curr, atol=1e-4):
                b_curr = b_new
                a_curr = a_new
                break
            b_curr = b_new
            a_curr = a_new
        
        opt_bid[n, :] = b_curr
        opt_ask[n, :] = a_curr
        
        # Calculate V_curr
        p_b_final = 1.0 - np.exp(-A * np.exp(-k * b_curr) * dt)
        p_a_final = 1.0 - np.exp(-A * np.exp(-k * a_curr) * dt)
        
        # For inactive, set p to 0
        p_b_final[~mask_bid_active] = 0.0
        p_a_final[~mask_ask_active] = 0.0
        
        # Expected value
        # prob_q_plus = p_b (1-p_a)
        # prob_q_minus = (1-p_b) p_a
        # prob_q = 1 - ...
        
        prob_q_plus = p_b_final * (1.0 - p_a_final)
        prob_q_minus = (1.0 - p_b_final) * p_a_final
        prob_q = 1.0 - prob_q_plus - prob_q_minus
        
        # Reward
        reward = p_b_final * b_curr + p_a_final * a_curr
        
        # V_next is already shifted correctly?
        # V_next[q_idx] corresponds to q.
        # V_next[q_idx+1] is V(q+1)
        # We used V_q_plus = roll(V_next, -1) which is V(q+1) for index q.
        # So:
        exp_val = prob_q_plus * V_q_plus + prob_q_minus * V_q_minus + prob_q * V_q
        
        # Risk
        risk_curr = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        V_curr = reward - risk_curr + exp_val
        
        # Handle boundaries for V update?
        # If q=Q, prob_q_plus should be 0.
        # Our p_b_final is 0 for q=Q, so prob_q_plus is 0. Correct.
        # If q=-Q, p_a_final is 0, so prob_q_minus is 0. Correct.
        
        V_next = V_curr
        
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    
    if key not in _TABLE_CACHE:
        opt_bid, opt_ask = _precompute_regime(
            params['sigma'], params['A'], params['k'], params['gamma'], 
            params['T'], params['steps'], params['Q']
        )
        _TABLE_CACHE[key] = (opt_bid, opt_ask)
    else:
        opt_bid, opt_ask = _TABLE_CACHE[key]
    
    q_indices = np.asarray(inventory) + params['Q']
    
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    
    return bid_depths, ask_depths
