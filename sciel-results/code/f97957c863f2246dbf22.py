import numpy as np
import time

# Global cache for DP solutions
_dp_cache = {}
_start_time = time.time()
TIME_LIMIT = 300.0 # Safety margin before 330s kill

def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Solves the Bellman equation for the given parameters.
    Returns a dictionary containing optimal bid and ask depths for each step and inventory.
    """
    dt = T / steps
    
    # Grid for depths
    # Use log-spaced grid for better resolution at small depths (high fill prob)
    # b in [0, b_max]. Log space doesn't include 0. 
    # We can use np.geomspace if b_min > 0. 
    # Let's use a hybrid: include 0, then log-spaced from eps to b_max.
    b_max = 3.0 / k  # Slightly larger range to be safe
    eps = 1e-5
    # 80 points as suggested by knowledge
    n_grid = 80
    # Create log-spaced grid from eps to b_max
    b_log = np.geomspace(eps, b_max, n_grid - 1)
    b_grid = np.concatenate(([0.0], b_log))
    # Ensure sorted
    b_grid = np.sort(b_grid)
    
    a_grid = b_grid.copy()
    
    # Precompute probabilities and rewards for the grid
    # p_b = 1 - exp(-A * exp(-k * b) * dt)
    # Handle b=0: exp(0)=1.
    p_b = 1.0 - np.exp(-A * np.exp(-k * b_grid) * dt)
    p_a = 1.0 - np.exp(-A * np.exp(-k * a_grid) * dt)
    
    # Expected immediate reward components
    R_b = p_b * b_grid
    R_a = p_a * a_grid
    
    # Arrays to store optimal depths
    # shape: (steps, 2*Q+1)
    opt_b = np.zeros((steps, 2*Q + 1))
    opt_a = np.zeros((steps, 2*Q + 1))
    
    # V_n(q)
    V_next = np.zeros(2*Q + 1)
    
    # Inventory indices: 0 corresponds to q = -Q, ..., 2Q corresponds to q = Q
    # q = i - Q
    
    q_vals = np.arange(-Q, Q + 1)
    
    for n in range(steps - 1, -1, -1):
        # Check time
        if time.time() - _start_time > TIME_LIMIT - 10:
             # Return current best guess (zeros/inf) to avoid timeout
             # This is a fallback. Ideally we finish.
             # Fill remaining steps with 0/inf
             for n_rem in range(n, -1, -1):
                 opt_b[n_rem, :] = 0.0
                 opt_a[n_rem, :] = 0.0
                 opt_b[n_rem, 2*Q] = np.inf
                 opt_a[n_rem, 0] = np.inf
             return {
                 'opt_b': opt_b,
                 'opt_a': opt_a,
                 'Q': Q
             }
        
        # Compute deltas for all q
        V_0 = V_next
        V_plus = np.zeros_like(V_next)
        V_minus = np.zeros_like(V_next)
        
        # q+1 exists if q < Q, i.e., i < 2Q
        V_plus[:2*Q] = V_next[1:]
        # q-1 exists if q > -Q, i.e., i > 0
        V_minus[1:] = V_next[:-1]
        
        Delta_plus = V_plus - V_0
        Delta_minus = V_minus - V_0
        Delta_curv = V_plus + V_minus - 2 * V_0
        
        # Construct Objective Matrix (Nb, Na, Nq)
        # Using broadcasting to avoid explicit meshgrids for memory efficiency
        Nb = len(b_grid)
        Na = len(a_grid)
        Nq = 2*Q + 1
        
        # R_b (Nb, 1, 1) + R_a (1, Na, 1)
        Obj = R_b[:, np.newaxis, np.newaxis] + R_a[np.newaxis, :, np.newaxis]
        
        # Penalty (1, 1, Nq)
        penalty = -0.5 * gamma * sigma**2 * q_vals**2 * dt
        Obj += penalty[np.newaxis, np.newaxis, :]
        
        # V_0 (1, 1, Nq)
        Obj += V_0[np.newaxis, np.newaxis, :]
        
        # p_b (Nb, 1, 1) * Delta_plus (1, 1, Nq)
        Obj += p_b[:, np.newaxis, np.newaxis] * Delta_plus[np.newaxis, np.newaxis, :]
        
        # p_a (1, Na, 1) * Delta_minus (1, 1, Nq)
        Obj += p_a[np.newaxis, :, np.newaxis] * Delta_minus[np.newaxis, np.newaxis, :]
        
        # - p_b (Nb, 1, 1) * p_a (1, Na, 1) * Delta_curv (1, 1, Nq)
        # This results in (Nb, Na, Nq)
        Obj -= (p_b[:, np.newaxis, np.newaxis] * p_a[np.newaxis, :, np.newaxis]) * Delta_curv[np.newaxis, np.newaxis, :]
        
        # Handle boundaries
        # For q=Q (index 2Q), bid fill is impossible.
        # We want to force p_b to be effectively 0 or ignore the bid term.
        # The term "no bid fill at q=Q" means the transition to Q+1 doesn't happen.
        # In the model, if we quote bid, and a fill *would* occur, it is blocked.
        # Does the inventory change? No.
        # Do we get the reward? "A fill earns its depth". If no fill, no reward.
        # So if the event is blocked, it's treated as no fill.
        # Thus, for q=Q, the bid side should behave as if p_b=0.
        # This implies R_b=0, transition term 0, curvature term 0.
        # We can achieve this by setting the columns of Obj corresponding to q=Q 
        # to only depend on a. 
        # Easier: After argmax, we can overwrite. 
        # But the argmax might pick a small b (high prob) because the penalty/transition terms are wrong if we don't mask.
        # Let's mask the objective for boundary states.
        
        # For q=Q (last index), set Obj[:, :, -1] to maximize over a only.
        # The optimal b should be such that p_b=0 (b=inf). 
        # Since we can't quote inf, we should pick the largest b in grid? 
        # Or just return inf from quote().
        # The DP value V_curr(Q) should be max_a [ R_a + ... ].
        # The b dimension should not affect it.
        # So we can set Obj[:, :, -1] = Obj[0, :, -1] (any b, result same if we ignore b terms).
        # To do this correctly, we should zero out the b-dependent terms for q=Q.
        
        # Let's create a mask for valid bid/ask
        # Actually, simpler: 
        # For q=Q, the "effective" p_b is 0.
        # We can manually calculate the objective for q=Q and q=-Q separately?
        # Or modify Obj.
        
        # Let's modify Obj for boundaries before argmax.
        # q=Q is index -1. q=-Q is index 0.
        
        # For q=Q:
        # Terms involving p_b should be 0.
        # Obj[:, :, -1] currently has:
        # R_b[i] + R_a[j] + penalty + V_0 + p_b[i]*Delta_plus + p_a[j]*Delta_minus - p_b[i]*p_a[j]*Delta_curv
        # We want:
        # 0 + R_a[j] + penalty + V_0 + 0 + p_a[j]*Delta_minus - 0
        # So we need to subtract R_b[i], p_b[i]*Delta_plus, and add back p_b[i]*p_a[j]*Delta_curv?
        # This is messy.
        
        # Alternative: Set p_b = 0 for the last column calculation.
        # We can just overwrite the slice.
        
        # Calculate "pure ask" objective for q=Q
        # R_a (1, Na, 1) + penalty (1, 1, 1) + V_0 (1, 1, 1) + p_a (1, Na, 1)*Delta_minus (1, 1, 1)
        # Result shape (1, Na, 1). Broadcast to (Nb, Na, 1)?
        # We want the max over a. The b index doesn't matter.
        # So we can set Obj[:, :, -1] to a constant value with respect to i (b-index).
        
        # Let's compute the correct objective for q=Q and q=-Q.
        
        # q=Q (index -1):
        # Delta_plus is invalid (should be 0 contribution).
        # Delta_minus is valid.
        # Delta_curv is invalid (should be 0 contribution).
        
        # Correct Obj_Q = R_a + penalty_Q + V_0_Q + p_a * Delta_minus_Q
        # R_a shape (Na,) -> (1, Na, 1)
        # penalty_Q scalar
        # V_0_Q scalar
        # Delta_minus_Q scalar
        # p_a shape (Na,) -> (1, Na, 1)
        
        Obj_Q = R_a[np.newaxis, :, np.newaxis] + penalty[-1][np.newaxis, np.newaxis, np.newaxis] + V_0[-1][np.newaxis, np.newaxis, np.newaxis] + p_a[np.newaxis, :, np.newaxis] * Delta_minus[-1][np.newaxis, np.newaxis, np.newaxis]
        
        # Broadcast to (Nb, Na, 1)
        Obj[:, :, -1] = Obj_Q[:, :, np.newaxis] # Wait, Obj_Q is (1, Na, 1). 
        # Actually Obj_Q is (1, Na, 1). 
        # We need to fill (Nb, Na, 1).
        # Since it doesn't depend on i, we can just assign the slice.
        # Numpy broadcasting: assigning (1, Na, 1) to (Nb, Na, 1) works?
        # Yes.
        
        # q=-Q (index 0):
        # Correct Obj_-Q = R_b + penalty_-Q + V_0_-Q + p_b * Delta_plus_-Q
        Obj_minus_Q = R_b[:, np.newaxis, np.newaxis] + penalty[0][np.newaxis, np.newaxis, np.newaxis] + V_0[0][np.newaxis, np.newaxis, np.newaxis] + p_b[:, np.newaxis, np.newaxis] * Delta_plus[0][np.newaxis, np.newaxis, np.newaxis]
        Obj[:, :, 0] = Obj_minus_Q[:, :, np.newaxis]
        
        # Find max
        max_idx = np.argmax(Obj, axis=(0, 1))
        # max_idx shape (Nq,)
        b_idx = max_idx // Na
        a_idx = max_idx % Na
        
        # Extract optimal depths
        # Vectorized assignment
        opt_b[n, :] = b_grid[b_idx]
        opt_a[n, :] = a_grid[a_idx]
        
        # Enforce boundary conditions (safety)
        opt_b[n, 2*Q] = np.inf
        opt_a[n, 0] = np.inf
        
        V_next = np.max(Obj, axis=(0, 1))
        
    return {
        'opt_b': opt_b,
        'opt_a': opt_a,
        'Q': Q
    }

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _dp_cache:
        _dp_cache[key] = _solve_dp(sigma, A, k, gamma, T, steps, Q)
        
    solution = _dp_cache[key]
    opt_b = solution['opt_b']
    opt_a = solution['opt_a']
    
    # Determine current step
    dt = T / steps
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
        
    # Get depths for current inventory
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2*Q)
    
    bid_depths = opt_b[n, q_idx]
    ask_depths = opt_a[n, q_idx]
    
    return bid_depths, ask_depths
