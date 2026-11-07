import numpy as np
import time

# Global cache for DP solutions
_dp_cache = {}

def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Solves the Bellman equation for the given parameters.
    Returns a dictionary containing optimal bid and ask depths for each step and inventory.
    """
    dt = T / steps
    # Grid for depths
    # We use a grid of 'lambda' values which are proportional to fill probability
    # b = -1/k * ln(lambda / (A*dt))
    # lambda range: 
    # min lambda corresponds to max depth
    # max lambda corresponds to min depth (or negative depth)
    
    # Let's define grid in terms of b directly for simplicity, or lambda.
    # b range: [0, 2.0/k] is a safe bet. 
    # If k is small, 2/k is large.
    # Let's use log-spaced grid for b? No, linear is fine for small range.
    # Let's try 200 points for better precision.
    b_min = 0.0
    b_max = 2.0 / k
    b_grid = np.linspace(b_min, b_max, 200)
    a_grid = b_grid.copy()
    
    # Precompute probabilities and rewards for the grid
    # p_b = 1 - exp(-A * exp(-k * b) * dt)
    p_b = 1.0 - np.exp(-A * np.exp(-k * b_grid) * dt)
    p_a = 1.0 - np.exp(-A * np.exp(-k * a_grid) * dt)
    
    # Expected immediate reward components
    # R_b = p_b * b
    # R_a = p_a * a
    R_b = p_b * b_grid
    R_a = p_a * a_grid
    
    # Create meshgrid for b and a
    B_grid, A_grid = np.meshgrid(b_grid, a_grid, indexing='ij')
    P_b_grid, P_a_grid = np.meshgrid(p_b, p_a, indexing='ij')
    R_b_grid, R_a_grid = np.meshgrid(R_b, R_a, indexing='ij')
    P_ba_grid = P_b_grid * P_a_grid
    
    # Arrays to store optimal depths
    # shape: (steps, 2*Q+1)
    opt_b = np.zeros((steps, 2*Q + 1))
    opt_a = np.zeros((steps, 2*Q + 1))
    
    # V_n(q)
    V_next = np.zeros(2*Q + 1)
    
    # Inventory indices: 0 corresponds to q = -Q, ..., 2Q corresponds to q = Q
    # q = i - Q
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(2*Q + 1)
        
        # Compute deltas for all q
        # q index i
        # V_next[i] is V_{n+1}(q)
        # V_+ = V_{n+1}(q+1) -> V_next[i+1] if i < 2Q
        # V_- = V_{n+1}(q-1) -> V_next[i-1] if i > 0
        # V_0 = V_next[i]
        
        V_0 = V_next
        V_plus = np.zeros_like(V_next)
        V_minus = np.zeros_like(V_next)
        
        # q+1 exists if q < Q, i.e., i < 2Q
        V_plus[:2*Q] = V_next[1:]
        # q-1 exists if q > -Q, i.e., i > 0
        V_minus[1:] = V_next[:-1]
        
        # If boundary, V_plus or V_minus should be V_0?
        # "no bid fill at q=Q" -> prob 0.
        # In our DP, if q=Q, we can set p_b=0 effectively.
        # But we are maximizing over b. If we choose b, p_b > 0.
        # However, the transition to Q+1 is forbidden.
        # So for q=Q, the term involving V_plus should be 0.
        # Similarly for q=-Q, term involving V_minus should be 0.
        
        # Let's modify the objective calculation to handle boundaries.
        # Objective J(b,a,q) = R_b + R_a - penalty + V_0 + p_b(Delta_+) + p_a(Delta_-) - p_b p_a Delta_curv
        # Delta_+ = V_plus - V_0
        # Delta_- = V_minus - V_0
        # Delta_curv = V_plus + V_minus - 2 V_0
        
        # For q=Q (i=2Q): V_plus is invalid. Set Delta_+ = 0?
        # Actually, if q=Q, a bid fill doesn't change inventory? Or is rejected?
        # "no bid fill at q=Q" implies the event "bid fill" leading to Q+1 has prob 0.
        # So the contribution to expected value from the bid fill path is 0.
        # This is equivalent to setting p_b = 0 in the transition part, but we still pay for the quote? No, if no fill, no reward.
        # If we quote and it doesn't fill, we get 0 reward.
        # If we quote and it "would" fill but is blocked, we get 0 reward and stay at Q.
        # So effectively, for q=Q, the bid side is "dead".
        # We can just set the optimal b to inf (or large) so p_b ~ 0.
        # Or we can mask the grid for q=Q to only consider b that result in p_b=0?
        # Easier: In the objective function, for q=Q, set Delta_+ = 0 and R_b = 0?
        # No, if we quote b, and it doesn't fill, we get 0.
        # If we quote b, and the model says "no fill at Q", then p_b_effective = 0.
        # So R_b_effective = 0.
        # So for q=Q, we should maximize over a only, and set b to something that yields 0 prob (or just ignore b).
        # Since we return depths, we can return inf for b if q=Q.
        
        # Let's compute the objective matrix for all q.
        # Shape: (num_b, num_a, num_q)
        
        # Delta_+ shape (num_q)
        Delta_plus = V_plus - V_0
        Delta_minus = V_minus - V_0
        Delta_curv = V_plus + V_minus - 2 * V_0
        
        # Mask for boundaries
        # q=Q is index 2Q. q=-Q is index 0.
        mask_bid_dead = np.zeros(2*Q + 1, dtype=bool)
        mask_bid_dead[2*Q] = True
        mask_ask_dead = np.zeros(2*Q + 1, dtype=bool)
        mask_ask_dead[0] = True
        
        # For q where bid is dead, we want to force p_b to be 0.
        # We can achieve this by setting the objective to -inf for all b except the one with lowest p_b?
        # Or simply, for those q, we only optimize over a.
        
        # Vectorized approach:
        # R_b_grid shape (Nb, Na) -> broadcast to (Nb, Na, Nq)
        # This is memory heavy if Nb, Na are large. 200*200*61 = 2.4M floats. OK.
        
        Obj = np.zeros((len(b_grid), len(a_grid), 2*Q + 1))
        
        # Add R_b + R_a
        # R_b depends on b index, R_a on a index.
        # We can add them separately.
        Obj += R_b_grid[:, :, np.newaxis] + R_a_grid[:, :, np.newaxis]
        
        # Add penalty (depends on q)
        q_vals = np.arange(-Q, Q + 1)
        penalty = -0.5 * gamma * sigma**2 * q_vals**2 * dt
        Obj += penalty[np.newaxis, np.newaxis, :]
        
        # Add V_0
        Obj += V_0[np.newaxis, np.newaxis, :]
        
        # Add p_b * Delta_+
        # p_b shape (Nb, 1, 1) ? No, p_b_grid is (Nb, Na).
        # We need to multiply p_b (which depends on b) with Delta_plus (depends on q).
        # Term: p_b[i, j] * Delta_plus[k]
        # We can reshape p_b to (Nb, 1, 1) and Delta_plus to (1, 1, Nq)?
        # No, p_b_grid is (Nb, Na). The value p_b[i, j] is just p_b[i].
        # So we can use p_b[:, np.newaxis, np.newaxis] * Delta_plus[np.newaxis, np.newaxis, :]
        # This creates a (Nb, 1, Nq) array, which broadcasts to (Nb, Na, Nq) when added?
        # Wait, Obj is (Nb, Na, Nq).
        # p_b[:, np.newaxis, np.newaxis] is (Nb, 1, 1).
        # Delta_plus[np.newaxis, np.newaxis, :] is (1, 1, Nq).
        # Product is (Nb, 1, Nq).
        # Adding this to Obj (Nb, Na, Nq) works via broadcasting?
        # Yes, (Nb, 1, Nq) broadcasts to (Nb, Na, Nq).
        
        Obj += p_b[:, np.newaxis, np.newaxis] * Delta_plus[np.newaxis, np.newaxis, :]
        Obj += p_a[np.newaxis, :, np.newaxis] * Delta_minus[np.newaxis, np.newaxis, :]
        
        # Add - p_b p_a Delta_curv
        # p_b p_a shape (Nb, Na).
        # Delta_curv shape (Nq).
        # Product shape (Nb, Na, 1) * (1, 1, Nq) -> (Nb, Na, Nq).
        Obj -= (p_b_grid * p_a_grid)[:, :, np.newaxis] * Delta_curv[np.newaxis, np.newaxis, :]
        
        # Handle boundaries
        # For q=Q (index 2Q), bid fill is impossible.
        # This means the term p_b * (V_plus - V_0) should be 0, and R_b should be 0.
        # And the term -p_b p_a Delta_curv should be 0?
        # If bid doesn't fill, p_b_eff = 0.
        # So R_b_eff = 0.
        # Transition: only ask can fill.
        # So effectively, for q=Q, we should set p_b = 0 in the objective.
        # We can do this by setting the corresponding column of Obj to -inf for all b > 0?
        # Or simply, for q=Q, we know optimal b is such that p_b=0 (i.e., b=inf).
        # We can just overwrite the result for q=Q after argmax.
        
        # Similarly for q=-Q, optimal a is inf.
        
        # Find max
        max_idx = np.argmax(Obj, axis=(0, 1))
        # max_idx shape (Nq,)
        # Convert to b_idx, a_idx
        b_idx = max_idx // len(a_grid)
        a_idx = max_idx % len(a_grid)
        
        # Extract optimal depths
        for q_idx in range(2*Q + 1):
            opt_b[n, q_idx] = b_grid[b_idx[q_idx]]
            opt_a[n, q_idx] = a_grid[a_idx[q_idx]]
            
        # Enforce boundary conditions
        # If q == Q, set bid depth to inf (no quote)
        opt_b[n, 2*Q] = np.inf
        # If q == -Q, set ask depth to inf (no quote)
        opt_a[n, 0] = np.inf
        
        V_next = V_curr # Wait, V_curr is not computed yet from max.
        # V_curr(q) = max Obj[:, :, q]
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
        # Check time budget
        start_time = time.time()
        _dp_cache[key] = _solve_dp(sigma, A, k, gamma, T, steps, Q)
        elapsed = time.time() - start_time
        # If it takes too long, we might fail, but 200x200 grid is fast.
        
    solution = _dp_cache[key]
    opt_b = solution['opt_b']
    opt_a = solution['opt_a']
    
    # Determine current step
    dt = T / steps
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
        
    # Get depths for current inventory
    # inventory is array of ints
    # Map inventory q to index: q + Q
    q_idx = inventory + Q
    
    # Clip indices to valid range just in case
    q_idx = np.clip(q_idx, 0, 2*Q)
    
    bid_depths = opt_b[n, q_idx]
    ask_depths = opt_a[n, q_idx]
    
    return bid_depths, ask_depths
