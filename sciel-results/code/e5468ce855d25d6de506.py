import numpy as np
from functools import lru_cache

# Cache for DP tables
_dp_cache = {}

def _compute_dp(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _dp_cache:
        return _dp_cache[key]
        
    dt = T / steps
    # Grid for probabilities
    # u_max = 1 - exp(-A * exp(-k*0) * dt) = 1 - exp(-A*dt)
    u_max = 1 - np.exp(-A * dt)
    # Use a grid of probabilities
    # We need enough resolution to find the optimum
    num_grid = 100
    u_grid = np.linspace(0, u_max, num_grid)
    v_grid = np.linspace(0, u_max, num_grid) # Symmetric
    
    # Precompute depths for grid points
    # b = -1/k * ln(u / (A * dt))
    # Handle u=0 -> b=inf
    # Handle u > A*dt * exp(0) = A*dt? No.
    # u = 1 - exp(-A e^{-kb} dt).
    # If b=0, u = 1 - exp(-A dt).
    # If b -> inf, u -> 0.
    # So u in [0, 1-exp(-A dt)].
    
    # Calculate b for each u in grid
    # u = 1 - exp(-A e^{-kb} dt)
    # 1-u = exp(-A e^{-kb} dt)
    # ln(1-u) = -A e^{-kb} dt
    # e^{-kb} = -ln(1-u) / (A dt)
    # -kb = ln(-ln(1-u) / (A dt))
    # b = -1/k * ln(-ln(1-u) / (A dt))
    
    def prob_to_depth(u):
        # u is array
        # Avoid log(0)
        u_safe = np.clip(u, 1e-9, 1 - 1e-9)
        val = -np.log(1 - u_safe) / (A * dt)
        # val must be > 0
        val = np.clip(val, 1e-9, None)
        b = -np.log(val) / k
        return b

    b_grid = prob_to_depth(u_grid)
    a_grid = prob_to_depth(v_grid)
    
    # Store optimal depths
    opt_b = np.zeros((steps, 2*Q + 1))
    opt_a = np.zeros((steps, 2*Q + 1))
    
    # V_next is value at n+1
    # V_n is value at n
    # q index: 0 to 2Q corresponds to q = -Q to Q
    
    V_next = np.zeros(2*Q + 1)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(2*Q + 1)
        for i, q in enumerate(range(-Q, Q + 1)):
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Get V values for neighbors
            # q+1 index is i+1, q-1 is i-1
            V_q = V_next[i]
            V_q_plus = V_next[i+1] if i < 2*Q else 0.0 # If q=Q, no bid fill, so this term is 0? 
            # Actually, if q=Q, u must be 0.
            
            # We can just optimize over the grid and enforce constraints
            # If q == Q, u must be 0.
            # If q == -Q, v must be 0.
            
            # Create meshgrid of u, v
            U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
            
            # Mask invalid states
            if q == Q:
                # u must be 0
                U = np.where(U == 0, U, np.inf) # This will make b inf?
                # Better: just set u=0 for all
                U = np.zeros_like(U)
            if q == -Q:
                V = np.zeros_like(V)
                
            # Calculate b, a for this grid
            # We can reuse b_grid, a_grid if we map indices
            # But U, V are values.
            # Let's just use the precomputed b_grid, a_grid based on indices
            
            # Re-indexing:
            # U is from u_grid. So B = b_grid[:, None] * np.ones_like(U)
            # V is from v_grid. So A_depth = a_grid[None, :] * np.ones_like(U)
            
            B = np.tile(b_grid[:, None], (1, len(v_grid)))
            A_depth = np.tile(a_grid[None, :], (len(u_grid), 1))
            
            # Probabilities
            P_b = U
            P_a = V
            
            # Immediate Reward
            # p_b b + p_a a
            # Note: if u=0, b is inf? No, we defined b_grid for u_grid.
            # u_grid[0] = 0. b_grid[0] = inf?
            # prob_to_depth(0) -> log(0) error.
            # We clipped u to 1e-9. So b is large but finite.
            # This is fine.
            
            R = P_b * B + P_a * A_depth
            
            # Future Value
            # P(bid only) = P_b (1-P_a)
            # P(ask only) = (1-P_b) P_a
            # P(neither) = (1-P_b)(1-P_a)
            # P(both) = P_b P_a
            
            # Inventory changes:
            # Bid only: q -> q+1
            # Ask only: q -> q-1
            # Both/Neither: q -> q
            
            # If q=Q, bid only is impossible. P(bid only) should be 0.
            # Our masking set U=0 if q=Q. So P_b=0.
            # Then P(bid only) = 0.
            # P(both) = 0.
            # P(ask only) = (1-0) P_a = P_a.
            # P(neither) = 1-P_a.
            # This matches the constraint.
            
            FV = P_b * (1 - P_a) * V_q_plus + \
                 (1 - P_b) * P_a * V_next[i-1] if i > 0 else 0.0 + \
                 (1 - P_b) * (1 - P_a) * V_q + \
                 P_b * P_a * V_q
            
            # Wait, V_next[i-1] is V(q-1).
            # If i=0 (q=-Q), V(q-1) is out of bounds.
            # But if q=-Q, V=0 (masking). So P_a=0.
            # Term (1-P_b) P_a V(q-1) becomes 0.
            # So we can just use 0.0 for V_next[i-1] if i=0.
            
            V_q_minus = V_next[i-1] if i > 0 else 0.0
            
            FV = P_b * (1 - P_a) * V_q_plus + \
                 (1 - P_b) * P_a * V_q_minus + \
                 (1 - P_b) * (1 - P_a) * V_q + \
                 P_b * P_a * V_q
            
            Total = R + FV
            
            # Find max
            idx = np.unravel_index(np.argmax(Total), Total.shape)
            opt_b[n, i] = B[idx]
            opt_a[n, i] = A_depth[idx]
            
            V_curr[i] = -cost + Total[idx]
            
        V_next = V_curr
        
    _dp_cache[key] = (opt_b, opt_a)
    return _dp_cache[key]

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n = int(t / (T / steps))
    if n >= steps: n = steps - 1
    
    opt_b, opt_a = _compute_dp(sigma, A, k, gamma, T, steps, Q)
    
    # inventory is in [-Q, Q]
    # Map to index [0, 2Q]
    indices = inventory + Q
    
    bid_depths = opt_b[n, indices]
    ask_depths = opt_a[n, indices]
    
    return bid_depths, ask_depths
