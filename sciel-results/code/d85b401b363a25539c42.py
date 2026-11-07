"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed policies per regime
_policy_cache = {}


def _compute_optimal_depths(params):
    """
    Compute optimal bid/ask depths via dynamic programming.
    Returns (bid_depths, ask_depths) arrays of shape (steps, 2*Q+1).
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)

    # Value function: V[n, i] where i is index into q_values
    # V[steps, :] = 0 (no terminal charge)
    V_next = np.zeros(n_states)

    # Store optimal actions
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))

    # Grid of depths to search over
    # Depth must be > 0 and finite. Use a reasonable grid.
    # The probability is 1 - exp(-A * exp(-k * d) * dt)
    # For small d, exp(-k*d) is large, prob is high.
    # For large d, exp(-k*d) is small, prob is low.
    # We want to balance inventory risk vs. capture.
    
    # Let's define a grid of depths.
    # Minimum depth: small enough to get high fill prob.
    # Maximum depth: large enough to have low fill prob (to avoid inventory risk).
    # 1/k is the "neutral" depth where exp(-k*1/k) = 1/e.
    
    d_min = 0.01
    d_max = 10.0 / k
    n_depths = 50
    depth_grid = np.linspace(d_min, d_max, n_depths)
    
    # Precompute probabilities for each depth
    # p_fill[d] = 1 - exp(-A * exp(-k * d) * dt)
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    # Earned value per fill = depth
    # Cost per step = gamma/2 * sigma^2 * q^2 * dt

    # Backward induction
    for n in range(steps - 1, -1, -1):
        # V_curr[i] = max over bid_d, ask_d of:
        # p_bid * (bid_d + V_next[i+1]) + (1-p_bid)*(1-p_ask)*V_next[i] + p_ask*(ask_d + V_next[i-1]) + (1-p_bid)*p_ask*V_next[i]
        # - gamma/2 * sigma^2 * q_i^2 * dt
        # Note: fills are independent.
        # Let p_b = p_fill(bid_d), p_a = p_fill(ask_d)
        # Expected next inventory:
        # q+1 with prob p_b * (1-p_a)
        # q-1 with prob (1-p_b) * p_a
        # q with prob 1 - p_b - p_a + p_b*p_a
        # Wait, bid fill increases inventory?
        # "A fill earns its depth and moves inventory one unit"
        # Usually buying (bid hit) increases inventory (long). Selling (ask hit) decreases inventory.
        # Let's assume:
        # Bid hit: buy at mid - bid_depth. Inventory increases by 1.
        # Ask hit: sell at mid + ask_depth. Inventory decreases by 1.
        
        # Objective: depths earned.
        # If bid hit: earn bid_depth.
        # If ask hit: earn ask_depth.
        
        # Reward = p_b * bid_d + p_a * ask_d
        # Transition:
        # q -> q+1 with prob p_b * (1 - p_a)
        # q -> q-1 with prob (1 - p_b) * p_a
        # q -> q with prob 1 - p_b - p_a + p_b * p_a
        
        # V_curr[i] = max_{d_b, d_a} [ p_b*d_b + p_a*d_a + p_b*(1-p_a)*V_next[i+1] + (1-p_b)*p_a*V_next[i-1] + (1-p_b-p_a+p_b*p_a)*V_next[i] - (gamma/2)*sigma^2*q_i^2*dt ]
        
        # Boundary conditions:
        # No bid fill at q = Q (i = 2*Q). So p_b must be 0? Or we just don't quote bid.
        # "no bid fill at q = Q, no ask fill at q = -Q"
        # This implies we can set bid_depth = inf (or just avoid the term) at Q.
        # Similarly for ask at -Q.
        
        # To simplify, we can iterate over possible depths.
        # Since n_states is small (61) and n_depths is 50, 50*50*61*1000 is ~150M operations.
        # This might be too slow in Python loops.
        # Vectorize over states?
        
        # Let's try to vectorize the max over (d_b, d_a) for all states simultaneously.
        # This is hard because the transition depends on i (q).
        # However, V_next is fixed for this step.
        
        # Alternative: Since the action space is small, maybe we can approximate or use a smarter grid.
        # Or, note that the optimal depth might be close to 1/k adjusted by inventory.
        
        # Let's try a coarser grid first to see if it's fast enough.
        # n_depths = 20. 20*20*61*1000 = 24M. Still slow in pure Python.
        
        # We can precompute the "value" of each depth combination for each state?
        # No, V_next changes each step.
        
        # Let's optimize the inner loop.
        # For each state i, we need to find max over d_b, d_a.
        # We can precompute terms that don't depend on d_b or d_a?
        # Term: (1-p_b-p_a+p_b*p_a)*V_next[i] = V_next[i] - p_b*V_next[i] - p_a*V_next[i] + p_b*p_a*V_next[i]
        
        # Let's restructure:
        # Value(d_b, d_a, i) = p_b*d_b + p_a*d_a + p_b*(1-p_a)*V_next[i+1] + (1-p_b)*p_a*V_next[i-1] + (1-p_b-p_a+p_b*p_a)*V_next[i] - cost_i
        
        # Expand:
        # = p_b*d_b + p_a*d_a + p_b*V_next[i+1] - p_b*p_a*V_next[i+1] + p_a*V_next[i-1] - p_b*p_a*V_next[i-1] + V_next[i] - p_b*V_next[i] - p_a*V_next[i] + p_b*p_a*V_next[i] - cost_i
        # = p_b*(d_b + V_next[i+1] - V_next[i]) + p_a*(d_a + V_next[i-1] - V_next[i]) + V_next[i] - cost_i - p_b*p_a*(V_next[i+1] + V_next[i-1] - 2*V_next[i])
        
        # Let Delta_b(i) = V_next[i+1] - V_next[i]
        # Let Delta_a(i) = V_next[i-1] - V_next[i]
        # Let Curv(i) = V_next[i+1] + V_next[i-1] - 2*V_next[i]
        
        # Value = p_b*(d_b + Delta_b(i)) + p_a*(d_a + Delta_a(i)) + V_next[i] - cost_i - p_b*p_a*Curv(i)
        
        # This is still coupled in p_b*p_a.
        # But if Curv(i) is small, we can ignore it?
        # V is concave in q (inventory risk), so V_next[i+1] + V_next[i-1] - 2*V_next[i] <= 0.
        # So -p_b*p_a*Curv(i) >= 0. It's a positive term.
        
        # Let's just do the double loop but vectorize over states if possible?
        # Actually, for a fixed d_b and d_a, the value for all states i can be computed in parallel.
        # But we need the max over d_b, d_a.
        # So we can iterate d_b, d_a and keep a running max for each state i.
        
        # Initialize max_val and best actions
        max_val = np.full(n_states, -np.inf)
        best_b = np.zeros(n_states)
        best_a = np.zeros(n_states)
        
        # Precompute Delta_b, Delta_a, Curv for current V_next
        # Need to handle boundaries for V_next[i+1] and V_next[i-1]
        # At i=0 (q=-Q), i-1 is out of bounds. But no ask fill at -Q, so p_a should be 0?
        # If we enforce p_a=0 at i=0, then the term involving V_next[i-1] vanishes.
        # Similarly at i=n_states-1 (q=Q), p_b=0.
        
        # To handle this cleanly, let's just set V_next boundaries to 0 or handle in the loop.
        # Actually, if q=-Q, we cannot sell (ask). So optimal ask_depth should be inf (p_a=0).
        # If q=Q, we cannot buy (bid). So optimal bid_depth should be inf (p_b=0).
        
        # We can enforce this by only considering valid depths for boundary states.
        # Or simpler: set V_next[i-1] = V_next[i] and V_next[i+1] = V_next[i] for boundaries?
        # No, that changes the physics.
        # Better: when i=0, force p_a=0. When i=n_states-1, force p_b=0.
        
        # Let's modify the loop to respect boundaries.
        # For each d_b, d_a:
        #   p_b = p_fill[d_b]
        #   p_a = p_fill[d_a]
        
        #   For all i:
        #   If i == 0: p_a_eff = 0
        #   If i == n_states-1: p_b_eff = 0
        #   Else: p_b_eff = p_b, p_a_eff = p_a
        
        #   Compute val[i] using p_b_eff, p_a_eff.
        
        # This can be vectorized!
        
        # Create masks for boundaries
        # But p_b_eff and p_a_eff depend on i.
        
        # Let's create arrays of p_b_eff and p_a_eff for each (d_b, d_a) pair?
        # That's 50*50*61 = 152500 floats. Fine.
        
        # Actually, we can precompute p_b_eff_grid and p_a_eff_grid?
        # No, p_b_eff depends on i only for the last state.
        
        # Let's just do it.
        
        # Precompute p_fill array
        # p_fill is shape (n_depths,)
        
        # Loop over d_b index
        for ib in range(n_depths):
            pb = p_fill[ib]
            db = depth_grid[ib]
            
            # If pb == 0, skip? No, db might be useful? No, if prob is 0, no fill, no earn.
            # But depth doesn't matter if prob is 0.
            
            for ia in range(n_depths):
                pa = p_fill[ia]
                da = depth_grid[ia]
                
                # Effective probabilities
                # p_b_eff: 0 at last state, else pb
                # p_a_eff: 0 at first state, else pa
                
                p_be = np.full(n_states, pb)
                p_be[-1] = 0.0
                
                p_ae = np.full(n_states, pa)
                p_ae[0] = 0.0
                
                # Compute value for all states
                # val = p_be*(db + Delta_b) + p_ae*(da + Delta_a) + V_next - cost - p_be*p_ae*Curv
                # Need Delta_b, Delta_a, Curv
                
                # Delta_b[i] = V_next[i+1] - V_next[i]
                # Delta_a[i] = V_next[i-1] - V_next[i]
                # Curv[i] = V_next[i+1] + V_next[i-1] - 2*V_next[i]
                
                # Handle boundaries for Delta/Curv
                # For i=0: Delta_a[0] = 0 (since no ask fill, effectively V_next[-1] doesn't matter)
                # For i=n_states-1: Delta_b[n_states-1] = 0
                
                # Let's compute Delta_b, Delta_a, Curv safely
                Delta_b = np.zeros(n_states)
                Delta_a = np.zeros(n_states)
                Curv = np.zeros(n_states)
                
                # Interior
                Delta_b[1:-1] = V_next[2:] - V_next[1:-1]
                Delta_a[1:-1] = V_next[:-2] - V_next[1:-1]
                Curv[1:-1] = V_next[2:] + V_next[:-2] - 2*V_next[1:-1]
                
                # Boundaries
                # At i=0: no ask fill, so p_ae=0. Terms with p_ae vanish.
                # val[0] = p_be[0]*(db + Delta_b[0]) + V_next[0] - cost[0]
                # We need Delta_b[0] = V_next[1] - V_next[0]
                Delta_b[0] = V_next[1] - V_next[0]
                
                # At i=n_states-1: no bid fill, so p_be=0.
                # val[-1] = p_ae[-1]*(da + Delta_a[-1]) + V_next[-1] - cost[-1]
                # We need Delta_a[-1] = V_next[-2] - V_next[-1]
                Delta_a[-1] = V_next[-2] - V_next[-1]
                
                # Now compute val
                cost = 0.5 * gamma * sigma**2 * (q_values**2) * dt
                val = (p_be * (db + Delta_b) + p_ae * (da + Delta_a) + V_next - cost - p_be * p_ae * Curv)
                
                # Update max
                mask = val > max_val
                max_val[mask] = val[mask]
                best_b[mask] = db
                best_a[mask] = da
        
        # Store optimal actions for step n
        opt_bid[n, :] = best_b
        opt_ask[n, :] = best_a
        
        # Update V_next for next iteration (backward)
        V_next = max_val
    
    return opt_bid, opt_ask


def _get_policy(params):
    """Get cached policy or compute it."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_optimal_depths(params)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Quote depths based on precomputed optimal policy.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 30s margin
    
    steps = params['steps']
    Q = params['Q']
    
    # Get or compute policy
    # If computation takes too long, we need a fallback.
    # But _compute_optimal_depths is deterministic and cached.
    # First call might be slow.
    
    try:
        opt_bid, opt_ask = _get_policy(params)
    except Exception:
        # Fallback to seed behavior
        depth = np.full(len(inventory), 1.0 / params['k'])
        return depth, depth.copy()
    
    # Determine current step index
    # t = n * dt
    # dt = T / steps
    dt = params['T'] / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    # Map inventory to index
    # q in [-Q, Q]
    # index = q + Q
    # inventory is (4000,) ints
    idx = inventory + Q
    
    # Clip indices to be safe
    idx = np.clip(idx, 0, 2*Q)
    
    # Get depths
    bid_depths = opt_bid[n, idx]
    ask_depths = opt_ask[n, idx]
    
    # Check for inf/NaN
    # The grid depths are finite.
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
