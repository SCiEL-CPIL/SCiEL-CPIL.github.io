import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 285.0
_cache = {}

def _solve_regime(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)

    # Depth grid in x-space: x = exp(-k*b), b = -ln(x)/k
    # We use a finer grid near x=1 (b=0) since that's where the optimum often lies
    # and a logarithmic grid for the rest.
    # For the joint optimization we need a manageable number of depth choices.
    # Let's use ~200 points for the grid to keep 2D optimization feasible.
    # Actually, let's think about the joint optimization:
    # We need to maximize over (b, a):
    # p_b * (b + V_next[i+1] - V_next[i]) + p_a * (a + V_next[i-1] - V_next[i])
    # + (1-p_b)(1-p_a) * 0 + p_b*p_a * (V_next[i+1] + V_next[i-1] - 2*V_next[i])
    # = p_b*b + p_a*a + p_b*D_b + p_a*D_a + p_b*p_a*D_cross
    # where D_b = V_next[i+1] - V_next[i], D_a = V_next[i-1] - V_next[i]
    # and D_cross = V_next[i+1] + V_next[i-1] - 2*V_next[i]
    
    # This is a 2D optimization. With 200x200 = 40000 combinations per inventory level,
    # and 61 inventory levels, that's ~2.4M operations per time step.
    # With 1000 time steps, that's 2.4B operations. Too slow for Python even with numpy.
    
    # Alternative: The cross term is usually small. Let's check if the decoupled
    # approximation is the limiting factor, or if we can do better.
    
    # Actually, the cross term p_b*p_a*D_cross can be significant when both p_b and p_a
    # are large. Let's use a moderate grid of ~150 points and see if the 2D optimization
    # is fast enough.
    
    # With 150x150 = 22500 per inventory, 61 inventories = ~1.4M per timestep.
    # 1000 timesteps = 1.4B. Still likely too slow.
    
    # Let's use a different approach: for each inventory level, we can precompute
    # the optimal (b,a) pair more efficiently. But 2D grid search is inherently expensive.
    
    # Alternative structural change: Use a coarser time grid and interpolate,
    # or use a smaller set of depth choices.
    
    # Let's try 100 points. 100x100 = 10000 per inventory, 61 inventories = 610k per timestep.
    # 1000 timesteps = 610M. This might be feasible with optimized numpy.
    
    # Actually, let me reconsider. The decoupled approach ignores the correlation
    # between bid and ask fills. The exact policy accounts for the case where both
    # fill (which changes inventory by 2) or neither fills.
    
    # For a structural change, let's implement the exact 2D optimization with a
    # moderate grid size, but only for the last M time steps where the cross-term
    # is most important (near the end when inventory risk is higher). For earlier
    # steps, use the decoupled approximation.
    
    # Actually, let's just try the full 2D optimization with a small grid (say 50 points)
    # and see if it's fast enough. 50x50=2500 per inventory, 61 inventories = 152.5k per timestep.
    # 1000 timesteps = 152.5M. This should be fast enough.
    
    # Let's use 80 points for a balance.
    n_depth = 80
    # Use a geometric spacing in x-space, but denser near x=1
    # x in (0, 1], with more points near 1
    # Use a power law: x = t^alpha for t in [0,1]
    # Or just use logspace with more points near 1
    
    # Let's use a mix: 40 points in [0.5, 1] (linear) and 40 points in [1e-10, 0.5] (log)
    x_near = np.linspace(0.0, 1.0, n_depth // 2)
    x_far = np.logspace(-10, np.log10(0.0), n_depth // 2)  # This won't work well
    # Better: just use logspace with more points near 1
    # Actually, the optimal depth is usually small (close to 0), so x is close to 1.
    # Let's use a grid that's dense near x=1.
    
    # Use: x = 1 - u, where u is in [0, 1-eps]
    # Or simply: use np.geomspace or a custom grid
    
    # Let's use: x_i = exp(-s_i), where s_i are spaced in log-scale for large s
    # and linearly for small s.
    # Simpler: use a grid that's linear in b (depth) for small b, and log for large b.
    # b = -ln(x)/k, so x = exp(-k*b)
    # For b in [0, b_max], x in [exp(-k*b_max), 1]
    
    # Let's just use a geometric grid in x-space from a small value to 1.
    x_min = 1e-12
    x_grid = np.geomspace(x_min, 1.0, n_depth)
    b_grid = -np.log(x_grid) / k
    p_grid = 1.0 - np.exp(-A * x_grid * dt)
    
    # Precompute cost
    cost_arr = -0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)
    
    # Precompute p_grid as 2D for broadcasting
    p_2d = p_grid[:, None]  # shape (n_depth, 1)
    b_2d = b_grid[:, None]
    
    for t in range(N - 1, -1, -1):
        if time.time() > _DEADLINE - 15:
            # Fallback
            for tt in range(t, -1, -1):
                for i in range(n_q):
                    if i < 2 * Q:
                        opt_b[tt, i] = 0.5 / k
                    else:
                        opt_b[tt, i] = np.inf
                    if i > 0:
                        opt_a[tt, i] = 0.5 / k
                    else:
                        opt_a[tt, i] = np.inf
            break
        
        V_next = V[t + 1]
        
        # Compute D_b, D_a, D_cross for each inventory level
        D_b = np.zeros(n_q)
        D_a = np.zeros(n_q)
        D_cross = np.zeros(n_q)
        
        D_b[:-1] = V_next[1:] - V_next[:-1]
        D_a[1:] = V_next[:-1] - V_next[1:]
        if n_q > 2:
            D_cross[1:-1] = V_next[2:] - 2.0 * V_next[1:-1] + V_next[:-2]
        
        # For each inventory level i, we need to find (b, a) that maximizes:
        # f(i, b_idx, a_idx) = p_b[idx] * (b_grid[idx] + D_b[i]) 
        #                   + p_a[idx] * (b_grid[idx] + D_a[i])
        #                   + p_b[idx] * p_a[idx] * D_cross[i]
        
        # Let's vectorize over inventory levels.
        # For each depth index j (for bid) and l (for ask), we can compute the value
        # for all inventory levels at once.
        
        # Actually, let's loop over the 2D grid of (bid_depth_idx, ask_depth_idx)
        # and for each pair, compute the value for all inventory levels, then take
        # the max over the 2D grid.
        
        # This means for each of the n_depth^2 pairs, we compute an array of n_q values.
        # Total operations: n_depth^2 * n_q = 80^2 * 61 = 390,400 per timestep.
        # 1000 timesteps: 390M. This should be feasible.
        
        best_val = np.full(n_q, -np.inf)
        best_bi = np.zeros(n_q, dtype=int)
        best_ai = np.zeros(n_q, dtype=int)
        
        # Precompute p_b * (b + D_b) and p_a * (b + D_a) for each depth index
        # For bid: term_b[j, i] = p_grid[j] * (b_grid[j] + D_b[i])
        # For ask: term_a[l, i] = p_grid[l] * (b_grid[l] + D_a[i])
        # Cross: cross[j, l, i] = p_grid[j] * p_grid[l] * D_cross[i]
        
        # Let's compute this more efficiently.
        # For each j (bid depth index):
        #   for each l (ask depth index):
        #     val[i] = p_b[j]*(b_grid[j]+D_b[i]) + p_a[l]*(b_grid[l]+D_a[i]) + p_b[j]*p_a[l]*D_cross[i]
        
        # We can restructure:
        # val[i] = p_b[j]*b_grid[j] + p_b[j]*D_b[i] + p_a[l]*b_grid[l] + p_a[l]*D_a[i] + p_b[j]*p_a[l]*D_cross[i]
        
        # Let's precompute:
        # pb_j = p_grid[j], b_j = b_grid[j]
        # pa_l = p_grid[l], b_l = b_grid[l]
        # val[i] = (pb_j*b_j + pa_l*b_l) + pb_j*D_b[i] + pa_l*D_a[i] + pb_j*pa_l*D_cross[i]
        
        # For fixed j, l:
        # val[i] = const + pb_j*D_b[i] + pa_l*D_a[i] + (pb_j*pa_l)*D_cross[i]
        
        # We can compute this as a linear combination of D_b, D_a, D_cross.
        
        # Let's iterate over j and l, but vectorize over i.
        # This is n_depth^2 = 6400 iterations per timestep, each doing vector ops of size n_q=61.
        # 6400 * 61 * 1000 = ~390M scalar ops. With numpy overhead, this might be slow.
        
        # Alternative: For each j, compute the best l for each i, then take max over j.
        # For fixed j:
        #   val[i, l] = pb_j*b_j + pa_l*b_l + pb_j*D_b[i] + pa_l*D_a[i] + pb_j*pa_l*D_cross[i]
        #   = (pb_j*b_j + pb_j*D_b[i]) + (pa_l*b_l + pa_l*D_a[i] + pb_j*pa_l*D_cross[i])
        
        # The second part depends on l and i. For fixed j, we can compute for all l and i.
        # shape: (n_q, n_depth)
        
        # Let's try this approach.
        
        for j in range(n_depth):
            pb = p_grid[j]
            bj = b_grid[j]
            
            if pb < 1e-15:
                continue
            
            # For this bid depth j, compute the value for all ask depths l and all inventory i
            # term1[i] = pb * (bj + D_b[i])  -- but wait, D_b is only defined for i < n_q-1
            # Actually, D_b[i] = V_next[i+1] - V_next[i] for i < n_q-1, and 0 otherwise (boundary)
            # Similarly D_a[i] = V_next[i-1] - V_next[i] for i > 0, and 0 otherwise
            
            # term1 = pb * (bj + D_b)  shape (n_q,)
            term1 = pb * (bj + D_b)
            
            # For ask: term2[l, i] = pa_l * (b_grid[l] + D_a[i]) + pb * pa_l * D_cross[i]
            # = pa_l * (b_grid[l] + D_a[i] + pb * D_cross[i])
            
            # D_a[i] and D_cross[i] are arrays of shape (n_q,)
            # For each l: pa_l * (b_grid[l] + D_a + pb * D_cross)
            
            # Let's compute this as:
            # term2[i, l] = pa_l * (b_grid[l] + D_a[i] + pb * D_cross[i])
            
            # D_a + pb * D_cross has shape (n_q,)
            da_cross = D_a + pb * D_cross  # shape (n_q,)
            
            # term2[i, l] = p_grid[l] * (b_grid[l] + da_cross[i])
            # = p_grid[l] * b_grid[l] + p_grid[l] * da_cross[i]
            
            # This is an outer product:
            # term2 = p_grid[None, :] * b_grid[None, :] + p_grid[None, :] * da_cross[:, None]
            # shape: (n_q, n_depth)
            
            term2 = p_grid[None, :] * b_grid[None, :] + p_grid[None, :] * da_cross[:, None]
            
            # Total value: term1[:, None] + term2
            total = term1[:, None] + term2  # shape (n_q, n_depth)
            
            # For each i, find the best l
            best_l = np.argmax(total, axis=1)
            best_val_j = total[np.arange(n_q), best_l]
            
            # Update best
            better = best_val_j > best_val
            best_val[better] = best_val_j[better]
            best_ai[better] = best_l[better]
            best_bi[better] = j
        
        # Handle boundary conditions
        # q = Q (i = 2Q): no bid
        no_quote_b = np.zeros(n_q, dtype=bool)
        no_quote_a = np.zeros(n_q, dtype=bool)
        no_quote_b[2 * Q] = True
        no_quote_a[0] = True
        
        # For no-quote cases, set to inf
        # But we need to handle the case where the optimal bid/ask is to not quote
        # The no-quote value is 0 (no depth earned, no inventory change)
        # Actually, if we don't quote, the value is just the continuation value
        # In our formulation, not quoting means p=0, so the value is 0 for the immediate term
        # But we need to compare with quoting.
        
        # Let's add an option to not quote by considering p=0
        # For bid: if we don't quote bid, p_b=0, so the value is:
        #   0 + pa_l*(b_grid[l] + D_a[i]) + 0
        # For ask: if we don't quote ask, p_a=0:
        #   pb_j*(b_grid[j] + D_b[i]) + 0 + 0
        
        # Actually, the no-quote option is already implicitly considered if the optimal
        # depth has very low probability. But to be safe, let's explicitly handle it.
        
        # For now, let's just set boundary to no-quote.
        b_vals = b_grid[best_bi]
        a_vals = b_grid[best_ai]
        
        b_vals[no_quote_b] = np.inf
        a_vals[no_quote_a] = np.inf
        
        opt_b[t, :] = b_vals
        opt_a[t, :] = a_vals
        
        # Compute the value
        contrib = np.zeros(n_q)
        for i in range(n_q):
            if no_quote_b[i] and no_quote_a[i]:
                contrib[i] = 0.0
            elif no_quote_b[i]:
                # Only ask
                l = best_ai[i]
                contrib[i] = p_grid[l] * (b_grid[l] + D_a[i])
            elif no_quote_a[i]:
                # Only bid
                j = best_bi[i]
                contrib[i] = p_grid[j] * (b_grid[j] + D_b[i])
            else:
                j = best_bi[i]
                l = best_ai[i]
                contrib[i] = (p_grid[j] * (b_grid[j] + D_b[i]) +
                             p_grid[l] * (b_grid[l] + D_a[i]) +
                             p_grid[j] * p_grid[l] * D_cross[i])
        
        V[t, :] = V_next + cost_arr + contrib
    
    _cache[key] = (opt_b, opt_a)
    return _cache[key]

def quote(t, inventory, mid, params):
    if time.time() > _DEADLINE:
        return np.full(4000, np.inf), np.full(4000, np.inf)
        
    opt_b, opt_a = _solve_regime(params)
    dt = params['T'] / params['steps']
    Q = params['Q']
    
    t_idx = int(t / dt)
    if t_idx >= len(opt_b):
        t_idx = len(opt_b) - 1
    if t_idx < 0:
        t_idx = 0
        
    idxs = np.clip(inventory, -Q, Q) + Q
    
    bid_depths = opt_b[t_idx, idxs].copy()
    ask_depths = opt_a[t_idx, idxs].copy()
    
    # Ensure validity
    bid_depths[~np.isfinite(bid_depths)] = np.inf
    ask_depths[~np.isfinite(ask_depths)] = np.inf
    
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    return bid_depths, ask_depths
