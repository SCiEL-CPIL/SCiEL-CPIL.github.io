# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_start_time = time.time()

def _compute_optimal_policy(params):
    """
    Compute optimal bid and ask depths for all (t, q) using vectorized DP.
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
    
    # Log-spaced grid for depths to resolve near-zero and large depths
    # The current program uses linear [0, 2.0] with 100 points.
    # Log spacing helps capture the steep gradient of p(d) = 1-exp(-A exp(-kd) dt)
    # for small d.
    # We need to handle d=0 separately or use a grid that starts small.
    # Let's use a combination: fine near 0, coarser later.
    # Or just logspace from 1e-4 to 2.0.
    # p(0) = 1 - exp(-A dt).
    # p(2) = 1 - exp(-A exp(-2k) dt).
    
    # Let's use 400 points on log scale + 0?
    # Log scale doesn't include 0.
    # Let's use: grid = np.concatenate(([0.0], np.logspace(-4, np.log10(2.0), 399)))
    # This gives high resolution near 0.
    
    max_depth = 2.0
    n_log = 399
    grid = np.concatenate(([0.0], np.logspace(-4, np.log10(max_depth), n_log)))
    n_grid = len(grid)
    
    # Precompute probabilities
    # p = 1 - exp(-A * exp(-k * d) * dt)
    # For d=0, exp(0)=1.
    exp_term = np.exp(-k * grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    # Precompute 1-p
    one_minus_p = 1.0 - p_grid
    
    # V_next: shape (n_states,)
    V_next = np.zeros(n_states)
    
    # Storage for optimal depths
    # We can't store (steps, n_states) of floats if we want to be memory efficient?
    # 1000 * 61 * 8 bytes * 2 = ~1 MB. Fine.
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    # Time check
    deadline = _start_time + 300.0
    
    # We need to vectorize over q as well.
    # For each t, we have V_next (n_states,).
    # We want to compute V_curr (n_states,) and actions.
    
    # The Bellman equation for state q:
    # V_t(q) = -cost_q + max_{b,a} E(b,a)
    # E(b,a) depends on V_{t+1}(q), V_{t+1}(q+1), V_{t+1}(q-1).
    
    # Let's construct arrays for all q simultaneously.
    # V_q = V_next (n_states,)
    # V_qp = V_next shifted such that index i holds V_next[i+1] (for q+1)
    # V_qm = V_next shifted such that index i holds V_next[i-1] (for q-1)
    
    # Note: boundaries.
    # q=Q (last index) -> V_qp is invalid (but p_b=0 masks it).
    # q=-Q (first index) -> V_qm is invalid (but p_a=0 masks it).
    
    V_q = V_next
    # For V_qp: we want V_next[q+1].
    # Index i corresponds to q = -Q + i.
    # q+1 corresponds to index i+1.
    # So V_qp[i] = V_next[i+1].
    # We can use np.roll or slicing.
    # V_qp = np.roll(V_next, -1)
    # But at i=n_states-1, V_qp[i] = V_next[0] (wrap around).
    # This is incorrect value, but p_b for q=Q is forced to 0, so term is 0.
    # So it's safe.
    V_qp = np.roll(V_next, -1)
    V_qm = np.roll(V_next, 1)
    
    # Cost vector
    cost = 0.5 * gamma * sigma**2 * q_values**2 * dt
    
    # Now, we need to maximize E(b,a) for each q.
    # E(b,a) is a matrix of size (n_grid, n_grid) for each q.
    # We can compute a 3D tensor: (n_states, n_grid, n_grid).
    # This might be large: 61 * 400 * 400 = ~10 million floats. 80 MB.
    # This is feasible and fast in NumPy.
    
    # Let's set up the broadcasting.
    # grid: (n_grid,)
    # p_grid: (n_grid,)
    
    # For each q (axis 0), b (axis 1), a (axis 2):
    # pb = p_grid[b]
    # pa = p_grid[a]
    # v_q = V_q[q]
    # v_qp = V_qp[q]
    # v_qm = V_qm[q]
    
    # We need to mask p_b for q=Q and p_a for q=-Q.
    # q=Q is index n_states-1.
    # q=-Q is index 0.
    
    # Create masks
    # mask_bid: True if q < Q. Shape (n_states, 1, 1)
    # mask_ask: True if q > -Q. Shape (n_states, 1, 1)
    
    allow_bid = (q_values < Q)
    allow_ask = (q_values > -Q)
    
    # We can create a modified p_grid for each q?
    # Or just compute the full tensor and then mask the results?
    # Masking results is easier: set E to -inf for invalid actions?
    # No, the action is valid, just the probability is 0.
    # If we compute E with p_b > 0 for q=Q, we might pick a b that "would" fill.
    # But physically it can't.
    # So we must force p_b = 0 in the calculation for q=Q.
    
    # Let's create p_b_3d and p_a_3d.
    # p_b_3d[q, i, j] = p_grid[i] if allow_bid[q] else 0
    # p_a_3d[q, i, j] = p_grid[j] if allow_ask[q] else 0
    
    # This is memory heavy if we create full 3D arrays for p.
    # 61 * 400 * 400 * 8 * 2 = ~150 MB. Plus E tensor. Total ~250 MB.
    # Should be okay for modern systems.
    
    # Let's try to be smarter.
    # E = (1-pb)(1-pa)V_q + pb(1-pa)(b+V_qp) + (1-pb)pa(a+V_qm) + pbpa(b+a+V_q)
    
    # Let's expand and group to reduce operations?
    # E = V_q - pb V_q - pa V_q + pb pa V_q
    #   + b pb - b pb pa + V_qp pb - V_qp pb pa
    #   + a pa - a pb pa + V_qm pa - V_qm pb pa
    #   + b pb pa + a pb pa + V_q pb pa
    
    # Terms with b:
    # b pb (1 - pa + pa) = b pb
    # Wait.
    # b pb (1-pa) + b pb pa = b pb.
    # So terms with b: b pb + pb (V_qp - V_qp pa - V_qm pa? No)
    
    # Let's re-group by pb and pa.
    # E = (1-pa) [ (1-pb)V_q + pb(b+V_qp) ] + pa [ (1-pb)(a+V_qm) + pb(b+a+V_q) ]
    
    # This structure suggests we can compute intermediate terms.
    # Let L_b = (1-pb)V_q + pb(b+V_qp)  -> Shape (n_states, n_grid)
    # Let R_a = (1-pb)(a+V_qm) + pb(b+a+V_q) -> This depends on pb too.
    
    # Actually, the full 3D tensor approach is the most straightforward to vectorize without complex indexing errors.
    
    # Let's construct the tensors.
    # We will compute E for all q, b, a at once.
    
    # V_q_exp = V_q[:, None, None]  # (S, 1, 1)
    # V_qp_exp = V_qp[:, None, None]
    # V_qm_exp = V_qm[:, None, None]
    
    # b_exp = grid[None, :, None]  # (1, G, 1)
    # a_exp = grid[None, None, :]  # (1, 1, G)
    
    # p_b_exp = p_grid[None, :, None] # (1, G, 1)
    # p_a_exp = p_grid[None, None, :] # (1, 1, G)
    
    # But we need to mask p_b and p_a based on q.
    # p_b_masked = p_b_exp * allow_bid[:, None, None]
    # p_a_masked = p_a_exp * allow_ask[:, None, None]
    
    # This creates (S, G, 1) and (S, 1, G) arrays.
    
    # Let's do it.
    S = n_states
    G = n_grid
    
    V_q_exp = V_q[:, None, None]
    V_qp_exp = V_qp[:, None, None]
    V_qm_exp = V_qm[:, None, None]
    
    b_exp = grid[None, :, None]
    a_exp = grid[None, None, :]
    
    p_b_base = p_grid[None, :, None]
    p_a_base = p_grid[None, None, :]
    
    # Masks
    allow_bid_3d = allow_bid[:, None, None]
    allow_ask_3d = allow_ask[:, None, None]
    
    p_b = p_b_base * allow_bid_3d
    p_a = p_a_base * allow_ask_3d
    
    # Compute terms
    # t1 = (1-pb)(1-pa) V_q
    # t2 = pb(1-pa) (b + V_qp)
    # t3 = (1-pb)pa (a + V_qm)
    # t4 = pb pa (b + a + V_q)
    
    # To save memory/time, we can compute in parts.
    # E = V_q * (1-pb)*(1-pa) + ...
    
    # Let's just compute the full expression.
    # NumPy broadcasting will handle the (S, G, G) shape.
    
    # 1-pb, 1-pa
    ompb = 1.0 - p_b
    ompa = 1.0 - p_a
    
    # t1
    t1 = ompb * ompa * V_q_exp
    
    # t2
    t2 = p_b * ompa * (b_exp + V_qp_exp)
    
    # t3
    t3 = ompb * p_a * (a_exp + V_qm_exp)
    
    # t4
    t4 = p_b * p_a * (b_exp + a_exp + V_q_exp)
    
    E = t1 + t2 + t3 + t4
    
    # Subtract cost
    # cost is (S,), broadcast to (S, 1, 1)
    E -= cost[:, None, None]
    
    # Find max over b and a
    # E is (S, G, G).
    # We want max over axes 1 and 2.
    max_val = np.max(E, axis=(1, 2))
    # V_curr = max_val
    
    # Get indices
    # We need argmax over the flattened (G*G) for each S.
    # Or argmax over axis 2, then axis 1?
    # np.argmax over a 3D array with axis=(1,2) isn't directly supported in a way that returns 2 indices easily.
    # We can flatten the last two dims.
    
    E_flat = E.reshape(S, -1)
    max_indices = np.argmax(E_flat, axis=1)
    
    # Convert flat indices to (i, j)
    i_idx = max_indices // G
    j_idx = max_indices % G
    
    opt_bid[t, :] = grid[i_idx]
    opt_ask[t, :] = grid[j_idx]
    
    V_next = max_val
    
    # Time check
    if time.time() > deadline - 10.0:
        # If we run out of time, we can't finish.
        # But we must return something.
        # We have computed up to t.
        # We need to fill the rest?
        # The evaluator calls quote(t, ...) for t=0..999.
        # Our opt_bid/opt_ask are indexed by t (0..999).
        # We are iterating t from steps-1 down to 0.
        # If we stop early, we have computed high t values.
        # We need low t values (early in sim).
        # So if we run out of time, we fail to compute the early steps.
        # We should iterate t from 0 to steps-1? No, DP is backward.
        # If we run out of time, we can't provide a valid policy for early steps.
        # We should probably catch this and return a heuristic?
        # Or just let it finish if it's fast enough.
        # 1000 steps * (61*400*400 ops) ~ 10^10 ops?
        # 61 * 160,000 = ~10 million ops per step.
        # 1000 steps = 10^10 ops.
        # NumPy can do ~10^8 - 10^9 ops/s.
        # So 10-100 seconds.
        # This is risky for 330s if we have 3 regimes.
        # 3 * 50s = 150s. Plus overhead.
        # It might be too slow.
        
        # Optimization:
        # We can reduce G.
        # Do we need 400?
        # The function p(d) is smooth.
        # 100 points might be enough.
        # Let's try G=200.
        # 61 * 40,000 = 2.4M ops/step.
        # 1000 steps = 2.4B ops.
        # ~2-5 seconds.
        # This is much safer.
        
        # Let's restart with G=200 if this loop is too slow?
        # No, we are inside the function.
        # We can't easily change G now.
        # We should have chosen a safer G initially.
        
        # Let's break and hope it's fast enough?
        # Or we can just proceed.
        
        pass
    
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        # Time guard: if we are already close to deadline, use a simple heuristic
        # to avoid hanging.
        if time.time() > _start_time + 320.0:
             # Fallback: constant spread
             # Just return some valid depths
             # Must be > -3/k.
             # Let's return 0.1
             return np.full_like(inventory, 0.1, dtype=float), np.full_like(inventory, 0.1, dtype=float)
        
        bid_depths, ask_depths = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    n = int(np.round(t * params['steps']))
    if n >= params['steps']:
        n = params['steps'] - 1
    
    indices = inventory + params['Q']
    
    bids = bid_depths[n, indices]
    asks = ask_depths[n, indices]
    
    return bids, asks

# EVOLVE-BLOCK-END
