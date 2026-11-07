"""Optimal market making with inventory risk - pre-computed HJB lookup."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 280  # Safety margin before 330s kill
_policy_cache = {}


def _solve_regime(sigma, A, k, gamma, Q, steps, T):
    """Solve the discrete HJB backward induction for a given regime."""
    dt = T / steps
    n_q = 2 * Q + 1  # inventory states: -Q, ..., Q
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    q_idx = np.arange(n_q)
    
    # Depth grid for searching optimal depths
    # Use a reasonably fine grid; depths are typically in [0.01, 5.0]
    d_min = 0.005
    d_max = 5.0
    n_d = 200
    depth_grid = np.linspace(d_min, d_max, n_d)
    
    # Value function: V[n, q_idx] = max expected reward from step n onward
    # V[steps, q] = 0 (no terminal charge)
    V = np.zeros((steps + 1, n_q), dtype=np.float64)
    
    # Optimal depths: store the index into depth_grid for bid and ask
    opt_bid = np.zeros((steps, n_q), dtype=np.int32)
    opt_ask = np.zeros((steps, n_q), dtype=np.int32)
    
    # Precompute fill probabilities for each depth
    # p_bid(d) = 1 - exp(-A * exp(-k*d) * dt)
    # p_ask(d) = same formula
    exp_kd = np.exp(-k * depth_grid)
    lambda_d = A * exp_kd * dt
    p_fill = 1.0 - np.exp(-lambda_d)  # shape (n_d,)
    
    # Inventory risk cost per step
    risk_cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt  # shape (n_q,)
    
    # Mid move std
    sigma_dt = sigma * np.sqrt(dt)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # For each inventory state q, try all combinations of bid_depth and ask_depth
        # To save time, we can vectorize over inventory states
        
        # V_next = V[n+1, :] shape (n_q,)
        V_next = V[n + 1, :]
        
        # For each (bid_d, ask_d) pair, compute the expected value
        # This is expensive if done naively. Let's vectorize.
        
        # p_bid shape (n_d,), p_ask shape (n_d,)
        # For inventory q (index i):
        #   If q >= Q: bid_depth = inf (no bid quote)
        #   If q <= -Q: ask_depth = inf (no ask quote)
        #   
        #   Fill probabilities:
        #   p_b = p_fill[bid_idx] if q < Q else 0
        #   p_a = p_fill[ask_idx] if q > -Q else 0
        #   
        #   Outcomes (independent):
        #   1. No fill: prob (1-p_b)(1-p_a), reward = 0, next_q = q
        #   2. Bid fill only: prob p_b(1-p_a), reward = -d_bid, next_q = q+1
        #   3. Ask fill only: prob (1-p_b)p_a, reward = +d_ask, next_q = q-1
        #   4. Both fill: prob p_b*p_a, reward = -d_bid + d_ask, next_q = q (net zero)
        #   
        #   Plus risk cost: -risk_cost[i]
        #   Plus expected next value: E[V_next[next_q]]
        
        best_val = np.full(n_q, -np.inf)
        best_bi = np.zeros(n_q, dtype=np.int32)
        best_ai = np.zeros(n_q, dtype=np.int32)
        
        # For efficiency, process in chunks or use the fact that depth grid is shared
        # Let's compute for all depth pairs but vectorize over q
        
        # Actually, let's be smarter: for each depth pair (bi, ai), compute the
        # value for all q states at once.
        
        pb = p_fill  # (n_d,)
        pa = p_fill  # (n_d,)
        db = depth_grid  # (n_d,)
        da = depth_grid  # (n_d,)
        
        # Loop over bid depth indices (vectorized over ask and q)
        for bi in range(n_d):
            p_b = pb[bi]
            d_b = db[bi]
            
            # For all ask depths simultaneously
            # p_a shape (n_d,)
            # d_a shape (n_d,)
            
            # For each q state:
            # valid_bid = (q < Q) i.e., q_idx < Q  (since q_vals = -Q..Q, index Q corresponds to q=0... wait)
            # q_vals[i] = -Q + i
            # q < Q means -Q + i < Q means i < 2Q, so indices 0..2Q-1 can bid
            # q > -Q means -Q + i > -Q means i > 0, so indices 1..2Q can ask
            
            # Let's define masks
            can_bid = np.ones(n_q, dtype=bool)
            can_bid[2*Q] = False  # q = Q, cannot bid
            can_ask = np.ones(n_q, dtype=bool)
            can_ask[0] = False  # q = -Q, cannot ask
            
            # For q that can bid: p_b_eff = p_b, else 0
            # For q that can ask: p_a_eff = p_a[ai], else 0
            
            # Expected next inventory:
            # next_q = q + (bid_fill ? 1 : 0) - (ask_fill ? 1 : 0)
            # E[next_q] = q + p_b_eff - p_a_eff
            # But we need E[V[next_q]], not V[E[next_q]]
            
            # The four outcomes:
            # 1. No fill: prob (1-pb)(1-pa), next_q = q
            # 2. Bid only: prob pb(1-pa), next_q = q+1
            # 3. Ask only: prob (1-pb)pa, next_q = q-1
            # 4. Both: prob pb*pa, next_q = q
            
            # So E[V[next_q]] = (1-pb)(1-pa)*V[q] + pb(1-pa)*V[q+1] + (1-pb)pa*V[q-1] + pb*pa*V[q]
            # = [(1-pb)(1-pa) + pb*pa] * V[q] + pb(1-pa)*V[q+1] + (1-pb)pa*V[q-1]
            
            # Reward: -d_b * I(bid_fill) + d_a * I(ask_fill)
            # E[reward] = -d_b * pb + d_a * pa
            # But only if the side is valid. If can't bid, pb=0 effectively.
            
            # Let's compute for all q at once for a given (bi, ai):
            # This inner loop over ai is still O(n_d) per bi, total O(n_d^2 * n_q) per step
            # With n_d=200, n_q=61, steps=1000: 200*200*61*1000 = 2.44e9 - too slow!
            
            # Need a different approach. Let's reduce n_d or use a smarter method.
            
            # Alternative: for each q, the optimal bid and ask depths are separable
            # if we ignore the interaction (both fill) term. But that's an approximation.
            
            # Better: reduce depth grid size. n_d = 50 might be enough.
            # 50*50*61*1000 = 1.525e8 - still might be slow in Python.
            
            # Let's use n_d = 30 and see.
            # 30*30*61*1000 = 5.49e7 - feasible in NumPy if vectorized properly.
            
            pass  # This approach is too slow with the nested Python loop
        
        # Let's restart with a smaller grid and proper vectorization
        # Actually, let me restructure: for each step, for each q, find optimal (bi, ai)
        # But that's n_q * n_d^2 per step = 61 * 900 * 1000 = 5.49e7 with n_d=30
        # In Python this is still slow. Need full NumPy vectorization.
        
        # New approach: vectorize over all (bi, ai, q) combinations
        # bi: n_d, ai: n_d, q: n_q
        # Total: n_d * n_d * n_q = 900 * 61 = 54900 per step, times 1000 steps = 5.49e7
        # Each involves some arithmetic. In NumPy this should be fast.
        
        # Create meshes
        # bi_mesh: (n_d, 1, 1)
        # ai_mesh: (1, n_d, 1)  
        # q_mesh: (1, 1, n_q)
        
        # But we need V_next[q], V_next[q+1], V_next[q-1]
        # V_next shape (n_q,)
        
        # Let's build the value for all (bi, ai, q) triples
        # pb_bi = p_fill[bi] shape (n_d,)
        # pa_ai = p_fill[ai] shape (n_d,)
        
        # For q index i:
        #   q_val = -Q + i
        #   can_bid_i = (i < 2*Q)
        #   can_ask_i = (i > 0)
        
        # pb_eff = pb_bi * can_bid_i
        # pa_eff = pa_ai * can_ask_i
        
        # E[V_next] = [(1-pb_eff)(1-pa_eff) + pb_eff*pa_eff] * V_next[i] 
        #            + pb_eff*(1-pa_eff)*V_next[i+1]  (if i+1 < n_q)
        #            + (1-pb_eff)*pa_eff*V_next[i-1]  (if i-1 >= 0)
        
        # E[reward] = -db_bi * pb_eff + da_ai * pa_eff
        
        # Total = E[V_next] + E[reward] - risk_cost[i]
        
        # Let's do this with broadcasting
        # bi_idx: (n_d, 1, 1)
        # ai_idx: (1, n_d, 1)
        # q_idx:  (1, 1, n_q)
        
        bi_idx = np.arange(n_d).reshape(n_d, 1, 1)
        ai_idx = np.arange(n_d).reshape(1, n_d, 1)
        q_idx = np.arange(n_q).reshape(1, 1, n_q)
        
        pb = p_fill[bi_idx]  # (n_d, 1, 1)
        pa = p_fill[ai_idx]  # (1, n_d, 1)
        db = depth_grid[bi_idx]  # (n_d, 1, 1)
        da = depth_grid[ai_idx]  # (1, n_d, 1)
        
        # Masks
        can_bid = (q_idx < 2*Q).astype(np.float64)  # (1, 1, n_q)
        can_ask = (q_idx > 0).astype(np.float64)  # (1, 1, n_q)
        
        pb_eff = pb * can_bid  # (n_d, 1, n_q)
        pa_eff = pa * can_ask  # (1, n_d, n_q)
        
        # V_next lookups
        # V_next[i], V_next[i+1], V_next[i-1]
        # For i+1: if i+1 >= n_q, use 0 (out of bounds shouldn't happen due to masks)
        Vn = V_next  # (n_q,)
        Vn_plus = np.zeros(n_q)
        Vn_plus[:-1] = Vn[1:]
        Vn_plus[-1] = 0  # q=Q, bid fill would go to Q+1 which is invalid, but can_bid=0 there
        
        Vn_minus = np.zeros(n_q)
        Vn_minus[1:] = Vn[:-1]
        Vn_minus[0] = 0  # q=-Q, ask fill would go to -Q-1 which is invalid, but can_ask=0 there
        
        Vn_q = Vn.reshape(1, 1, n_q)  # (1, 1, n_q)
        Vn_qp = Vn_plus.reshape(1, 1, n_q)
        Vn_qm = Vn_minus.reshape(1, 1, n_q)
        
        # E[V_next]
        term1 = (1-pb_eff)*(1-pa_eff) + pb_eff*pa_eff  # (n_d, n_d, n_q)
        term2 = pb_eff*(1-pa_eff)
        term3 = (1-pb_eff)*pa_eff
        
        E_Vnext = term1 * Vn_q + term2 * Vn_qp + term3 * Vn_qm
        
        # E[reward]
        E_reward = -db * pb_eff + da * pa_eff  # (n_d, n_d, n_q)
        
        # Risk cost
        rc = risk_cost.reshape(1, 1, n_q)  # (1, 1, n_q)
        
        # Total value
        total = E_Vnext + E_reward - rc  # (n_d, n_d, n_q)
        
        # Find argmax over (bi, ai) for each q
        # Reshape to (n_d*n_d, n_q)
        total_2d = total.reshape(n_d*n_d, n_q)
        best_idx = np.argmax(total_2d, axis=0)  # (n_q,)
        
        # Decode best_idx back to (bi, ai)
        best_bi = best_idx // n_d  # (n_q,)
        best_ai = best_idx % n_d   # (n_q,)
        
        opt_bid[n, :] = best_bi.astype(np.int32)
        opt_ask[n, :] = best_ai.astype(np.int32)
        
        # Update V[n, q] = max value
        V[n, :] = total_2d[best_idx, np.arange(n_q)]
        
        # Check deadline
        if time.time() > _DEADLINE - 30:
            break
    
    return {
        'V': V,
        'opt_bid': opt_bid,
        'opt_ask': opt_ask,
        'depth_grid': depth_grid,
        'dt': dt,
    }


def _get_policy(params):
    """Get or compute the optimal policy for the given regime."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['Q'], params['steps'], params['T'])
    if key not in _policy_cache:
        _policy_cache[key] = _solve_regime(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['Q'], params['steps'], params['T']
        )
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """Return optimal bid/ask depths using pre-computed HJB policy."""
    try:
        policy = _get_policy(params)
    except Exception:
        # Fallback to heuristic
        sigma = params['sigma']
        A = params['A']
        k = params['k']
        gamma = params['gamma']
        Q = params['Q']
        q = inventory.astype(np.float64)
        base_depth = 1.0 / k
        beta = 0.02
        bid_depth = np.maximum(base_depth + beta * q, 0.01)
        ask_depth = np.maximum(base_depth - beta * q, 0.01)
        bid_depth[inventory >= Q] = np.inf
        ask_depth[inventory <= -Q] = np.inf
        return bid_depth, ask_depth
    
    Q = params['Q']
    steps = params['steps']
    depth_grid = policy['depth_grid']
    opt_bid = policy['opt_bid']
    opt_ask = policy['opt_ask']
    
    # Determine step index
    dt = policy['dt']
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Map inventory to index: q_idx = q + Q
    q_idx = inventory + Q  # (4000,) ints in [0, 2Q]
    q_idx = np.clip(q_idx, 0, 2*Q)
    
    # Look up optimal depth indices
    bi = opt_bid[n, q_idx]  # (4000,)
    ai = opt_ask[n, q_idx]  # (4000,)
    
    # Convert to actual depths
    bid_depth = depth_grid[bi]  # (4000,)
    ask_depth = depth_grid[ai]  # (4000,)
    
    # Boundary conditions: no bid at q=Q, no ask at q=-Q
    bid_depth[inventory >= Q] = np.inf
    ask_depth[inventory <= -Q] = np.inf
    
    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
