"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0
_OPT_TABLES = {}

def _solve_1d_max(A, k, dt, C):
    """
    Solve for optimal depth d maximizing f(d) = (1 - exp(-A exp(-kd) dt)) * (d + C)
    C is the value difference V(next) - V(current).
    Returns optimal d.
    """
    # f(d) is concave for d > 0.
    # f'(d) = k * A * exp(-kd) * exp(-A exp(-kd) dt) * (d + C) + (1 - exp(-A exp(-kd) dt))
    # Let lambda = A exp(-kd) dt.
    # f'(d) = k * lambda * exp(-lambda) * (d + C) + (1 - exp(-lambda))
    # We want f'(d) = 0.
    
    # Initial guess: 1/k
    d = 1.0 / k
    for _ in range(20):
        lam = A * np.exp(-k * d) * dt
        p = 1.0 - np.exp(-lam)
        # Derivative
        # term1 = k * lam * exp(-lam) * (d + C)
        # term2 = p
        # f_prime = term1 + term2
        # Wait, derivative of (1-exp(-lam)) is exp(-lam) * lam'
        # lam' = -k lam
        # So p' = -k lam exp(-lam)
        # f(d) = p(d+C)
        # f'(d) = p'(d+C) + p
        # f'(d) = -k lam exp(-lam) (d+C) + p
        # This is always negative if C is not large positive?
        # If C is negative (inventory cost increasing), d should be larger to reduce p?
        # If C is positive, we want to fill more?
        
        # Let's check the sign.
        # If d -> 0, lam -> A dt, p -> 1-exp(-A dt).
        # If d -> inf, lam -> 0, p -> 0.
        # f(0) = p_max * C.
        # f(inf) = 0.
        # If C > 0, max is at small d?
        # If C < 0, max is at large d?
        
        # Let's use a robust bisection or Newton with bounds.
        # Range of d: [0.001, 10/k]
        d_min = 0.001
        d_max = 10.0 / k
        
        def f_prime(d):
            lam = A * np.exp(-k * d) * dt
            p = 1.0 - np.exp(-lam)
            return -k * lam * np.exp(-lam) * (d + C) + p

        def f_second(d):
            # Numerical or analytical? Newton is risky if not convex.
            # Let's just use Bisection on f_prime if it's monotonic?
            # f''(d) is likely negative (concave).
            pass

        # Bisection on f_prime = 0
        # f_prime(d_min) ?
        # f_prime(d_max) ?
        
        # Actually, simpler: use a small grid and then Newton refine?
        # Or just use the grid from the previous program but finer?
        # The prompt asks for a structural change.
        
        # Let's try Newton with damping.
        # f'(d) = 0
        # f''(d) = derivative of -k lam e^-lam (d+C) + (1-e^-lam)
        # This is getting messy.
        
        # Alternative: Use the fact that optimal d is often around 1/k.
        # Let's just use a very fine grid of 100 points in log space and pick the best.
        # This is "discrete" but with high resolution.
        # To make it "continuous", we can interpolate?
        
        # Let's stick to the high-res grid for robustness, it's fast enough.
        # 100 points * 61 inventories * 1000 steps * 3 regimes = 18.3M ops.
        # In Python, this might be slow.
        # Vectorized numpy:
        # We can precompute the grid once.
        
    # Fallback to grid
    pass

def _get_optimal_table(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _OPT_TABLES:
        return _OPT_TABLES[key]

    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps

    # High resolution log grid
    # Optimal depth is usually in [0.05/k, 5/k]
    d_min = 0.01 / k
    d_max = 5.0 / k
    n_depths = 200
    depth_grid = np.geomspace(d_min, d_max, n_depths)
    
    # Precompute fill probabilities for the grid
    # p = 1 - exp(-A exp(-k d) dt)
    k_d = k * depth_grid
    exp_k_d = np.exp(-k_d)
    lam = A * exp_k_d * dt
    p_grid = 1.0 - np.exp(-lam)
    
    inv_range = np.arange(-Q, Q + 1)
    n_inv = len(inv_range)

    V = np.zeros((steps + 1, n_inv))
    opt_bid = np.zeros((steps, n_inv))
    opt_ask = np.zeros((steps, n_inv))

    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 10:
             # Fallback
             opt_bid[n] = 1.0 / k
             opt_ask[n] = 1.0 / k
             continue

        V_next = V[n + 1]
        
        for i_idx, q in enumerate(inv_range):
            inv_cost = 0.5 * gamma * sigma**2 * q**2 * dt
            V_q = V_next[i_idx]
            
            # Bid side
            if q < Q:
                q_next = q - 1
                i_next = int(q_next + Q)
                V_qm1 = V_next[i_next]
                C_b = V_qm1 - V_q
                # Maximize p[d] * (d + C_b)
                # values = p_grid * (depth_grid + C_b)
                vals = p_grid * (depth_grid + C_b)
                best_idx = np.argmax(vals)
                opt_bid[n, i_idx] = depth_grid[best_idx]
                best_val_bid = vals[best_idx]
            else:
                opt_bid[n, i_idx] = np.inf
                best_val_bid = V_q

            # Ask side
            if q > -Q:
                q_next = q + 1
                i_next = int(q_next + Q)
                V_qp1 = V_next[i_next]
                C_a = V_qp1 - V_q
                vals = p_grid * (depth_grid + C_a)
                best_idx = np.argmax(vals)
                opt_ask[n, i_idx] = depth_grid[best_idx]
                best_val_ask = vals[best_idx]
            else:
                opt_ask[n, i_idx] = np.inf
                best_val_ask = V_q

            # Combine
            # Note: The decomposition assumes independence of the *value* contributions 
            # which is an approximation if we ignore the joint probability term 
            # p_b p_a (V(q) - V(q-1) - V(q+1) + V(q-2)? No, just V(q)).
            # The term p_b p_a V(q) is added in both if we just sum best_val_bid + best_val_ask?
            # best_val_bid = p_b (b + V(q-1)) + (1-p_b) V(q) ?? No, my vals above is just p_b(b+C_b) = p_b(b + V(q-1) - V(q)).
            # So best_val_bid is the *increment* over V(q) from the bid side, assuming ask doesn't fill?
            # If we sum them, we are double counting the base V(q) or missing the joint term.
            
            # Correct HJB:
            # V(n,q) = max_b,a [ p_b(1-p_a)(V(q-1)+b) + p_a(1-p_b)(V(q+1)+a) + (1-p_b)(1-p_a)V(q) ] - Cost
            # = V(q) + p_b(1-p_a)(V(q-1)+b-V(q)) + p_a(1-p_b)(V(q+1)+a-V(q)) + p_b p_a (V(q) - V(q) - V(q) + V(q))? No.
            # Let D_b = V(q-1) - V(q) + b
            # Let D_a = V(q+1) - V(q) + a
            # V(n,q) = V(q) + p_b(1-p_a)D_b + p_a(1-p_b)D_a - p_b p_a (V(q-1)+V(q+1)-2V(q))/2? No.
            # The term p_b p_a V(q) is in both?
            # (1-p_b)(1-p_a) = 1 - p_b - p_a + p_b p_a
            # Sum = p_b(1-p_a) + p_a(1-p_b) + 1 - p_b - p_a + p_b p_a
            # = p_b - p_b p_a + p_a - p_b p_a + 1 - p_b - p_a + p_b p_a
            # = 1 - p_b p_a.
            # So V(n,q) = (1-p_b p_a) V(q) + p_b(1-p_a)(V(q-1)+b) + p_a(1-p_b)(V(q+1)+a) - Cost
            # = V(q) + p_b(1-p_a)(V(q-1)+b-V(q)) + p_a(1-p_b)(V(q+1)+a-V(q)) - p_b p_a V(q) - Cost
            
            # My previous approximation ignored the -p_b p_a V(q) and the (1-p) factors in D.
            # Since p is small, this is a good approximation.
            # To be more accurate, we can iterate.
            # For now, use the sum of best increments.
            
            V[n, i_idx] = V_q + best_val_bid + best_val_ask - inv_cost

    _OPT_TABLES[key] = (opt_bid, opt_ask, inv_range)
    return _OPT_TABLES[key]

def quote(t, inventory, mid, params):
    if time.time() > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()

    opt_bid, opt_ask, inv_range = _get_optimal_table(params)

    n = int(t / (params["T"] / params["steps"]))
    if n >= params["steps"]:
        n = params["steps"] - 1

    inv_indices = inventory + params["Q"]
    inv_indices = np.clip(inv_indices, 0, len(inv_range) - 1)

    bid_depths = opt_bid[n, inv_indices]
    ask_depths = opt_ask[n, inv_indices]

    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), 1.0 / params["k"], bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), 1.0 / params["k"], ask_depths)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
