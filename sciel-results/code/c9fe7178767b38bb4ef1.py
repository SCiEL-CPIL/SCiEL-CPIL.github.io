"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via DP over (time, inventory) states.
    Uses closed-form first-order conditions for depth optimization.
    """
    dt = T / steps
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    # Depth bounds
    depth_min = 1e-4
    depth_max = 20.0 / k

    # Precompute for speed
    A_dt = A * dt
    k_inv = 1.0 / k

    # V_next[q_idx] where q_idx = q + Q, q in [-Q, Q]
    V_next = np.zeros(2 * Q + 1)

    # Store optimal depths: shape (steps, 2*Q+1)
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)

    q_vals = np.arange(-Q, Q + 1)

    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 3.0:
            # Fill remaining steps with current best estimate
            for nn in range(n - 1, -1, -1):
                opt_bid[nn] = opt_bid[n]
                opt_ask[nn] = opt_ask[n]
            break

        V_curr = np.zeros(2 * Q + 1)
        penalty = (gamma / 2.0) * sigma**2 * q_vals**2 * dt

        for q_idx in range(2 * Q + 1):
            q = q_idx - Q
            pen = penalty[q_idx]

            if q == Q:
                # Only ask quote allowed
                delta = V_next[q_idx - 1] - V_next[q_idx]
                # Maximize p(d) * (d + delta) where p(d) = 1 - exp(-A*exp(-k*d)*dt)
                # FOC: (d + delta) * (-k * exp(-k*d) * A_dt) + p(d) = 0
                # p(d) = (d + delta) * k * A_dt * exp(-k*d)
                # Let x = k*d, then: (1 - exp(-A_dt*exp(-x))) = (x/k + delta) * k * A_dt * exp(-x)
                # Use bisection/Newton on f(x) = (1 - exp(-A_dt*exp(-x))) - (x + k*delta) * A_dt * exp(-x) = 0
                # Actually: p(d) - (d+delta)*k*A_dt*exp(-k*d) = 0
                # Let's use a simple approach: try a few candidates and pick best
                
                # Candidate depths: use the structure of the problem
                # For small dt, p(d) ≈ A*exp(-k*d)*dt
                # FOC: (d+delta)*k*A*exp(-k*d)*dt + A*exp(-k*d)*dt = 0
                # (d+delta)*k + 1 = 0 => d* = -1/k - delta
                # But need d > 0, so d* = max(-1/k - delta, depth_min)
                
                d_star = max(-k_inv - delta, depth_min)
                d_star = min(d_star, depth_max)
                
                # Also try a few other candidates
                candidates = np.array([depth_min, d_star, min(2.0/k, depth_max), depth_max])
                p = 1.0 - np.exp(-A_dt * np.exp(-k * candidates))
                vals = p * (candidates + delta)
                best_i = np.argmax(vals)
                best_d = candidates[best_i]
                best_val = vals[best_i] - pen + V_next[q_idx]
                
                V_curr[q_idx] = best_val
                opt_ask[n, q_idx] = best_d
                opt_bid[n, q_idx] = np.inf

            elif q == -Q:
                # Only bid quote allowed
                delta = V_next[q_idx + 1] - V_next[q_idx]
                d_star = max(-k_inv - delta, depth_min)
                d_star = min(d_star, depth_max)
                
                candidates = np.array([depth_min, d_star, min(2.0/k, depth_max), depth_max])
                p = 1.0 - np.exp(-A_dt * np.exp(-k * candidates))
                vals = p * (candidates + delta)
                best_i = np.argmax(vals)
                best_d = candidates[best_i]
                best_val = vals[best_i] - pen + V_next[q_idx]
                
                V_curr[q_idx] = best_val
                opt_bid[n, q_idx] = best_d
                opt_ask[n, q_idx] = np.inf

            else:
                # Both sides allowed
                delta_plus = V_next[q_idx + 1] - V_next[q_idx]
                delta_minus = V_next[q_idx - 1] - V_next[q_idx]
                
                # Optimize bid: maximize p(d) * (d + delta_plus)
                d_star_b = max(-k_inv - delta_plus, depth_min)
                d_star_b = min(d_star_b, depth_max)
                candidates_b = np.array([depth_min, d_star_b, min(2.0/k, depth_max), depth_max])
                p_b = 1.0 - np.exp(-A_dt * np.exp(-k * candidates_b))
                vals_b = p_b * (candidates_b + delta_plus)
                best_bi = np.argmax(vals_b)
                best_db = candidates_b[best_bi]
                best_val_b = vals_b[best_bi]
                
                # Optimize ask: maximize p(d) * (d + delta_minus)
                d_star_a = max(-k_inv - delta_minus, depth_min)
                d_star_a = min(d_star_a, depth_max)
                candidates_a = np.array([depth_min, d_star_a, min(2.0/k, depth_max), depth_max])
                p_a = 1.0 - np.exp(-A_dt * np.exp(-k * candidates_a))
                vals_a = p_a * (candidates_a + delta_minus)
                best_ai = np.argmax(vals_a)
                best_da = candidates_a[best_ai]
                best_val_a = vals_a[best_ai]
                
                best_val = best_val_b + best_val_a - pen + V_next[q_idx]
                
                V_curr[q_idx] = best_val
                opt_bid[n, q_idx] = best_db
                opt_ask[n, q_idx] = best_da

        V_next = V_curr

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return _POLICY_CACHE[key]


def _analytical_quote(t, inventory, mid, params):
    """
    Avellaneda-Stoikov style analytical quote.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    remaining = T - t
    
    base = 1.0 / k
    skew_coeff = gamma * sigma**2 * remaining / 2.0
    c = skew_coeff / (2.0 * k)
    
    bid_depths = base + c * inventory
    ask_depths = base - c * inventory
    
    bid_depths = np.clip(bid_depths, 1e-4, 20.0 / k)
    ask_depths = np.clip(ask_depths, 1e-4, 20.0 / k)
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Quote using pre-computed optimal policy with analytical fallback."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    n = int(round(t / (T / steps)))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    # Try to compute policy if not cached and we have time
    if key not in _POLICY_CACHE:
        if time.time() < _DEADLINE - 15.0:
            try:
                _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
            except Exception:
                pass
    
    if key in _POLICY_CACHE:
        opt_bid, opt_ask = _POLICY_CACHE[key]
        q_idx = np.clip(inventory + Q, 0, 2 * Q)
        bid_depths = opt_bid[n, q_idx].copy()
        ask_depths = opt_ask[n, q_idx].copy()
        bid_depths = np.where(bid_depths < 1e-4, np.inf, bid_depths)
        ask_depths = np.where(ask_depths < 1e-4, np.inf, ask_depths)
        bid_depths = np.asarray(bid_depths, dtype=np.float64)
        ask_depths = np.asarray(ask_depths, dtype=np.float64)
        return bid_depths, ask_depths
    else:
        return _analytical_quote(t, inventory, mid, params)
# EVOLVE-BLOCK-END
