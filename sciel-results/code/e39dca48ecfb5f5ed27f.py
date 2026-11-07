"""
Optimal market making via DP with continuous Newton refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 320.0


def _heuristic_depths(params, t, inventory):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    q = inventory.astype(np.float64)
    t_remaining = np.maximum(T - t, 0.0)
    d0 = 1.0 / k
    skew_scale = gamma * sigma * sigma * t_remaining
    alpha = skew_scale / (A * k) if A > 0 and k > 0 else 0.0
    alpha = min(alpha, 6.0 / k)
    bid_depth = d0 + alpha * q
    ask_depth = d0 - alpha * q
    min_depth = 0.01 / k
    max_depth = 10.0 / k
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)
    bid_depth = np.where(inventory >= Q, np.inf, bid_depth)
    ask_depth = np.where(inventory <= -Q, np.inf, ask_depth)
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= -3.0 / k), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= -3.0 / k), ask_depth, np.inf)
    return bid_depth, ask_depth


def _newton_refine(d_b, d_a, D1, Dm, A, k, dt):
    """
    Given initial depths d_b, d_a and value differences D1, Dm,
    run alternating Newton iterations to find the optimal depths.
    Returns refined (d_b, d_a).
    """
    max_iter = 15
    tol = 1e-10
    for _ in range(max_iter):
        # Compute p_b, p_a, dp_b, dp_a
        exp_b = np.exp(-k * d_b)
        exp_a = np.exp(-k * d_a)
        p_b = 1.0 - np.exp(-A * exp_b * dt)
        p_a = 1.0 - np.exp(-A * exp_a * dt)
        
        # Derivatives
        dp_b = -k * A * dt * exp_b * np.exp(-A * exp_b * dt)
        dp_a = -k * A * dt * exp_a * np.exp(-A * exp_a * dt)
        
        # FOC for d_b: p_b + dp_b * (d_b + D1*(1-p_a) - Dm*p_a) = 0
        # => d_b_new = d_b - [p_b + dp_b*(d_b + D1*(1-p_a) - Dm*p_a)] / [dp_b + dp_b2*(d_b + D1*(1-p_a) - Dm*p_a)]
        # Second derivative of p_b:
        # dp_b/dd_b = -k*A*dt*exp(-k*d_b)*exp(-A*exp(-k*d_b)*dt)
        # d2p_b/dd_b2 = -k*A*dt * [-k*exp(-k*d_b)*exp(-A*exp(-k*d_b)*dt) + exp(-k*d_b)*(-A*exp(-k*d_b)*dt)*(-k)*exp(-A*exp(-k*d_b)*dt)]
        # = -k*A*dt*exp(-k*d_b)*exp(-A*exp(-k*d_b)*dt)*[-k + k*A*exp(-k*d_b)*dt]
        # = k^2*A*dt*exp(-k*d_b)*exp(-A*exp(-k*d_b)*dt)*[1 - A*exp(-k*d_b)*dt]
        d2p_b = k**2 * A * dt * exp_b * np.exp(-A * exp_b * dt) * (1.0 - A * exp_b * dt)
        
        # FOC: f(d_b) = p_b + dp_b * (d_b + D1*(1-p_a) - Dm*p_a)
        # f'(d_b) = dp_b + dp_b*(d_b + D1*(1-p_a) - Dm*p_a) + d2p_b*(d_b + D1*(1-p_a) - Dm*p_a)
        #          = dp_b + (dp_b + d2p_b*(d_b + D1*(1-p_a) - Dm*p_a)) * (d_b + D1*(1-p_a) - Dm*p_a)
        
        bracket_b = d_b + D1 * (1.0 - p_a) - Dm * p_a
        f_b = p_b + dp_b * bracket_b
        fprime_b = dp_b + (dp_b + d2p_b * bracket_b) * bracket_b
        
        if abs(fprime_b) > 1e-15:
            d_b_new = d_b - f_b / fprime_b
        else:
            d_b_new = d_b
        
        # Similarly for d_a
        d2p_a = k**2 * A * dt * exp_a * np.exp(-A * exp_a * dt) * (1.0 - A * exp_a * dt)
        bracket_a = d_a + Dm * (1.0 - p_b) - D1 * p_b
        f_a = p_a + dp_a * bracket_a
        fprime_a = dp_a + (dp_a + d2p_a * bracket_a) * bracket_a
        
        if abs(fprime_a) > 1e-15:
            d_a_new = d_a - f_a / fprime_a
        else:
            d_a_new = d_a
        
        # Clamp to valid range
        min_d = 0.001
        max_d = 20.0 / k
        d_b_new = max(min_d, min(max_d, d_b_new))
        d_a_new = max(min_d, min(max_d, d_a_new))
        
        if abs(d_b_new - d_b) < tol and abs(d_a_new - d_a) < tol:
            d_b, d_a = d_b_new, d_a_new
            break
        d_b, d_a = d_b_new, d_a_new
    
    return d_b, d_a


def _compute_optimal_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_q = 2 * Q + 1
    q_range = np.arange(-Q, Q + 1)

    # Coarse grid for initial solution
    depth_min = 0.005
    depth_max = 15.0 / k
    n_depths = 120
    depth_grid = np.geomspace(depth_min, depth_max, n_depths)

    exp_term = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    depth_p = depth_grid * p_fill

    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    inv_cost_factor = (gamma / 2.0) * sigma**2 * dt

    # Backward DP with coarse grid
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 40:
            # Fallback
            for n2 in range(n, -1, -1):
                for qi in range(n_q):
                    q = q_range[qi]
                    d = 1.0 / k + gamma * sigma**2 * q * (T - n2 * dt) / (A * k)
                    d = max(d, depth_min)
                    bid_table[n2, qi] = d
                    ask_table[n2, qi] = d
            bid_table[:, n_q - 1] = np.inf
            ask_table[:, 0] = np.inf
            return bid_table, ask_table

        Vn1 = V[n + 1]

        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]

        D1 = V_plus - Vn1
        Dm1 = V_minus - Vn1

        for qi in range(n_q):
            q_val = q_range[qi]
            cost = inv_cost_factor * q_val * q_val

            if qi == n_q - 1:
                Pb = np.zeros(1)
                DPb = np.zeros(1)
            else:
                Pb = p_fill
                DPb = depth_p

            if qi == 0:
                Pa = np.zeros(1)
                DPa = np.zeros(1)
            else:
                Pa = p_fill
                DPa = depth_p

            # J = d_b*p_b + d_a*p_a + p_b*D1*(1-p_a) + p_a*Dm*(1-p_b) - cost
            term1 = DPb[:, None] + DPa[None, :]
            term2 = Pb[:, None] * D1[qi] * (1.0 - Pa[None, :]) + Pa[None, :] * Dm1[qi] * (1.0 - Pb[:, None])
            J = term1 + term2 - cost

            idx_max = np.unravel_index(np.argmax(J), J.shape)
            i_max, j_max = idx_max

            V[n, qi] = J[i_max, j_max]
            bid_table[n, qi] = depth_grid[i_max]
            ask_table[n, qi] = depth_grid[j_max]

    # Newton refinement pass
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 20:
            break

        Vn1 = V[n + 1]
        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]
        D1 = V_plus - Vn1
        Dm1 = V_minus - Vn1

        for qi in range(n_q):
            q_val = q_range[qi]
            cost = inv_cost_factor * q_val * q_val

            if qi == n_q - 1:
                # No bid
                d_a_init = ask_table[n, qi]
                d_a_refined = d_a_init
                # Optimize d_a only: J = d_a*p_a + p_a*Dm*(1-p_b) but p_b=0
                # J = d_a*p_a + p_a*Dm - cost
                # FOC: p_a + d_a*dp_a + Dm*dp_a = 0
                exp_a = np.exp(-k * d_a_refined)
                p_a = 1.0 - np.exp(-A * exp_a * dt)
                dp_a = -k * A * dt * exp_a * np.exp(-A * exp_a * dt)
                d2p_a = k**2 * A * dt * exp_a * np.exp(-A * exp_a * dt) * (1.0 - A * exp_a * dt)
                bracket = d_a_refined + Dm1[qi]
                f = p_a + dp_a * bracket
                fp = dp_a + (dp_a + d2p_a * bracket) * bracket
                if abs(fp) > 1e-15:
                    d_a_refined = d_a_refined - f / fp
                d_a_refined = max(0.001, min(20.0 / k, d_a_refined))
                # Iterate
                for _ in range(10):
                    exp_a = np.exp(-k * d_a_refined)
                    p_a = 1.0 - np.exp(-A * exp_a * dt)
                    dp_a = -k * A * dt * exp_a * np.exp(-A * exp_a * dt)
                    d2p_a = k**2 * A * dt * exp_a * np.exp(-A * exp_a * dt) * (1.0 - A * exp_a * dt)
                    bracket = d_a_refined + Dm1[qi]
                    f = p_a + dp_a * bracket
                    fp = dp_a + (dp_a + d2p_a * bracket) * bracket
                    if abs(fp) > 1e-15:
                        d_a_new = d_a_refined - f / fp
                    else:
                        break
                    d_a_new = max(0.001, min(20.0 / k, d_a_new))
                    if abs(d_a_new - d_a_refined) < 1e-10:
                        d_a_refined = d_a_new
                        break
                    d_a_refined = d_a_new
                
                ask_table[n, qi] = d_a_refined
                # Recompute V
                exp_a = np.exp(-k * d_a_refined)
                p_a = 1.0 - np.exp(-A * exp_a * dt)
                V[n, qi] = d_a_refined * p_a + p_a * Dm1[qi] - cost
            elif qi == 0:
                # No ask
                d_b_init = bid_table[n, qi]
                d_b_refined = d_b_init
                for _ in range(10):
                    exp_b = np.exp(-k * d_b_refined)
                    p_b = 1.0 - np.exp(-A * exp_b * dt)
                    dp_b = -k * A * dt * exp_b * np.exp(-A * exp_b * dt)
                    d2p_b = k**2 * A * dt * exp_b * np.exp(-A * exp_b * dt) * (1.0 - A * exp_b * dt)
                    bracket = d_b_refined + D1[qi]
                    f = p_b + dp_b * bracket
                    fp = dp_b + (dp_b + d2p_b * bracket) * bracket
                    if abs(fp) > 1e-15:
                        d_b_new = d_b_refined - f / fp
                    else:
                        break
                    d_b_new = max(0.001, min(20.0 / k, d_b_new))
                    if abs(d_b_new - d_b_refined) < 1e-10:
                        d_b_refined = d_b_new
                        break
                    d_b_refined = d_b_new
                
                bid_table[n, qi] = d_b_refined
                exp_b = np.exp(-k * d_b_refined)
                p_b = 1.0 - np.exp(-A * exp_b * dt)
                V[n, qi] = d_b_refined * p_b + p_b * D1[qi] - cost
            else:
                # Both sides active
                d_b_init = bid_table[n, qi]
                d_a_init = ask_table[n, qi]
                d_b, d_a = _newton_refine(d_b_init, d_a_init, D1[qi], Dm1[qi], A, k, dt)
                
                bid_table[n, qi] = d_b
                ask_table[n, qi] = d_a
                
                exp_b = np.exp(-k * d_b)
                exp_a = np.exp(-k * d_a)
                p_b = 1.0 - np.exp(-A * exp_b * dt)
                p_a = 1.0 - np.exp(-A * exp_a * dt)
                V[n, qi] = d_b * p_b + d_a * p_a + p_b * D1[qi] * (1.0 - p_a) + p_a * Dm1[qi] * (1.0 - p_b) - cost

    bid_table[:, n_q - 1] = np.inf
    ask_table[:, 0] = np.inf

    return bid_table, ask_table


def quote(t, inventory, mid, params):
    if time.time() - _START_TIME > TIME_LIMIT:
        return _heuristic_depths(params, t, inventory)

    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key not in _policy_cache:
        bid_table, ask_table = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_table, ask_table)
    else:
        bid_table, ask_table = _policy_cache[key]

    steps = params['steps']
    Q = params['Q']
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    q_index = inventory + Q
    q_index = np.clip(q_index, 0, 2 * Q)

    bid_depths = bid_table[n, q_index].copy()
    ask_depths = ask_table[n, q_index].copy()

    min_valid = -3.0 / params['k']
    bid_depths = np.where((bid_depths > min_valid) & np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where((ask_depths > min_valid) & np.isfinite(ask_depths), ask_depths, np.inf)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
