"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed optimal depths per (sigma, A, k, gamma, Q, steps)
_depth_cache = {}
_start_time = time.time()


def _solve_depth_for_cost(C, k, A, dt):
    """
    Solve for the optimal depth d given marginal cost C.
    The optimal d maximizes: d * p(d) - C * p(d) = p(d) * (d - C)
    where p(d) = 1 - exp(-A * exp(-k*d) * dt)
    
    FOC: (1 - exp(-L)) - k*L*(d - C)*exp(-L) = 0
    where L = A * exp(-k*d) * dt
    
    Equivalently: exp(L) - 1 = k*L*(d - C)
    
    For C <= 0: optimal d = 0 (always want to fill)
    For C > 0: solve numerically or use approximation
    """
    if C <= 0:
        return 0.0
    
    # Use the small-dt approximation: d ≈ C + 1/k
    # But refine with a few Newton iterations on the exact equation
    # exp(L) - 1 - k*L*(d - C) = 0, where L = A*exp(-k*d)*dt
    
    # Initial guess
    d = C + 1.0 / k
    
    # Clamp to reasonable range
    if d < 0:
        d = 0.0
    max_d = 30.0 / k  # very large depth means almost no fill
    if d > max_d:
        d = max_d
    
    # Newton's method on g(d) = exp(L) - 1 - k*L*(d - C)
    # where L = A*exp(-k*d)*dt
    # g'(d) = exp(L) * dL/dd - k*(dL/dd*(d-C) + L)
    # dL/dd = -k*L
    # g'(d) = exp(L)*(-k*L) - k*(-k*L*(d-C) + L)
    #        = -k*L*exp(L) + k^2*L*(d-C) - k*L
    #        = k*L*(k*(d-C) - exp(L) - 1)
    
    for _ in range(10):
        L = A * np.exp(-k * d) * dt
        if L < 1e-15:
            break
        g = np.exp(L) - 1.0 - k * L * (d - C)
        # g'(d) = k*L*(k*(d-C) - exp(L) - 1)
        gp = k * L * (k * (d - C) - np.exp(L) - 1.0)
        if abs(gp) < 1e-15:
            break
        d_new = d - g / gp
        if d_new < 0:
            d_new = 0.0
        if d_new > max_d:
            d_new = max_d
        if abs(d_new - d) < 1e-12:
            d = d_new
            break
        d = d_new
    
    return d


def _precompute_depths(sigma, A, k, gamma, Q, steps, T=1.0):
    """
    Precompute optimal bid/ask depths for each inventory level using
    a quadratic value function approximation with backward induction.
    
    Returns arrays bid_depths[q_idx] and ask_depths[q_idx]
    where q_idx ranges over -Q..Q (size 2Q+1).
    """
    dt = T / steps
    nq = 2 * Q + 1
    
    # Value function: V(q) ≈ a * q^2 + b * q + c
    # We track a and b through backward induction.
    # At terminal (t = steps): V = 0, so a = 0, b = 0
    
    a_next = 0.0
    b_next = 0.0
    
    # Backward induction: for each time step from steps-1 down to 0
    # The penalty term (gamma/2)*sigma^2*q^2*dt is incurred at each step
    # for the inventory BEFORE fills.
    
    # In steady state (time-homogeneous), a and b converge.
    # Let's iterate until convergence.
    
    a_curr = gamma * sigma**2 * dt / 2.0  # initial guess
    b_curr = 0.0
    
    for iteration in range(200):
        a_new = 0.0
        b_new = 0.0
        
        # For each inventory level, compute optimal depths and expected value
        sum_a_q2 = 0.0
        sum_b_q = 0.0
        count = 0
        
        for q in range(-Q, Q + 1):
            # Marginal cost of bid fill (q -> q+1)
            C_b = a_next * (2 * q + 1) + b_next
            # Marginal cost of ask fill (q -> q-1)
            C_a = -a_next * (2 * q - 1) + b_next
            
            # Optimal depths
            d_b = _solve_depth_for_cost(C_b, k, A, dt)
            d_a = _solve_depth_for_cost(C_a, k, A, dt)
            
            # Fill probabilities
            L_b = A * np.exp(-k * d_b) * dt
            L_a = A * np.exp(-k * d_a) * dt
            p_b = 1.0 - np.exp(-L_b)
            p_a = 1.0 - np.exp(-L_a)
            
            # Expected reward from fills
            reward = d_b * p_b + d_a * p_a
            
            # Expected future value change
            # E[V(q_next)] - V(q) = p_b*(1-p_a)*(V(q+1)-V(q)) + (1-p_b)*p_a*(V(q-1)-V(q))
            # V(q+1) - V(q) = a_next*(2q+1) + b_next
            # V(q-1) - V(q) = -a_next*(2q-1) + b_next
            delta_b = a_next * (2 * q + 1) + b_next
            delta_a = -a_next * (2 * q - 1) + b_next
            ev_change = p_b * (1 - p_a) * delta_b + (1 - p_b) * p_a * delta_a
            
            # Total value at this step (excluding constant c)
            # V(q, t) = (gamma/2)*sigma^2*q^2*dt + reward + V(q, t+1) + ev_change
            # = (gamma/2)*sigma^2*q^2*dt + reward + a_next*q^2 + b_next*q + c_next + ev_change
            # The a coefficient: (gamma/2)*sigma^2*dt + a_next
            # The b coefficient: b_next + (terms linear in q from reward and ev_change)
            
            # For the a coefficient:
            sum_a_q2 += ((gamma / 2.0) * sigma**2 * dt + a_next) * q * q
            # For the b coefficient, we need the linear part
            # reward and ev_change may have linear terms in q
            # Let's just accumulate the full V(q) - a*q^2 to get b
            V_q = ((gamma / 2.0) * sigma**2 * dt + a_next) * q * q + b_next * q + reward + ev_change
            sum_b_q += V_q - ((gamma / 2.0) * sigma**2 * dt + a_next) * q * q
            count += 1
        
        if count > 0:
            # Fit a and b from the accumulated values
            # We want V(q) ≈ a_new * q^2 + b_new * q
            # Use least squares: minimize sum (V_q - a_new*q^2 - b_new*q)^2
            # Normal equations:
            # sum V_q * q^2 = a_new * sum q^4 + b_new * sum q^3
            # sum V_q * q = a_new * sum q^3 + b_new * sum q^2
            
            q_vals = np.arange(-Q, Q + 1)
            q2 = q_vals * q_vals
            q3 = q_vals * q_vals * q_vals
            q4 = q2 * q2
            
            sum_q2 = np.sum(q2)
            sum_q3 = np.sum(q3)
            sum_q4 = np.sum(q4)
            
            # We need sum V_q * q^2 and sum V_q * q
            # Recompute V_q for all q
            V_vals = np.zeros(nq)
            for idx, q in enumerate(range(-Q, Q + 1)):
                C_b = a_next * (2 * q + 1) + b_next
                C_a = -a_next * (2 * q - 1) + b_next
                d_b = _solve_depth_for_cost(C_b, k, A, dt)
                d_a = _solve_depth_for_cost(C_a, k, A, dt)
                L_b = A * np.exp(-k * d_b) * dt
                L_a = A * np.exp(-k * d_a) * dt
                p_b = 1.0 - np.exp(-L_b)
                p_a = 1.0 - np.exp(-L_a)
                reward = d_b * p_b + d_a * p_a
                delta_b = a_next * (2 * q + 1) + b_next
                delta_a = -a_next * (2 * q - 1) + b_next
                ev_change = p_b * (1 - p_a) * delta_b + (1 - p_b) * p_a * delta_a
                V_vals[idx] = ((gamma / 2.0) * sigma**2 * dt + a_next) * q * q + b_next * q + reward + ev_change
            
            sum_Vq2 = np.sum(V_vals * q2)
            sum_Vq = np.sum(V_vals * q_vals)
            
            # Solve:
            # sum_q4 * a_new + sum_q3 * b_new = sum_Vq2
            # sum_q3 * a_new + sum_q2 * b_new = sum_Vq
            det = sum_q4 * sum_q2 - sum_q3 * sum_q3
            if abs(det) < 1e-15:
                a_new = a_curr
                b_new = b_curr
            else:
                a_new = (sum_Vq2 * sum_q2 - sum_Vq * sum_q3) / det
                b_new = (sum_Vq * sum_q4 - sum_Vq2 * sum_q3) / det
            
            # Check convergence
            if abs(a_new - a_curr) < 1e-10 and abs(b_new - b_curr) < 1e-10:
                a_curr = a_new
                b_curr = b_new
                break
            
            a_curr = a_new
            b_curr = b_new
    
    # Final a and b
    a_final = a_curr
    b_final = b_curr
    
    # Compute final depths
    bid_depths = np.zeros(nq)
    ask_depths = np.zeros(nq)
    
    for i in range(nq):
        q = i - Q
        C_b = a_final * (2 * q + 1) + b_final
        C_a = -a_final * (2 * q - 1) + b_final
        
        d_b = _solve_depth_for_cost(C_b, k, A, dt)
        d_a = _solve_depth_for_cost(C_a, k, A, dt)
        
        # Cap depths to avoid extremely large values
        max_depth = 20.0 / k
        if d_b > max_depth:
            d_b = np.inf
        if d_a > max_depth:
            d_a = np.inf
        
        bid_depths[i] = d_b
        ask_depths[i] = d_a
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Inventory-aware quoting with precomputed optimal depths."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    Q = params["Q"]
    steps = params["steps"]
    T = params["T"]
    
    # Deadline guard
    if time.time() - _start_time > 295:
        # Return default depths
        default_depth = 1.0 / k
        bid_depths = np.full(4000, default_depth, dtype=np.float64)
        ask_depths = np.full(4000, default_depth, dtype=np.float64)
        bid_depths[inventory >= Q] = np.inf
        ask_depths[inventory <= -Q] = np.inf
        return bid_depths, ask_depths
    
    # Check cache
    key = (sigma, A, k, gamma, Q, steps, T)
    if key not in _depth_cache:
        _depth_cache[key] = _precompute_depths(sigma, A, k, gamma, Q, steps, T)
    
    bid_depths_table, ask_depths_table = _depth_cache[key]
    
    # Map inventory to index
    q = inventory
    indices = (q + Q).astype(np.int64)
    indices = np.clip(indices, 0, 2 * Q)
    
    bid_depths = bid_depths_table[indices].copy()
    ask_depths = ask_depths_table[indices].copy()
    
    # Enforce no bid fill at q = Q, no ask fill at q = -Q
    bid_depths[q >= Q] = np.inf
    ask_depths[q <= -Q] = np.inf
    
    # Ensure all values are valid (no NaN, no values below -3/k)
    min_valid = -3.0 / k
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= min_valid), ask_depths, np.inf)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
