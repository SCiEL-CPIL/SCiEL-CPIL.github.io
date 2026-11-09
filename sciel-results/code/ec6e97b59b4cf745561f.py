import numpy as np
from scipy.optimize import brentq
import time

_START = time.time()
_DEADLINE = _START + 310.0
_CACHE = {}

def _solve_depth(C, A, k, dt):
    """
    Solve for optimal depth d given coupling constant C:
    J(d) = p(d)*d + C*p(d), p(d) = 1 - exp(-A*exp(-k*d)*dt)
    Optimal d satisfies: p'(d)*(d+C) + p(d) = 0
    """
    # p(0) = 1 - exp(-A*dt)
    p_0 = 1.0 - np.exp(-A * dt)
    if p_0 <= 1e-15:
        return 0.0
    
    # p'(0) = -k*A*dt*(1-p_0)
    p_0_prime = -k * A * dt * (1.0 - p_0)
    
    # Check if d=0 is optimal: f(0) = p'(0)*C + p(0) <= 0
    f_0 = p_0_prime * C + p_0
    if f_0 <= 0:
        return 0.0
    
    # Closed-form approximation for small A*dt: p(d) ≈ A*exp(-k*d)*dt
    # f(d) ≈ A*dt*exp(-k*d) * (1 - k*(d+C))
    # Root: d = 1/k - C
    d_approx = 1.0 / k - C
    if d_approx <= 0:
        return 0.0
    
    # Define the function to root-find
    def f(d):
        u = np.exp(-k * d)
        p = 1.0 - np.exp(-A * u * dt)
        dp = -k * A * dt * u * (1.0 - p)
        return dp * (d + C) + p
    
    # Check if approximation is good enough
    f_approx = f(d_approx)
    if abs(f_approx) < 1e-8:
        return max(0.0, d_approx)
    
    # Use brentq with bracket
    # f is decreasing for small d, check sign change
    d_lower = 0.0
    d_upper = max(6.0 / k, d_approx * 2.0 + 1.0 / k)
    
    f_upper = f(d_upper)
    if f_upper > 0:
        # Need to go further
        d_upper = 12.0 / k
        f_upper = f(d_upper)
    
    if f_0 * f_upper > 0:
        # No sign change, return approximation
        return max(0.0, min(d_approx, d_upper))
    
    try:
        d_sol = brentq(f, d_lower, d_upper, xtol=1e-10, rtol=1e-12, maxiter=50)
        return max(0.0, d_sol)
    except (ValueError, RuntimeError):
        return max(0.0, d_approx)

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    
    V_next = np.zeros(n_q)
    opt_b = np.zeros((N, n_q))
    opt_a = np.zeros((N, n_q))
    
    # Precompute common terms
    p_0 = 1.0 - np.exp(-A * dt)
    
    for t in range(N - 1, -1, -1):
        # Deadline check
        if time.time() > _DEADLINE - 10:
            # Fallback: use simple heuristic
            default_depth = 1.0 / k
            for t_rem in range(t, -1, -1):
                opt_b[t_rem, :] = default_depth
                opt_a[t_rem, :] = default_depth
            return opt_b, opt_a
        
        V_curr = np.zeros(n_q)
        q_vals = np.arange(-Q, Q + 1)
        
        for qi in range(n_q):
            q = q_vals[qi]
            
            # Get value deltas
            if qi < n_q - 1:
                delta_plus = V_next[qi + 1] - V_next[qi]  # ask fill: q -> q+1
            else:
                delta_plus = 0.0
            
            if qi > 0:
                delta_minus = V_next[qi - 1] - V_next[qi]  # bid fill: q -> q-1
            else:
                delta_minus = 0.0
            
            # Fixed-point iteration for (b, a)
            b = 0.0
            a = 0.0
            
            can_bid = (qi > 0)       # q > -Q
            can_ask = (qi < n_q - 1)  # q < Q
            
            for _iter in range(20):
                b_new = 0.0
                a_new = 0.0
                
                if can_bid:
                    # C_b = delta_plus * (1 - p_a)
                    p_a_curr = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    C_b = delta_plus * (1.0 - p_a_curr)
                    b_new = _solve_depth(C_b, A, k, dt)
                
                if can_ask:
                    # C_a = delta_minus * (1 - p_b)
                    p_b_curr = 1.0 - np.exp(-A * np.exp(-k * b_new) * dt)
                    C_a = delta_minus * (1.0 - p_b_curr)
                    a_new = _solve_depth(C_a, A, k, dt)
                
                if abs(b_new - b) < 1e-9 and abs(a_new - a) < 1e-9:
                    b = b_new
                    a = a_new
                    break
                b = b_new
                a = a_new
            
            opt_b[t, qi] = b
            opt_a[t, qi] = a
            
            # Compute V_curr[qi]
            p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt) if can_bid else 0.0
            p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt) if can_ask else 0.0
            
            if qi == 0:  # q = -Q, only ask fill possible
                reward = p_a * a
                exp_next = (1.0 - p_a) * V_next[qi] + p_a * V_next[qi + 1]
            elif qi == n_q - 1:  # q = Q, only bid fill possible
                reward = p_b * b
                exp_next = (1.0 - p_b) * V_next[qi] + p_b * V_next[qi - 1]
            else:
                # Both fills possible, independently
                reward = p_b * b + p_a * a
                prob_q_plus = p_b * (1.0 - p_a)
                prob_q_minus = (1.0 - p_b) * p_a
                prob_q = 1.0 - prob_q_plus - prob_q_minus
                exp_next = prob_q_plus * V_next[qi + 1] + prob_q_minus * V_next[qi - 1] + prob_q * V_next[qi]
            
            # Inventory holding cost
            risk = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            V_curr[qi] = reward - risk + exp_next
        
        V_next = V_curr
    
    return opt_b, opt_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _CACHE:
        _CACHE[key] = _precompute_regime(sigma, A, k, gamma, T, steps, Q)
    
    opt_b, opt_a = _CACHE[key]
    
    dt = T / steps
    t_idx = int(round(t / dt))
    if t_idx >= steps:
        t_idx = steps - 1
    if t_idx < 0:
        t_idx = 0
    
    q_idx = np.clip(inventory + Q, 0, 2 * Q)
    
    b_depths = opt_b[t_idx, q_idx]
    a_depths = opt_a[t_idx, q_idx]
    
    # Ensure valid depths
    b_depths = np.where(np.isfinite(b_depths) & (b_depths >= 0), b_depths, 0.0)
    a_depths = np.where(np.isfinite(a_depths) & (a_depths >= 0), a_depths, 0.0)
    
    return b_depths, a_depths
