import numpy as np
import time
import threading

# Cache for precomputed tables
_TABLE_CACHE = {}
_TABLE_LOCK = threading.Lock()

# Global deadline tracking
_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # 20s safety margin before 330s kill

def _check_deadline():
    return time.time() >= _DEADLINE

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid and ask depths for all steps and inventories.
    Uses vectorized Newton's method for the optimality condition to speed up
    the fixed-point iteration.
    """
    dt = T / steps
    N = steps
    max_q = Q
    min_q = -Q
    n_states = 2 * Q + 1
    
    V_next = np.zeros(n_states)
    opt_bid = np.zeros((N, n_states))
    opt_ask = np.zeros((N, n_states))
    
    # Precompute q values for risk calculation
    q_vals = np.arange(n_states) - Q
    risk = (gamma / 2.0) * (sigma**2) * (q_vals**2) * dt
    
    # Initialize b and a
    b = np.zeros(n_states)
    a = np.zeros(n_states)
    
    # Precompute shifts for V_next
    V_next_plus = np.zeros(n_states)
    V_next_minus = np.zeros(n_states)
    V_next_plus[:-1] = V_next[1:]
    V_next_minus[1:] = V_next[:-1]
    
    # Constants for Newton's method
    # g(x) = 1 - exp(-A exp(-kx) dt)
    # g'(x) = -k A dt exp(-kx) exp(-A exp(-kx) dt)
    # h(x) = -g'(x) = k A dt exp(-kx) exp(-A exp(-kx) dt)
    # phi(x) = g(x)/h(x) - x
    # phi'(x) = (g'(x)h(x) - g(x)h'(x))/h(x)^2 - 1
    # We can compute phi'(x) numerically or analytically.
    # Let's use a small numerical derivative for simplicity and stability.
    eps = 1e-7
    
    for n in range(N - 1, -1, -1):
        if _check_deadline():
            # Return what we have so far, fill remaining with a reasonable default
            depth_default = 1.0 / k
            for n_fill in range(n, N):
                opt_bid[n_fill, :] = depth_default
                opt_ask[n_fill, :] = depth_default
            return opt_bid, opt_ask
        
        # Update deltas based on current V_next
        delta_plus = np.zeros(n_states)
        delta_minus = np.zeros(n_states)
        delta_plus[:-1] = V_next[1:] - V_next[:-1]
        delta_minus[1:] = V_next[:-1] - V_next[1:]
        
        # Fixed point iteration for b and a
        max_iter = 30
        for _iter in range(max_iter):
            if _check_deadline():
                break
            
            # Compute p_a from current a
            p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
            
            # Compute C_b
            C_b = delta_plus * (1.0 - p_a) - p_a * delta_minus
            
            # Newton's method to solve phi(b) = C_b
            # phi(x) = g(x)/h(x) - x - C_b = 0
            # We want to find b such that phi(b) = C_b.
            # Let F(x) = g(x)/h(x) - x - C_b.
            # F'(x) = (g'(x)h(x) - g(x)h'(x))/h(x)^2 - 1.
            
            # Compute g, h, g', h' for current b
            exp_kb = np.exp(-k * b)
            exp_Au_b = np.exp(-A * exp_kb * dt)
            g_b = 1.0 - exp_Au_b
            h_b = k * A * dt * exp_kb * exp_Au_b
            # h'(x) = -k h(x) + k^2 A dt exp(-2kx) exp(-A exp(-kx) dt)
            # Let's derive h'(x) carefully.
            # h(x) = k A dt exp(-kx) exp(-A exp(-kx) dt)
            # Let u = -kx, v = -A exp(-kx) dt.
            # h = k A dt e^u e^v.
            # h' = k A dt (e^u u' e^v + e^u e^v v')
            # u' = -k
            # v' = -A dt * exp(-kx) * (-k) = k A dt exp(-kx)
            # h' = k A dt e^u e^v (-k + k A dt exp(-kx))
            # h' = h (-k + k A dt exp(-kx))
            # h' = -k h + k^2 A dt exp(-kx) e^v
            # h' = -k h + k^2 A dt exp(-kx) exp(-A exp(-kx) dt)
            
            h_prime_b = -k * h_b + k**2 * A * dt * exp_kb * exp_Au_b
            g_prime_b = -h_b
            
            # F'(b) = (g'(b)h(b) - g(b)h'(b))/h(b)^2 - 1
            # Avoid division by zero
            h_b_safe = np.where(h_b < 1e-15, 1e-15, h_b)
            F_prime_b = (g_prime_b * h_b - g_b * h_prime_b) / (h_b_safe * h_b_safe) - 1.0
            
            # Newton step
            F_b = g_b / h_b_safe - b - C_b
            b_new = b - F_b / F_prime_b
            
            # Clamp b_new to valid range [0, 10/k]
            b_max = 10.0 / k
            b_new = np.clip(b_new, 0.0, b_max)
            
            # Compute p_b
            p_b = 1.0 - np.exp(-A * np.exp(-k * b_new) * dt)
            
            # Compute C_a
            C_a = delta_minus * (1.0 - p_b) - p_b * delta_plus
            
            # Newton's method for a
            exp_ka = np.exp(-k * a)
            exp_Au_a = np.exp(-A * exp_ka * dt)
            g_a = 1.0 - exp_Au_a
            h_a = k * A * dt * exp_ka * exp_Au_a
            h_prime_a = -k * h_a + k**2 * A * dt * exp_ka * exp_Au_a
            g_prime_a = -h_a
            
            h_a_safe = np.where(h_a < 1e-15, 1e-15, h_a)
            F_prime_a = (g_prime_a * h_a - g_a * h_prime_a) / (h_a_safe * h_a_safe) - 1.0
            
            F_a = g_a / h_a_safe - a - C_a
            a_new = a - F_a / F_prime_a
            a_new = np.clip(a_new, 0.0, b_max)
            
            # Convergence check
            if np.allclose(b_new, b, atol=1e-8) and np.allclose(a_new, a, atol=1e-8):
                b = b_new
                a = a_new
                break
            
            b = b_new
            a = a_new
        
        # Store optimal depths
        opt_bid[n, :] = b
        opt_ask[n, :] = a
        
        # Compute V_curr
        # p_b, p_a are already computed from final b, a
        p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
        p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
        
        # Handle boundaries
        p_b_effective = p_b.copy()
        p_a_effective = p_a.copy()
        p_b_effective[-1] = 0.0  # No bid fill at max inventory
        p_a_effective[0] = 0.0   # No ask fill at min inventory
        
        # Recalculate probabilities with effective p's
        prob_q_plus = p_b_effective * (1.0 - p_a_effective)
        prob_q_minus = (1.0 - p_b_effective) * p_a_effective
        prob_q_plus[-1] = 0.0
        prob_q_minus[0] = 0.0
        
        # Reward
        reward = p_b_effective * b + p_a_effective * a
        
        # Expectation
        exp_val = prob_q_plus * V_next_plus + prob_q_minus * V_next_minus
        prob_q = 1.0 - prob_q_plus - prob_q_minus
        exp_val += prob_q * V_next
        
        V_curr = reward - risk + exp_val
        
        V_next = V_curr
        # Update shifts for next iteration
        V_next_plus[:-1] = V_next[1:]
        V_next_minus[1:] = V_next[:-1]

    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    # Determine step index
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    
    with _TABLE_LOCK:
        if key not in _TABLE_CACHE:
            # Precompute
            start_time = time.time()
            opt_bid, opt_ask = _precompute_regime(
                params['sigma'], params['A'], params['k'], params['gamma'], 
                params['T'], params['steps'], params['Q']
            )
            _TABLE_CACHE[key] = (opt_bid, opt_ask)
            
            # Time guard (safety)
            if time.time() - start_time > 290:
                 depth = 1.0 / params["k"]
                 return np.full(len(inventory), depth), np.full(len(inventory), depth)

    opt_bid, opt_ask = _TABLE_CACHE[key]
    
    q_indices = np.asarray(inventory) + params['Q']
    
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    
    # Ensure valid outputs: no NaN, no -inf, no values below -3/k
    min_allowed = -3.0 / params['k']
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf) | (bid_depths < min_allowed), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf) | (ask_depths < min_allowed), np.inf, ask_depths)
    
    return bid_depths, ask_depths
