import numpy as np
import time
import threading

_TABLE_CACHE = {}
_TABLE_LOCK = threading.Lock()

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def _check_deadline():
    return time.time() >= _DEADLINE

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    N = steps
    max_q = Q
    min_q = -Q
    n_states = 2 * Q + 1
    
    # Precompute interpolation grid for the optimality condition
    # phi(x) = g(x)/h(x) - x
    # g(x) = 1 - exp(-A exp(-kx) dt)
    # h(x) = k A dt exp(-kx) exp(-A exp(-kx) dt)
    # We solve phi(x) = C for x
    
    x_max = 10.0 / k
    n_grid = 10000
    x_grid = np.linspace(0, x_max, n_grid)
    
    # Calculate phi(x)
    exp_kx = np.exp(-k * x_grid)
    exp_Au = np.exp(-A * exp_kx * dt)
    g = 1.0 - exp_Au
    h = k * A * dt * exp_kx * exp_Au
    
    # Avoid division by zero if h is very small (large x)
    # At x_max, h is small, g is small.
    # phi(x) ~ 1/k - x.
    # We can just compute it.
    phi = g / h - x_grid
    
    # phi is decreasing.
    # For np.interp, we need xp to be increasing.
    # So we use -phi as xp and x_grid as fp.
    # We want to find x given C.
    # -phi(x) = -C.
    # x = interp(-C, -phi, x_grid)
    
    neg_phi = -phi
    # Ensure neg_phi is strictly increasing? 
    # phi is strictly decreasing, so neg_phi is strictly increasing.
    
    V_next = np.zeros(n_states)
    opt_bid = np.zeros((N, n_states))
    opt_ask = np.zeros((N, n_states))
    
    # Precompute constants for speed
    # Not much to precompute since we use vectorized ops
    
    for n in range(N - 1, -1, -1):
        if _check_deadline():
            depth_default = 1.0 / k
            for n_fill in range(n, N):
                opt_bid[n_fill, :] = depth_default
                opt_ask[n_fill, :] = depth_default
            return opt_bid, opt_ask
        
        V_curr = np.zeros(n_states)
        
        # Initialize b and a for fixed point
        b = np.zeros(n_states)
        a = np.zeros(n_states)
        
        # Use a convergence tolerance that is slightly tighter for better precision
        tol = 1e-12
        max_iter = 200
        
        for _iter in range(max_iter):
            if _check_deadline():
                 break
                
            # Compute p_a from current a
            p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
            
            # Compute C_b for each q
            # C_b = (1 - p_a) * Delta_plus - p_a * Delta_minus
            # Delta_plus = V_next[q+1] - V_next[q]
            # Delta_minus = V_next[q-1] - V_next[q]
            
            delta_plus = np.zeros(n_states)
            delta_minus = np.zeros(n_states)
            
            # q < Q => idx < 2Q
            # V_next[idx+1] - V_next[idx]
            delta_plus[:-1] = V_next[1:] - V_next[:-1]
            # q > -Q => idx > 0
            # V_next[idx-1] - V_next[idx]
            delta_minus[1:] = V_next[:-1] - V_next[1:]
            
            C_b = (1.0 - p_a) * delta_plus - p_a * delta_minus
            
            # Solve for b: phi(b) = C_b
            # b = interp(-C_b, neg_phi, x_grid)
            # Clip C_b to valid range?
            # If C_b > phi(0), b should be 0.
            # If C_b < phi(x_max), b should be x_max.
            # np.interp handles this by clamping to the ends of the xp array.
            # xp = neg_phi. fp = x_grid.
            # If -C_b < neg_phi[0] (i.e. C_b > phi(0)), result is x_grid[0] = 0.
            # If -C_b > neg_phi[-1] (i.e. C_b < phi(x_max)), result is x_grid[-1] = x_max.
            
            b_new = np.interp(-C_b, neg_phi, x_grid)
            
            # Update p_b
            p_b = 1.0 - np.exp(-A * np.exp(-k * b_new) * dt)
            
            # Compute C_a
            # C_a = (1 - p_b) * Delta_minus - p_b * Delta_plus
            
            C_a = delta_minus * (1.0 - p_b) - p_b * delta_plus
            
            a_new = np.interp(-C_a, neg_phi, x_grid)
            
            # Convergence check
            if np.allclose(b_new, b, atol=tol) and np.allclose(a_new, a, atol=tol):
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
        
        # Transitions
        # prob_q_plus = p_b * (1 - p_a)
        # prob_q_minus = (1 - p_b) * p_a
        # prob_q = 1 - prob_q_plus - prob_q_minus
        
        prob_q_plus = p_b * (1.0 - p_a)
        prob_q_minus = (1.0 - p_b) * p_a
        
        # Handle boundaries
        # q = Q (idx = 2Q): no bid fill. prob_q_plus = 0.
        # q = -Q (idx = 0): no ask fill. prob_q_minus = 0.
        
        prob_q_plus[-1] = 0.0
        prob_q_minus[0] = 0.0
        
        # Reward
        reward = p_b * b + p_a * a
        reward[-1] = p_a[-1] * a[-1] # Only ask fill possible
        reward[0] = p_b[0] * b[0]     # Only bid fill possible
        
        # Risk
        q_vals = np.arange(n_states) - Q
        risk = (gamma / 2.0) * (sigma**2) * (q_vals**2) * dt
        
        # Expectation
        # V_next[q+1] for q < Q
        # V_next[q-1] for q > -Q
        
        V_next_plus = np.zeros(n_states)
        V_next_minus = np.zeros(n_states)
        V_next_plus[:-1] = V_next[1:]
        V_next_minus[1:] = V_next[:-1]
        
        # For boundaries, the probability is 0, so the value doesn't matter, 
        # but to be safe, we can set them to 0 or V_next[q].
        # Since prob is 0, term is 0.
        
        exp_val = prob_q_plus * V_next_plus + prob_q_minus * V_next_minus
        
        # The "stay" probability term:
        # prob_q = 1 - prob_q_plus - prob_q_minus
        # V_q = V_next[idx]
        V_q = V_next
        
        prob_q = 1.0 - prob_q_plus - prob_q_minus
        exp_val += prob_q * V_q
        
        V_curr = reward - risk + exp_val
        
        V_next = V_curr

    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    
    with _TABLE_LOCK:
        if key not in _TABLE_CACHE:
            start_time = time.time()
            opt_bid, opt_ask = _precompute_regime(
                params['sigma'], params['A'], params['k'], params['gamma'], 
                params['T'], params['steps'], params['Q']
            )
            _TABLE_CACHE[key] = (opt_bid, opt_ask)
            
            if time.time() - start_time > 290:
                 depth = 1.0 / params["k"]
                 return np.full(len(inventory), depth), np.full(len(inventory), depth)

    opt_bid, opt_ask = _TABLE_CACHE[key]
    
    q_indices = np.asarray(inventory) + params['Q']
    
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    
    min_allowed = -3.0 / params['k']
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf) | (bid_depths < min_allowed), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf) | (ask_depths < min_allowed), np.inf, ask_depths)
    
    return bid_depths, ask_depths
