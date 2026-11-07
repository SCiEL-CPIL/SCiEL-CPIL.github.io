"""Optimal market making with inventory risk using exact DP (fully vectorized)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_dp_cache = {}
_start_time = time.time()
_DEADLINE = 320.0  # seconds, safety margin before 330s kill

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

def _compute_dp_table(sigma, A, k, gamma, T, steps, Q):
    """Precompute DP table, fully vectorized."""
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1
    eps = 1e-15
    
    V = np.zeros((n_steps + 1, n_inv))
    bid_d = np.full((n_steps, n_inv), np.inf)
    ask_d = np.full((n_steps, n_inv), np.inf)
    
    p_max = 1.0 - np.exp(-A * dt)
    n_p = 80
    p_grid = np.linspace(0.0, p_max, n_p)
    p_safe = np.clip(p_grid, eps, p_max - eps)
    d_grid = -np.log(-np.log(1 - p_safe) / (A * dt)) / k
    d_grid[0] = 0.0
    
    for n in range(n_steps - 1, -1, -1):
        if _check_deadline():
            # Return what we have so far, fill rest with default
            for nn in range(n, n_steps):
                bid_d[nn] = 1.0 / k
                ask_d[nn] = 1.0 / k
            return bid_d, ask_d
        
        V_next = V[n + 1]
        q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
        inv_cost = (gamma / 2.0) * sigma * sigma * dt * q_vals * q_vals
        
        a1 = np.zeros(n_inv)
        a2 = np.zeros(n_inv)
        a1[:n_inv-1] = V_next[1:n_inv] - V_next[0:n_inv-1]
        a2[1:n_inv] = V_next[0:n_inv-1] - V_next[1:n_inv]
        
        V_new = np.zeros(n_inv)
        bid_new = np.full(n_inv, np.inf)
        ask_new = np.full(n_inv, np.inf)
        
        if n_inv > 2:
            int_slice = slice(1, n_inv - 1)
            a1_int = a1[int_slice]
            a2_int = a2[int_slice]
            n_int = len(a1_int)
            
            pb_col = p_grid[:, np.newaxis, np.newaxis]
            pa_row = p_grid[np.newaxis, :, np.newaxis]
            db_col = d_grid[:, np.newaxis, np.newaxis]
            da_row = d_grid[np.newaxis, :, np.newaxis]
            a1_exp = a1_int[np.newaxis, np.newaxis, :]
            a2_exp = a2_int[np.newaxis, np.newaxis, :]
            
            term1 = pb_col * (db_col + a1_exp)
            term2 = pa_row * (da_row + a2_exp)
            term3 = pb_col * pa_row * (a1_exp + a2_exp)
            
            val_3d = term1 + term2 - term3
            
            val_2d = val_3d.reshape(n_p * n_p, n_int)
            best_flat = np.argmax(val_2d, axis=0)
            best_i_pb = best_flat // n_p
            best_i_pa = best_flat % n_p
            
            V_new[int_slice] = val_2d[best_flat, np.arange(n_int)] - inv_cost[int_slice]
            bid_new[int_slice] = d_grid[best_i_pb]
            ask_new[int_slice] = d_grid[best_i_pa]
            
            zero_pb = best_i_pb == 0
            zero_pa = best_i_pa == 0
            if np.any(zero_pb):
                bid_new[int_slice][zero_pb] = np.inf
            if np.any(zero_pa):
                ask_new[int_slice][zero_pa] = np.inf
        
        if n_inv > 1:
            vals = p_grid * (d_grid + a1[0])
            best_i = np.argmax(vals)
            V_new[0] = vals[best_i] - inv_cost[0]
            bid_new[0] = np.inf if best_i == 0 else d_grid[best_i]
            ask_new[0] = np.inf
            
            vals = p_grid * (d_grid + a2[n_inv-1])
            best_i = np.argmax(vals)
            V_new[n_inv-1] = vals[best_i] - inv_cost[n_inv-1]
            ask_new[n_inv-1] = np.inf if best_i == 0 else d_grid[best_i]
            bid_new[n_inv-1] = np.inf
        
        V[n] = V_new
        bid_d[n] = bid_new
        ask_d[n] = ask_new
    
    return bid_d, ask_d


def _get_dp_tables(params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _dp_cache:
        _dp_cache[key] = _compute_dp_table(sigma, A, k, gamma, T, steps, Q)
    return _dp_cache[key]


def quote(t, inventory, mid, params):
    """Quote using precomputed DP table."""
    steps = params["steps"]
    Q = params["Q"]
    
    dt = params["T"] / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
    
    bid_table, ask_table = _get_dp_tables(params)
    
    q_indices = np.clip(inventory + Q, 0, 2 * Q)
    
    bid_depths = bid_table[n, q_indices].astype(np.float64)
    ask_depths = ask_table[n, q_indices].astype(np.float64)
    
    k_val = params["k"]
    bid_depths = np.where(bid_depths < -3.0 / k_val, np.inf, bid_depths)
    ask_depths = np.where(ask_depths < -3.0 / k_val, np.inf, ask_depths)
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
