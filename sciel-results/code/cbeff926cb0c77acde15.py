"""Market making with inventory risk - optimal policy via dynamic programming."""
import numpy as np

_policy_cache = {}

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """Compute optimal bid/ask depths for each (step, inventory) via backward DP."""
    dt = T / steps
    n_grid = 50
    depth_grid = np.linspace(0.005, 1.5, n_grid)
    
    V = np.zeros((steps + 1, 2 * Q + 1))
    opt_bid = np.zeros((steps, 2 * Q + 1))
    opt_ask = np.zeros((steps, 2 * Q + 1))
    
    for n in range(steps - 1, -1, -1):
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            inv_penalty = 0.5 * gamma * sigma**2 * q**2 * dt
            
            b_grid, a_grid = np.meshgrid(depth_grid, depth_grid)
            b_flat = b_grid.flatten()
            a_flat = a_grid.flatten()
            
            if q == Q:
                p_bid = np.zeros(len(b_flat))
            else:
                p_bid = 1.0 - np.exp(-A * np.exp(-k * b_flat) * dt)
            
            if q == -Q:
                p_ask = np.zeros(len(a_flat))
            else:
                p_ask = 1.0 - np.exp(-A * np.exp(-k * a_flat) * dt)
            
            vals = np.full(len(b_flat), -inv_penalty)
            vals += p_bid * p_ask * (b_flat + a_flat + V[n + 1, q_idx])
            if q < Q:
                vals += p_bid * (1 - p_ask) * (b_flat + V[n + 1, q_idx + 1])
            if q > -Q:
                vals += (1 - p_bid) * p_ask * (a_flat + V[n + 1, q_idx - 1])
            vals += (1 - p_bid) * (1 - p_ask) * V[n + 1, q_idx]
            
            best_idx = np.argmax(vals)
            V[n, q_idx] = vals[best_idx]
            opt_bid[n, q_idx] = b_flat[best_idx]
            opt_ask[n, q_idx] = a_flat[best_idx]
    
    return opt_bid, opt_ask


def _get_policy(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_policy(
            params["sigma"], params["A"], params["k"], params["gamma"],
            params["T"], params["steps"], params["Q"]
        )
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    opt_bid, opt_ask = _get_policy(params)
    dt = params["T"] / params["steps"]
    n = int(round(t / dt))
    n = min(n, params["steps"] - 1)
    Q = params["Q"]
    inv_clipped = np.clip(inventory, -Q, Q)
    q_indices = (inv_clipped + Q).astype(int)
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    return bid_depths, ask_depths
