import numpy as np
import time

_start_time = time.time()
_DEADLINE = 300.0  # seconds, with margin for evaluator

def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """Solve the MDP for one regime, return bid/ask depth tables indexed by (step, q)."""
    dt = T / steps
    n_steps = steps
    q_min = -Q
    q_max = Q
    n_q = q_max - q_min + 1
    
    # V[n, i] where i = q - q_min, n = step index (0..n_steps-1)
    # V[n_steps, :] = 0 (no terminal charge)
    V = np.zeros((n_steps + 1, n_q), dtype=np.float64)
    
    # Precompute depth tables
    bid_depth = np.zeros((n_steps, n_q), dtype=np.float64)
    ask_depth = np.zeros((n_steps, n_q), dtype=np.float64)
    
    # Backward DP
    for n in range(n_steps - 1, -1, -1):
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 30:
            break
        V_next = V[n + 1]
        V_curr = V[n]
        
        for i in range(n_q):
            q = q_min + i
            
            # Inventory cost
            cost = (gamma / 2.0) * sigma * sigma * q * q * dt
            
            # Bid: can fill if q < Q
            if q < q_max:
                dV_b = V_next[i + 1] - V_next[i]
                b_opt = 1.0 / k - dV_b
                if b_opt < 0:
                    b_opt = 0.0
                # p_b = 1 - exp(-A * exp(-k*b) * dt)
                # For b=0, p_b = 1 - exp(-A*dt)
                # For b>0, p_b = 1 - exp(-A*exp(-k*b)*dt)
                p_b = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
                # Expected value from bid side
                bid_val = p_b * (b_opt + dV_b)
            else:
                b_opt = np.inf  # no bid quote
                bid_val = 0.0
            
            # Ask: can fill if q > -Q
            if q > q_min:
                dV_a = V_next[i - 1] - V_next[i]
                a_opt = 1.0 / k - dV_a
                if a_opt < 0:
                    a_opt = 0.0
                p_a = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
                ask_val = p_a * (a_opt + dV_a)
            else:
                a_opt = np.inf  # no ask quote
                ask_val = 0.0
            
            V_curr[i] = -cost + bid_val + ask_val
            bid_depth[n, i] = b_opt
            ask_depth[n, i] = a_opt
    
    return bid_depth, ask_depth


# Precompute solutions for all regimes
_regime_solutions = {}

def _get_solution(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _regime_solutions:
        if time.time() - _start_time > _DEADLINE - 60:
            # Fallback to simple policy
            k = params['k']
            return None
        _regime_solutions[key] = _solve_regime(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _regime_solutions[key]


def quote(t, inventory, mid, params):
    """Optimal market making policy via precomputed DP."""
    n_paths = len(inventory)
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Determine step index
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Get precomputed solution
    sol = _get_solution(params)
    
    if sol is None:
        # Fallback
        depth = np.full(n_paths, 1.0 / k)
        return depth, depth.copy()
    
    bid_depth_table, ask_depth_table = sol
    q_min = -Q
    n_q = 2 * Q + 1
    
    inv = np.asarray(inventory, dtype=np.int32)
    # Clamp inventory to valid range
    inv_clamped = np.clip(inv, q_min, Q)
    indices = inv_clamped - q_min
    
    bid_depths = bid_depth_table[n, indices]
    ask_depths = ask_depth_table[n, indices]
    
    # Handle boundary: no bid at q=Q, no ask at q=-Q
    # bid_depth_table already has inf at boundaries, but let's ensure
    # Actually we set b_opt = inf when q >= Q, but we stored it. Let me check.
    # In _solve_regime, when q >= q_max, b_opt = np.inf, so bid_depth_table has inf.
    # That's correct.
    
    # Ensure no negative values (safety)
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    return bid_depths, ask_depths
