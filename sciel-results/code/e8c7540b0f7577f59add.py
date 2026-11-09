import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 260.0
_CACHE = {}

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_indices = np.arange(-Q, Q + 1)
    
    # Value function V[t, q_idx]
    V = np.zeros((N + 1, n_q))
    # Optimal depths opt_b[t, q_idx], opt_a[t, q_idx]
    opt_b = np.zeros((N, n_q))
    opt_a = np.zeros((N, n_q))
    
    # Grid for depths
    # Range 0 to 5/k
    max_depth = 5.0 / k
    grid_size = 40
    depths = np.linspace(0, max_depth, grid_size)
    # Precompute P(d) for grid
    P_grid = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    
    for t in range(N - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 10:
            # Fallback to seed if too slow
            for qi in range(n_q):
                opt_b[t, qi] = 1.0 / k
                opt_a[t, qi] = 1.0 / k
            return opt_b, opt_a
        
        V_next = V[t + 1, :]
        for qi in range(n_q):
            q = q_indices[qi]
            
            # Get value differences
            if qi > 0:
                delta_minus = V_next[qi - 1] - V_next[qi]
            else:
                delta_minus = 0.0 # q = -Q, no ask fill possible, so this term is 0 in effect?
                # Actually if q=-Q, P_ask is effectively 0.
                
            if qi < n_q - 1:
                delta_plus = V_next[qi + 1] - V_next[qi]
            else:
                delta_plus = 0.0 # q = Q, no bid fill possible.
            
            best_val = -np.inf
            best_b = 1.0 / k
            best_a = 1.0 / k
            
            # If q == Q, bid fill prob is 0. So b doesn't matter for reward/transition.
            # We can set b=0 and optimize a.
            # If q == -Q, ask fill prob is 0. Set a=0 and optimize b.
            
            if q == Q:
                # Optimize a only, b=0
                # J(a) = a P(a) + P(a) delta_minus + C
                # P(b)=0, so terms with P(b) vanish.
                # E[V] = (1-0)(1-P(a))V(q) + 0 + (1-0)P(a)V(q-1) + 0 = (1-P(a))V(q) + P(a)V(q-1)
                # Reward = a P(a)
                # Total = a P(a) + (1-P(a))V(q) + P(a)V(q-1)
                # = a P(a) + V(q) + P(a)(V(q-1) - V(q))
                # = a P(a) + V(q) + P(a) delta_minus
                for i_a in range(grid_size):
                    a = depths[i_a]
                    p_a = P_grid[i_a]
                    val = a * p_a + V_next[qi] + p_a * delta_minus
                    if val > best_val:
                        best_val = val
                        best_a = a
                opt_b[t, qi] = 0.0
                opt_a[t, qi] = best_a
                V[t, qi] = best_val
            elif q == -Q:
                # Optimize b only, a=0
                for i_b in range(grid_size):
                    b = depths[i_b]
                    p_b = P_grid[i_b]
                    val = b * p_b + V_next[qi] + p_b * delta_plus
                    if val > best_val:
                        best_val = val
                        best_b = b
                opt_b[t, qi] = best_b
                opt_a[t, qi] = 0.0
                V[t, qi] = best_val
            else:
                # General case
                # Maximize over b, a
                # J(b, a) = b P(b) + a P(a) + V(q) + P(b) delta_plus + P(a) delta_minus + P(b)P(a)(-delta_plus - delta_minus)
                # We can loop over a and find best b, or 2D loop.
                # 2D loop is 40x40=1600.
                for i_a in range(grid_size):
                    a = depths[i_a]
                    p_a = P_grid[i_a]
                    # Term depending on b:
                    # b P(b) + P(b) delta_plus + P(b) p_a (-delta_plus - delta_minus)
                    # = P(b) [ b + delta_plus - p_a(delta_plus + delta_minus) ]
                    # Constant part (w.r.t b):
                    # a p_a + V(q) + p_a delta_minus
                    const_a = a * p_a + V_next[qi] + p_a * delta_minus
                    coeff_b = delta_plus - p_a * (delta_plus + delta_minus)
                    
                    # Maximize P(b) * (b + coeff_b) + const_a
                    # P(b) is monotonic decreasing. b is increasing.
                    # The product is unimodal.
                    # We can just loop over b grid.
                    for i_b in range(grid_size):
                        b = depths[i_b]
                        p_b = P_grid[i_b]
                        val = const_a + p_b * (b + coeff_b)
                        if val > best_val:
                            best_val = val
                            best_b = b
                            best_a = a
                opt_b[t, qi] = best_b
                opt_a[t, qi] = best_a
                V[t, qi] = best_val
                
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
    
    t_idx = int(t / (T / steps))
    if t_idx >= steps:
        t_idx = steps - 1
        
    # Map inventory to index
    # inventory is in [-Q, Q]
    q_idx = inventory + Q
    
    b_depths = opt_b[t_idx, q_idx]
    a_depths = opt_a[t_idx, q_idx]
    
    return b_depths, a_depths
