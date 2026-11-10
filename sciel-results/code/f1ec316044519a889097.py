# EVOLVE-BLOCK-START
import numpy as np

# Precomputed optimal policy tables for each regime
# Regimes: (sigma, A, k, gamma)
_REGIMES = [
    (2.0, 140.0, 1.5, 0.05),
    (2.0, 140.0, 1.5, 0.5),
    (1.5, 50.0, 0.75, 0.2),
]

def _solve_depths_vectorized(deltas, alpha, k, is_bid=True):
    """
    Solve for optimal depths for an array of value differences (deltas).
    """
    d = np.full_like(deltas, 0.01)
    s = 1.0 if is_bid else -1.0
    
    for _ in range(10):
        exp_kd = np.exp(-k * d)
        exp_alpha_kd = np.exp(-alpha * exp_kd)
        p = 1.0 - exp_alpha_kd
        dp_dd = alpha * k * exp_kd * exp_alpha_kd
        
        grad = dp_dd * (s * d + delta) + p * s
        
        # Hessian: p''(d)(s d + delta) + 2 p'(d) s
        # p''(d) = p'(d) * (-k + alpha * k * exp_kd)
        p_dd = dp_dd * (-k + alpha * k * exp_kd)
        hessian = p_dd * (s * d + delta) + 2.0 * dp_dd * s
        
        hessian = np.where(np.abs(hessian) < 1e-12, 1e-12 * np.sign(hessian + 1e-30), hessian)
        
        step = -grad / hessian
        step = np.clip(step, -0.5, 0.5)
        d_new = d + step
        d_new = np.maximum(d_new, 0.0)
        
        if np.all(np.abs(d_new - d) < 1e-6):
            d = d_new
            break
        d = d_new
        
    return d

def _solve_regime(sigma, A, k, gamma, Q=30, T=1.0, steps=1000):
    dt = T / steps
    n_states = 2 * Q + 1
    q_min = -Q
    
    V = np.zeros(n_states)
    
    q_values = np.arange(q_min, q_min + n_states)
    risk_costs = (gamma / 2.0) * sigma**2 * q_values**2 * dt
    alpha = A * dt
    
    final_bid_depths = None
    final_ask_depths = None
    
    for n in range(steps - 1, -1, -1):
        V_qp = np.zeros(n_states)
        V_qm = np.zeros(n_states)
        
        V_qp[:-1] = V[1:]
        V_qm[1:] = V[:-1]
        
        delta_b = V_qp - V
        delta_a = V_qm - V
        
        bid_depths = _solve_depths_vectorized(delta_b, alpha, k, is_bid=True)
        ask_depths = _solve_depths_vectorized(delta_a, alpha, k, is_bid=False)
        
        bid_depths[-1] = np.inf
        ask_depths[0] = np.inf
        
        with np.errstate(over='ignore', invalid='ignore'):
            exp_kd_b = np.exp(-k * bid_depths)
            exp_kd_a = np.exp(-k * ask_depths)
            
            p_b = 1.0 - np.exp(-alpha * exp_kd_b)
            p_a = 1.0 - np.exp(-alpha * exp_kd_a)
        
        p_b[-1] = 0.0
        p_a[0] = 0.0
        
        term_b = np.where(p_b > 0, p_b * (bid_depths + V_qp), 0.0)
        term_a = np.where(p_a > 0, p_a * (-ask_depths + V_qm), 0.0)
        term_no_fill = (1.0 - p_b) * (1.0 - p_a) * V
        
        V_new = term_b + term_a + term_no_fill - risk_costs
        
        if n == 0:
            final_bid_depths = bid_depths.copy()
            final_ask_depths = ask_depths.copy()
            
        V = V_new

    return final_bid_depths, final_ask_depths

_POLICY_CACHE = {}
for (sigma, A, k, gamma) in _REGIMES:
    key = (sigma, A, k, gamma)
    _POLICY_CACHE[key] = _solve_regime(sigma, A, k, gamma)


def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    
    key = (sigma, A, k, gamma)
    if key not in _POLICY_CACHE:
        depth = 1.0 / k
        n = len(inventory)
        return np.full(n, depth), np.full(n, depth)
    
    bid_table, ask_table = _POLICY_CACHE[key]
    
    idx = inventory + Q
    idx = np.clip(idx, 0, 2 * Q)
    
    bid_depths = bid_table[idx]
    ask_depths = ask_table[idx]
    
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), 1.0/k, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), 1.0/k, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
