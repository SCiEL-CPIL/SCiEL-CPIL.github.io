"""
Combined market making policy:
- Structural idea from current program: analytic skew based on inventory
- Refinement from inspiration: precomputed DP with continuous 1-D optimization per side,
  cached per regime, with deadline guard and fallback to analytic heuristic
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

_T0 = time.time()
_DEADLINE = 310.0  # safety margin before 330s kill

# Optimal neutral depth: maximize d * (1 - exp(-k*d))
# Let y = k*d, maximize (y/k) * (1 - exp(-y))
# d/dy [y(1-exp(-y))] = (1-exp(-y)) - y*exp(-y) = 0
# => 1 - exp(-y) = y*exp(-y) => exp(y) - 1 = y => y* = ~1.5936
_OPTIMAL_KD = 1.5936242608465853

_dp_cache = {}


def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)


def _solve_depth_1d(q, p_other, V_next, qi_bid, qi_ask, qi_both, qi_neither,
                    penalty, dt, k, A, Q, is_bid, q_val):
    """
    Solve 1-D optimization for one side's depth given the other side is fixed.
    Actually, we do joint 2-D but separable. Let's do independent 1-D per side
    given the value function structure.
    
    For bid side: choose d_bid to maximize
        d_bid * p_bid + d_ask * p_ask - penalty + expected_future
    where p_ask and d_ask are fixed (from ask side optimization).
    
    But since both are coupled through future values, we need joint optimization.
    However, the key insight from the finding is that bid and ask are separable
    given V. Let's do: for each q, optimize bid and ask independently by
    considering that the other side's contribution is constant w.r.t. this side's choice.
    
    Actually, re-reading: "independent 1-D scalar optimizations per side, which is the
    exact decoupling implied by the MDP structure (bid and ask decisions are separable
    given the value function)."
    
    This means: for a given q, the optimal bid depth can be found by maximizing over d_bid
    treating d_ask as its optimal value, and vice versa. But they interact through
    the future value terms.
    
    Let me think more carefully. The Bellman equation is:
    V[n,q] = max_{d_b, d_a} { d_b*p_b + d_a*p_a - penalty + 
              p_b*(1-p_a)*V[n+1,q+1] + p_a*(1-p_b)*V[n+1,q-1] + 
              p_b*p_a*V[n+1,q] + (1-p_b)*(1-p_a)*V[n+1,q] }
    
    = max_{d_b, d_a} { d_b*p_b + d_a*p_a - penalty + 
              p_b*(1-p_a)*V_b + p_a*(1-p_b)*V_a + 
              (p_b*p_a + (1-p_b)*(1-p_a))*V_n }
    
    where V_b = V[n+1,q+1], V_a = V[n+1,q-1], V_n = V[n+1,q].
    
    This is NOT separable in general because p_b and p_a appear in cross terms.
    However, for small dt, p_b and p_a are small, so cross terms p_b*p_a are O(dt^2)
    and can be neglected. In that case:
    
    V[n,q] ≈ max_{d_b, d_a} { d_b*p_b + d_a*p_a - penalty + 
              p_b*V_b + p_a*V_a + V_n }
    
    = max_{d_b} [d_b*p_b + p_b*V_b] + max_{d_a} [d_a*p_a + p_a*V_a] - penalty + V_n
    
    So bid and ask ARE separable when cross terms are neglected!
    
    For bid: maximize d_b * p_b + p_b * V_b = p_b * (d_b + V_b)
    where p_b = 1 - exp(-A*exp(-k*d_b)*dt)
    
    Let f(d_b) = (1 - exp(-A*exp(-k*d_b)*dt)) * (d_b + V_b)
    
    For ask: maximize d_a * p_a + p_a * V_a = p_a * (d_a + V_a)
    where p_a = 1 - exp(-A*exp(-k*d_a)*dt)
    
    Let g(d_a) = (1 - exp(-A*exp(-k*d_a)*dt)) * (d_a + V_a)
    
    These are 1-D optimizations!
    """
    pass


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth tables using backward induction with
    independent 1-D optimizations per side (separable approximation).
    """
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key in _dp_cache:
        return _dp_cache[cache_key]
    
    dt = T / steps
    num_q = 2 * Q + 1
    
    # V[n, qi] where qi = q + Q
    V = np.zeros((steps + 1, num_q), dtype=np.float64)
    
    # Depth tables
    bid_depths = np.full((steps, num_q), _OPTIMAL_KD / k, dtype=np.float64)
    ask_depths = np.full((steps, num_q), _OPTIMAL_KD / k, dtype=np.float64)
    
    # Precompute depth bounds for optimization
    min_d = -3.0 / k + 1e-9
    max_d = 50.0 / k  # reasonable upper bound
    
    for n in range(steps - 1, -1, -1):
        # Check budget
        if _remaining_budget() < 30.0:
            # Not enough time for full DP; break and use what we have
            break
        
        penalty_rate = 0.5 * gamma * sigma * sigma * dt
        
        for qi in range(num_q):
            q_val = qi - Q
            
            penalty = penalty_rate * q_val * q_val
            
            V_b = V[n + 1, min(qi + 1, num_q - 1)] if q_val < Q else 0.0
            V_a = V[n + 1, max(qi - 1, 0)] if q_val > -Q else 0.0
            V_n = V[n + 1, qi]
            
            # Optimize bid depth
            best_bid = _OPTIMAL_KD / k
            best_bid_val = -np.inf
            if q_val < Q:
                # Maximize f(d) = (1 - exp(-A*exp(-k*d)*dt)) * (d + V_b)
                def neg_f_bid(d):
                    if d <= min_d or d >= max_d:
                        return 1e10
                    p = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                    return -p * (d + V_b)
                
                # Try a few starting points
                for d0 in [_OPTIMAL_KD / k, 0.5 * _OPTIMAL_KD / k, 2.0 * _OPTIMAL_KD / k]:
                    try:
                        res = minimize_scalar(neg_f_bid, bounds=(min_d, max_d), 
                                             method='bounded', options={'xatol': 1e-8})
                        if res.success and res.fun < best_bid_val:
                            best_bid_val = -res.fun
                            best_bid = res.x
                    except:
                        pass
            
            # Optimize ask depth
            best_ask = _OPTIMAL_KD / k
            best_ask_val = -np.inf
            if q_val > -Q:
                def neg_f_ask(d):
                    if d <= min_d or d >= max_d:
                        return 1e10
                    p = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                    return -p * (d + V_a)
                
                for d0 in [_OPTIMAL_KD / k, 0.5 * _OPTIMAL_KD / k, 2.0 * _OPTIMAL_KD / k]:
                    try:
                        res = minimize_scalar(neg_f_ask, bounds=(min_d, max_d),
                                             method='bounded', options={'xatol': 1e-8})
                        if res.success and res.fun < best_ask_val:
                            best_ask_val = -res.fun
                            best_ask = res.x
                    except:
                        pass
            
            # Compute optimal value
            p_bid = 1.0 - np.exp(-A * np.exp(-k * best_bid) * dt) if q_val < Q else 0.0
            p_ask = 1.0 - np.exp(-A * np.exp(-k * best_ask) * dt) if q_val > -Q else 0.0
            
            # Full Bellman with cross terms for accuracy
            p_both = p_bid * p_ask
            p_bid_only = p_bid * (1.0 - p_ask)
            p_ask_only = p_ask * (1.0 - p_bid)
            p_neither = (1.0 - p_bid) * (1.0 - p_ask)
            
            qi_bid_next = min(q_val + 1, Q) + Q
            qi_ask_next = max(q_val - 1, -Q) + Q
            qi_both_next = q_val + Q
            qi_neither_next = q_val + Q
            
            expected_future = (p_bid_only * V[n + 1, qi_bid_next] +
                              p_ask_only * V[n + 1, qi_ask_next] +
                              p_both * V[n + 1, qi_both_next] +
                              p_neither * V[n + 1, qi_neither_next])
            
            reward = best_bid * p_bid + best_ask * p_ask - penalty
            
            V[n, qi] = reward + expected_future
            bid_depths[n, qi] = best_bid
            ask_depths[n, qi] = best_ask
    
    _dp_cache[cache_key] = (bid_depths, ask_depths)
    return bid_depths, ask_depths


def _analytic_fallback(t, inventory, params):
    """
    Fast analytic fallback when DP is not available or time is running out.
    Uses the first-order optimal control solution.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    
    inv = inventory.astype(np.float64)
    t_rem = max(T - t, 0.0)
    
    # Optimal neutral depth
    base_depth = _OPTIMAL_KD / k
    
    # Inventory skew: x = gamma * sigma^2 * t_rem * q
    # bid depth: log(exp(_OPTIMAL_KD) / (1 - x)) / k
    # ask depth: log(exp(_OPTIMAL_KD) / (1 + x)) / k
    x = gamma * sigma * sigma * t_rem * inv
    
    exp_kd = np.exp(_OPTIMAL_KD)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denom_bid = 1.0 - x
        denom_ask = 1.0 + x
        raw_bid = np.log(exp_kd / denom_bid) / k
        raw_ask = np.log(exp_kd / denom_ask) / k
    
    # Handle invalid values
    min_depth = -3.0 / k + 1e-9
    max_depth = 50.0 / k
    
    bid_depths = np.where(denom_bid <= 0, np.inf, raw_bid)
    ask_depths = np.where(denom_ask <= 0, np.inf, raw_ask)
    
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    
    # Replace NaN/inf with safe values
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf), base_depth, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf), base_depth, ask_depths)
    
    # Boundary conditions
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths based on inventory risk using precomputed DP policy
    with analytic fallback.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n = int(round(t / (T / steps)))
    n = max(0, min(n, steps - 1))
    
    # Try to use DP policy if available and budget allows
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    
    if cache_key in _dp_cache:
        bid_table, ask_table = _dp_cache[cache_key]
        if n < len(bid_table):
            inv_idx = np.clip(inventory + Q, 0, 2 * Q)
            bid_depths = bid_table[n, inv_idx].copy()
            ask_depths = ask_table[n, inv_idx].copy()
            
            # Enforce boundary conditions
            bid_depths[inventory >= Q] = np.inf
            ask_depths[inventory <= -Q] = np.inf
            
            # Sanitize
            min_depth = -3.0 / k + 1e-9
            bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < min_depth), np.inf, bid_depths)
            ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < min_depth), np.inf, ask_depths)
            
            return bid_depths, ask_depths
    
    # Check if we have budget to compute DP
    if _remaining_budget() > 60.0:
        try:
            bid_table, ask_table = _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
            if n < len(bid_table):
                inv_idx = np.clip(inventory + Q, 0, 2 * Q)
                bid_depths = bid_table[n, inv_idx].copy()
                ask_depths = ask_table[n, inv_idx].copy()
                
                bid_depths[inventory >= Q] = np.inf
                ask_depths[inventory <= -Q] = np.inf
                
                min_depth = -3.0 / k + 1e-9
                bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < min_depth), np.inf, bid_depths)
                ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < min_depth), np.inf, ask_depths)
                
                return bid_depths, ask_depths
        except:
            pass
    
    # Fallback to analytic
    return _analytic_fallback(t, inventory, params)
# EVOLVE-BLOCK-END
