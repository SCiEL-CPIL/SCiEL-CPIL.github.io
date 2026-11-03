import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _solve_qp(S_target):
    """
    Solve the Convex QP:
    Minimize M
    Subject to:
    sum(h) = 1
    h >= 0
    sum(h^2) >= S_target
    (h*h)_k <= M for all k
    """
    # Variables: h (N), M (1)
    # x = [h_0, ..., h_{N-1}, M]
    
    # Initial guess: Uniform
    h0 = np.ones(N) / N
    M0 = np.max(np.convolve(h0, h0, mode='full'))
    x0 = np.concatenate([h0, [M0]])
    
    # Objective: Minimize M (last element)
    def objective(x):
        return x[-1]
        
    # Constraints
    # 1. Sum h = 1 (Equality)
    cons_eq = {'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - 1.0}
    
    # 2. Sum h^2 >= S_target  =>  S_target - sum(h^2) <= 0
    cons_sq = {'type': 'ineq', 'fun': lambda x: S_target - np.sum(x[:N]**2)}
    
    # 3. (h*h)_k <= M  =>  M - (h*h)_k >= 0
    # We need to define these for all k.
    # This will be many constraints.
    # N=50, so h*h has 2N-1 = 99 terms.
    # We can vectorize the constraint function if possible, but SLSQP expects scalar fun.
    # We can define a single function that returns the minimum of (M - (h*h)_k) for all k.
    # If min(M - peak) >= 0, then all are satisfied.
    cons_peak = {'type': 'ineq', 'fun': lambda x: x[-1] - np.max(np.convolve(x[:N], x[:N], mode='full'))}
    
    constraints = [cons_eq, cons_sq, cons_peak]
    
    bounds = [(0, None) for _ in range(N)] + [(0, None)]
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-9})
        if res.success:
            h_sol = res.x[:N]
            # Check if it actually satisfies the S_target constraint (numerical issues)
            if np.sum(h_sol**2) < S_target - 1e-6:
                return None
            return h_sol
    except Exception:
        pass
    return None

def construct():
    # Sweep S_target from low to high
    # Uniform: S = 1/N = 0.02
    # Spike: S = 1.0
    # We expect the optimal S to be somewhere in between.
    
    best_h = None
    best_score = 0.0
    
    # Start with uniform
    h_uni = np.ones(N) / N
    best_h = h_uni
    best_score = _score_raw(h_uni)
    
    # Sweep S_target
    # Use a grid of S values
    S_values = np.linspace(1.0/N, 1.0, 20)
    
    for S in S_values:
        h_sol = _solve_qp(S)
        if h_sol is not None:
            score = _score_raw(h_sol)
            if score > best_score:
                best_score = score
                best_h = h_sol
                
    # Refine around the best S found
    # Find the index of the best S
    # For simplicity, just do a few more points around the best one if we tracked it.
    # Here we just return the best found.
    
    return list(best_h)
