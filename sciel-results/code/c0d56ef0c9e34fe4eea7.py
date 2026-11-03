import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    # Initial guess: Uniform
    h = np.ones(N) / N
    
    # Objective to maximize: (sum h^2) / (sum h * max(conv))
    # We minimize the negative of this.
    # Constraint: sum h = 1, h >= 0
    
    def objective(h):
        if np.any(h < 0):
            return 1e9
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv == 0:
            return 1e9
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 == 0:
            return 1e9
        ratio = l2_sq / (l1 * max_conv)
        return -ratio

    def constraint_sum(h):
        return np.sum(h) - 1.0

    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    bounds = [(0.0, 1.0)] * N

    best_h = h.copy()
    best_score = -objective(h)

    # Use multiple starting points to avoid local minima
    # 1. Uniform
    # 2. Single spike
    # 3. Random sparse
    starts = [
        np.ones(N) / N,
        np.eye(N, N)[0], # Spike at 0
        np.eye(N, N)[N//2], # Spike at center
    ]
    
    # Add some random starts
    for _ in range(5):
        r = np.random.rand(N)
        r = r / np.sum(r)
        starts.append(r)

    for h0 in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-10})
            if res.success:
                h_res = res.x
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_h = h_res
        except Exception:
            pass

    # Local refinement with block coordinate descent if time permits
    if time.time() < deadline - 10:
        for _ in range(100):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                # Optimize h[i] while keeping others fixed
                # This is a 1D problem
                h_test = best_h.copy()
                # We need to maintain sum=1. 
                # Actually, SLSQP already handled constraints. 
                # Let's just do a simple line search or coordinate step.
                pass # Skipping complex coordinate descent for brevity and safety

    # Ensure non-negative and normalized
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) > 0:
        best_h = best_h / np.sum(best_h)
    else:
        best_h = np.ones(N) / N

    return list(best_h)
