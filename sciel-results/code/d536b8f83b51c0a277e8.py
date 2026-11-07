import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def compute_value(h):
    """Compute the exact value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        s = h.sum()
        if abs(s - target) < 1e-13:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if abs(s - target) > 1e-11:
        diff = target - s
        free = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    i = np.arange(N)
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    # Formulate: minimize t s.t. c_m(h) <= t for all m, sum(h)=N/2, 0<=h<=1
    # Variables: [h_0, ..., h_{N-1}, t]
    # c_m = sum_i h[i] * (1 - h[i+m]) for valid i
    
    def objective(v):
        h = v[:N]
        t = v[N]
        return t
    
    def constraints_func(v):
        h = v[:N]
        t = v[N]
        c = np.correlate(h, 1.0 - h, mode="full")
        # c_m - t <= 0 for all m
        return c - t
    
    # Use SLSQP with inequality constraints
    # Inequality: c_m - t <= 0 for all m (199 constraints)
    # Equality: sum(h) - N/2 = 0
    
    def make_constraints():
        # Inequality constraints: c_m - t <= 0
        def ineq_fun(v):
            h = v[:N]
            t = v[N]
            c = np.correlate(h, 1.0 - h, mode="full")
            return c - t
        
        # Equality constraint: sum(h) - N/2 = 0
        def eq_fun(v):
            return v[:N].sum() - N / 2.0
        
        return [
            {'type': 'ineq', 'fun': ineq_fun},
            {'type': 'eq', 'fun': eq_fun}
        ]
    
    constraints = make_constraints()
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)]
    
    # Multiple starting points
    starts = []
    
    # Start 1: uniform
    starts.append(np.full(N + 1, 0.5))
    
    # Start 2: bimodal alternating
    for phase in [0, 1]:
        h_bimod = np.where((np.arange(N) + phase) % 2 == 0, 0.0, 1.0)
        h_bimod = project(h_bimod)
        c = np.correlate(h_bimod, 1.0 - h_bimod, mode="full")
        t0 = float(np.max(c))
        starts.append(np.concatenate([h_bimod, [t0 + 0.1]]))
    
    # Start 3: sinusoidal patterns
    for k in [1, 2, 3, 5, 10]:
        for amp in [0.1, 0.2, 0.3, 0.4]:
            h_sin = 0.5 + amp * np.sin(2 * np.pi * k * i / N)
            h_sin = project(h_sin)
            c = np.correlate(h_sin, 1.0 - h_sin, mode="full")
            t0 = float(np.max(c))
            starts.append(np.concatenate([h_sin, [t0 + 0.1]]))
    
    # Start 4: random
    np.random.seed(42)
    for _ in range(5):
        h_rand = np.random.rand(N)
        h_rand = project(h_rand)
        c = np.correlate(h_rand, 1.0 - h_rand, mode="full")
        t0 = float(np.max(c))
        starts.append(np.concatenate([h_rand, [t0 + 0.1]]))
    
    for v0 in starts:
        if remaining() < 10:
            break
        
        try:
            res = minimize(
                objective,
                v0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < best_val:
                h_opt = res.x[:N]
                h_opt = project(h_opt)
                val = compute_value(h_opt)
                
                if val < best_val:
                    best_val = val
                    best_h = h_opt
                    
        except Exception:
            pass
    
    return best_h
