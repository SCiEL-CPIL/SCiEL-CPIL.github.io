import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Objective: minimize peak autoconvolution subject to L1 = 1, h >= 0
    # Score will be: sum(h^2) / (1 * peak_autoconv)
    
    def autoconv(h):
        # Full autoconvolution of h with itself
        # (h*h)[k] = sum_i h[i] * h[k-i] for valid i
        # Using np.convolve for full convolution
        return np.convolve(h, h, mode='full')
    
    def peak_autoconv(h):
        return np.max(autoconv(h))
    
    def objective(h):
        return peak_autoconv(h)
    
    def objective_grad(h):
        # Gradient of max_k sum_i h[i]*h[k-i]
        # This is complex; use numerical or analytic
        # For SLSQP, we can provide analytic gradient of the active constraint
        # But let's use a smoothed max or just let SLSQP handle it
        # Actually, let's compute gradient numerically for reliability
        grad = np.zeros_like(h)
        eps = 1e-7
        base = peak_autoconv(h)
        for i in range(len(h)):
            h_plus = h.copy()
            h_plus[i] += eps
            grad[i] = (peak_autoconv(h_plus) - base) / eps
            h_plus[i] = h
        return grad
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones_like(h)}
    ]
    bounds = [(0.0, None)] * N
    
    # Initial guess: uniform
    h0 = np.ones(N) / N
    
    best_h = h0.copy()
    best_score = -np.inf
    
    def compute_score(h):
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        l2_sq = np.sum(h**2)
        peak = peak_autoconv(h)
        if peak < 1e-12:
            return 0.0
        return l2_sq / (l1 * peak)
    
    best_score = compute_score(best_h)
    
    # Try multiple initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            h_init = np.ones(N) / N
        elif restart == 1:
            # Concentrated in center
            h_init = np.zeros(N)
            h_init[N//2 - 2:N//2 + 3] = 0.2
        elif restart == 2:
            # Two peaks
            h_init = np.zeros(N)
            h_init[N//4] = 0.5
            h_init[3*N//4] = 0.5
        elif restart == 3:
            # Random
            np.random.seed(42 + restart)
            h_init = np.random.dirichlet(np.ones(N))
        else:
            # Sparse random
            np.random.seed(100 + restart)
            h_init = np.zeros(N)
            idx = np.random.choice(N, size=5, replace=False)
            h_init[idx] = 1.0 / 5
            h_init = h_init / np.sum(h_init)
        
        try:
            res = minimize(
                objective,
                h_init,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < np.inf:
                h_res = res.x
                # Ensure nonnegativity and normalization
                h_res = np.maximum(h_res, 0)
                if np.sum(h_res) > 1e-12:
                    h_res = h_res / np.sum(h_res)
                
                score = compute_score(h_res)
                if score > best_score:
                    best_score = score
                    best_h = h_res.copy()
                    
        except Exception as e:
            pass
    
    # Convert to list of floats
    result = [float(x) for x in best_h]
    return result
