import numpy as np
import time
from scipy.optimize import minimize, SLSQP

N = 100

def _score(h):
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute grid
    x = np.arange(N)
    
    h_best = np.full(N, 0.5)
    score_best = _score(h_best)

    # Try different numbers of Fourier modes
    for K in [5, 8, 12, 16, 20]:
        if time.time() > deadline - 30:
            break
            
        # Basis functions
        # h = 0.5 + sum(c_k cos + s_k sin)
        # We use L-BFGS-B on the coefficients
        n_params = 2 * K
        
        def get_h(params):
            h = np.full(N, 0.5)
            for k in range(1, K + 1):
                c_k = params[2*(k-1)]
                s_k = params[2*(k-1)+1]
                h += c_k * np.cos(2 * np.pi * k * x / N)
                h += s_k * np.sin(2 * np.pi * k * x / N)
            return h

        # Adaptive penalty weight for sum constraint
        penalty_weight = 10.0
        
        def objective(params):
            h = get_h(params)
            h_clipped = np.clip(h, 0.0, 1.0)
            
            shifted = np.zeros(N)
            shifted[:-1] = h_clipped[1:]
            overlap = h_clipped * (1.0 - shifted)
            
            # Smooth max of overlap
            sm = _smooth_max(overlap, alpha=200.0)
            
            # Penalty for out-of-bounds
            penalty_bounds = 50.0 * (np.sum(np.maximum(0, h - 1.0)**2) + np.sum(np.maximum(0, -h)**2))
            
            # Penalty for sum deviation (soft)
            sum_dev = np.sum(h) - N/2.0
            penalty_sum = penalty_weight * sum_dev**2
            
            return sm + penalty_bounds + penalty_sum

        def objective_grad(params):
            h = get_h(params)
            h_clipped = np.clip(h, 0.0, 1.0)
            
            shifted = np.zeros(N)
            shifted[:-1] = h_clipped[1:]
            overlap = h_clipped * (1.0 - shifted)
            
            # Gradient of smooth max
            x_max = np.max(overlap)
            e = np.exp(200.0 * (overlap - x_max))
            p = e / np.sum(e)
            
            mask = (h >= 0.0) & (h <= 1.0)
            
            h_next = np.roll(h_clipped, -1)
            h_next[-1] = 0.0
            h_prev = np.roll(h_clipped, 1)
            h_prev[0] = 0.0
            
            grad_h = p * (1.0 - h_next) - np.roll(p, 1) * h_prev
            grad_h *= mask
            
            # Gradient of bounds penalty
            grad_penalty = np.zeros(N)
            grad_penalty[h > 1.0] = 100.0 * (h[h > 1.0] - 1.0)
            grad_penalty[h < 0.0] = -100.0 * h[h < 0.0]
            
            # Gradient of sum penalty
            sum_dev = np.sum(h) - N/2.0
            grad_sum = 2.0 * penalty_weight * sum_dev * np.ones(N)
            
            total_grad_h = grad_h + grad_penalty + grad_sum
            
            grad_params = np.zeros(n_params)
            for k in range(1, K + 1):
                cos_k = np.cos(2 * np.pi * k * x / N)
                sin_k = np.sin(2 * np.pi * k * x / N)
                grad_params[2*(k-1)] = np.dot(total_grad_h, cos_k)
                grad_params[2*(k-1)+1] = np.dot(total_grad_h, sin_k)
                
            return grad_params

        # Initial guess: small random coefficients
        np.random.seed(42)
        h0 = np.random.randn(n_params) * 0.1
        
        try:
            res = minimize(
                objective,
                h0,
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            
            h_candidate = get_h(res.x)
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            # Re-project to sum N/2 with iterative bisection on offset
            # Try to find an offset that makes sum = N/2 while keeping in [0,1]
            for _ in range(50):
                diff = N/2.0 - np.sum(h_candidate)
                h_candidate += diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                if abs(N/2.0 - np.sum(h_candidate)) < 1e-10:
                    break
                
            sc = _score(h_candidate)
            if sc < score_best:
                score_best = sc
                h_best = h_candidate.copy()
                
        except Exception as e:
            pass

    # Final polish: use SLSQP with exact constraints
    if time.time() < deadline - 10:
        try:
            def polish_objective(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return _smooth_max(overlap, alpha=500.0)
            
            def polish_grad(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                
                x_max = np.max(overlap)
                e = np.exp(500.0 * (overlap - x_max))
                p = e / np.sum(e)
                
                h_next = np.roll(h, -1)
                h_next[-1] = 0.0
                h_prev = np.roll(h, 1)
                h_prev[0] = 0.0
                
                grad = p * (1.0 - h_next) - np.roll(p, 1) * h_prev
                return grad
            
            constraints = [
                {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0, 'jac': lambda h: np.ones(N)}
            ]
            bounds = [(0.0, 1.0)] * N
            
            res_polish = minimize(
                polish_objective,
                h_best,
                jac=polish_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            
            if res_polish.success:
                h_polished = res_polish.x
                h_polished = np.clip(h_polished, 0.0, 1.0)
                # Ensure sum is exactly N/2
                diff = N/2.0 - np.sum(h_polished)
                h_polished += diff / N
                h_polished = np.clip(h_polished, 0.0, 1.0)
                
                sc = _score(h_polished)
                if sc < score_best:
                    score_best = sc
                    h_best = h_polished.copy()
                    
        except Exception as e:
            pass

    return h_best
