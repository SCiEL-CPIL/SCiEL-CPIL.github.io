import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, ifft
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess: uniform
    f = np.ones(N) / N
    
    # We want to minimize max(f*f).
    # Surrogate: log-sum-exp(beta * (f*f))
    
    # Helper to compute autocorrelation
    def get_autocorr(x):
        # x is length N
        # We need linear convolution, not circular.
        # Pad to 2N to avoid wrap-around
        x_pad = np.zeros(2*N)
        x_pad[:N] = x
        F = fft(x_pad)
        power = F * np.conj(F)
        c = ifft(power).real
        # The linear autocorrelation is in c[0:N] (lags 0 to N-1)
        # Note: scipy fft returns unnormalized forward, so inverse is 1/n.
        # The values are correct for linear convolution.
        return c[:N]

    def objective(x, beta):
        c = get_autocorr(x)
        # Soft max
        # To avoid overflow, subtract max
        max_c = np.max(c)
        if max_c == 0:
            return 0.0
        # log(sum(exp(beta * (c - max_c)))) + beta * max_c
        # Actually, we just need to minimize the surrogate.
        # The value of the surrogate is approx max_c.
        # We can just return the log-sum-exp value.
        # But we need to be careful with the scaling.
        
        # If beta is small, it's an average. If large, it's a max.
        # Let's return the soft-max value.
        
        # Optimization: if beta is very high, we can just use max?
        # No, gradient of max is not well-defined everywhere.
        
        scaled = beta * (c - max_c)
        # Clip to prevent underflow/overflow issues in exp
        scaled = np.clip(scaled, -500, 500)
        s = np.sum(np.exp(scaled))
        if s == 0:
            return 0.0
        return (1.0 / beta) * np.log(s) + max_c

    def gradient(x, beta):
        c = get_autocorr(x)
        max_c = np.max(c)
        scaled = beta * (c - max_c)
        scaled = np.clip(scaled, -500, 500)
        exp_scaled = np.exp(scaled)
        s = np.sum(exp_scaled)
        if s == 0:
            return np.zeros_like(x)
        
        # Probability weights for the soft max
        p = exp_scaled / s
        
        # Gradient of log-sum-exp(beta * c) w.r.t c is beta * p
        # Gradient of c w.r.t x
        # c[k] = sum_i x[i] x[i+k]
        # dc/dx[j] = sum_k p[k] * d c[k] / d x[j]
        # d c[k] / d x[j] = x[j+k] + x[j-k] (if indices valid)
        # Actually, let's use the FFT gradient or just compute the Jacobian-vector product?
        # For N=600, a dense Jacobian is 600x600, which is fine.
        # But we can compute the gradient of the soft-max directly.
        
        # Let's compute the gradient of the objective function J(x) = LSE(beta * (x*x))
        # dJ/dx_j = sum_k (dLSE/dc_k) * (dc_k/dx_j)
        # dLSE/dc_k = beta * p_k
        
        # dc_k/dx_j:
        # c_k = sum_i x_i x_{i+k}
        # Term where i=j: x_{j+k}
        # Term where i+k=j => i=j-k: x_{j-k}
        # So dc_k/dx_j = x_{j+k} + x_{j-k} (with zero padding outside [0, N-1])
        
        # This requires O(N^2) per gradient step if done naively.
        # N=600, N^2=360,000. This is very fast.
        
        grad = np.zeros(N)
        # We need to sum over k
        # grad[j] = sum_k (beta * p[k]) * (x[j+k] + x[j-k])
        
        # Let's construct the matrix M where M[j,k] = x[j+k] + x[j-k]
        # This is a Toeplitz-like structure.
        # We can compute this efficiently.
        
        # x_ext: padded x for indexing
        # x[j+k] is valid for 0 <= j+k < N
        # x[j-k] is valid for 0 <= j-k < N
        
        # Let's just loop over k for simplicity, N=600 is small.
        for k in range(N):
            if p[k] < 1e-12:
                continue
            weight = beta * p[k]
            # Contribution to grad[j] from lag k
            # x[j+k]
            # j+k < N => j < N-k
            # x[j-k]
            # j-k >= 0 => j >= k
            
            # Vectorized update:
            # grad[0:N-k] += weight * x[k:N]
            # grad[k:N] += weight * x[0:N-k]
            
            if N > k:
                grad[:N-k] += weight * x[k:]
                grad[k:] += weight * x[:N-k]
                
        return grad

    # Annealing schedule
    # Start with low beta, increase over time.
    # Total time ~ 140s.
    # Let's do a few restarts or a single long run with annealing.
    
    best_f = f.copy()
    best_score = objective(f, 100.0) # Initial rough score
    
    # We will run minimize for a fixed number of iterations or time.
    # Since we have a deadline, we check time in a loop if we do multiple phases.
    
    # Phase 1: Low beta (smooth)
    # Phase 2: Medium beta
    # Phase 3: High beta (sharp)
    
    betas = [10.0, 50.0, 100.0, 500.0, 1000.0]
    
    for b in betas:
        if time.time() > deadline - 10.0:
            break
            
        # Objective and gradient closures for current beta
        obj_func = lambda x: objective(x, b)
        grad_func = lambda x: gradient(x, b)
        
        # Constraints: sum(x) = 1
        # We handle sum(x)=1 by parameterization? 
        # L-BFGS-B doesn't handle equality constraints well.
        # We can use SLSQP which does.
        # Or we can normalize x inside the objective?
        # If we normalize, the gradient changes.
        # Better to use SLSQP with constraints.
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(
                obj_func, 
                f, 
                method='SLSQP', 
                jac=grad_func, 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 50, 'ftol': 1e-8, 'disp': False}
            )
            if res.success:
                f = res.x
                # Normalize just in case
                f = f / np.sum(f)
                # Evaluate true score (max autocorr)
                c = get_autocorr(f)
                true_score = np.max(c)
                if true_score < best_score:
                    best_score = true_score
                    best_f = f.copy()
        except Exception:
            pass

    # Final polish with a very high beta if time permits
    if time.time() < deadline - 5.0:
        b = 5000.0
        obj_func = lambda x: objective(x, b)
        grad_func = lambda x: gradient(x, b)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.0, 1.0)] * N
        try:
            res = minimize(
                obj_func, 
                best_f, 
                method='SLSQP', 
                jac=grad_func, 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
            )
            if res.success:
                f = res.x
                f = f / np.sum(f)
                c = get_autocorr(f)
                true_score = np.max(c)
                if true_score < best_score:
                    best_score = true_score
                    best_f = f.copy()
        except Exception:
            pass

    # Ensure non-negativity and normalization
    best_f = np.clip(best_f, 0.0, None)
    s = np.sum(best_f)
    if s > 0:
        best_f = best_f / s
    else:
        best_f = np.ones(N) / N

    return list(best_f)
