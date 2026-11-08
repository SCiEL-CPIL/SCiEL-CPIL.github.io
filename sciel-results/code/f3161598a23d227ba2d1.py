import numpy as np
from scipy.optimize import minimize
import time

N = 600
DELTA = (1/2) / N  # width of each interval = 1/1200

def compute_autoconvolution(h):
    """Compute the full autoconvolution (autocorrelation) of step function h using FFT."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    
    # Use FFT to compute autocorrelation efficiently
    # Pad to avoid circular convolution
    size = 2 * n
    f = np.zeros(size)
    f[:n] = h
    
    F = np.fft.fft(f)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    autocorr = autocorr[:n]  # take first n lags (non-negative)
    
    return autocorr * DELTA

def soft_max(arr, beta=50.0):
    """Compute a smooth approximation to max using log-sum-exp."""
    arr = np.asarray(arr, dtype=float)
    m = np.max(arr)
    if m < 1e-10:
        return m
    return m + (1.0/beta) * np.log(np.sum(np.exp(beta * (arr - m))))

def compute_autoconvolution_max_smooth(h, beta=50.0):
    """Compute a smooth approximation to the maximum of the autoconvolution."""
    autocorr = compute_autoconvolution(h)
    return soft_max(autocorr, beta)

def objective_smooth(h, beta=50.0):
    """Minimize smooth_max(autoconvolution) / (integral)^2"""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    integral = np.sum(h) * DELTA
    if integral < 1e-10:
        return 1e10
    max_ac = compute_autoconvolution_max_smooth(h, beta)
    return max_ac / (integral * integral)

def objective_exact(h):
    """Compute the exact (non-smooth) objective for final evaluation."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    integral = np.sum(h) * DELTA
    if integral < 1e-10:
        return 1e10
    autocorr = compute_autoconvolution(h)
    max_ac = np.max(autocorr)
    return max_ac / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_score = objective_exact(best_h)
    
    bounds = [(0, None)] * N
    np.random.seed(42)
    
    # Phase 1: Optimization with smooth surrogate
    # Use increasing beta values for annealing
    betas = [10.0, 50.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 20:  # Leave time for polish
            break
        
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            
            if restart == 0:
                x0 = np.ones(N)
            elif restart == 1:
                x0 = np.random.exponential(1.0, N)
            elif restart == 2:
                # Cosine-shaped
                x0 = np.cos(np.linspace(0, np.pi, N)) ** 2
                x0 = np.maximum(x0, 0)
            
            try:
                # Use smooth objective for optimization
                obj = lambda h: objective_smooth(h, beta)
                result = minimize(obj, x0, method='L-BFGS-B', 
                                 bounds=bounds, options={'maxiter': 50, 'maxfun': 150})
                
                if time.time() > deadline - 20:
                    break
                
                # Evaluate with exact objective
                exact_score = objective_exact(result.x)
                if exact_score < best_score:
                    best_score = exact_score
                    best_h = np.maximum(result.x, 0)
            except Exception:
                continue
    
    # Phase 2: Final polish with high precision on the best candidate
    if time.time() < deadline - 10:
        try:
            # Use the exact objective with tighter tolerances
            result = minimize(objective_exact, best_h, method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 200, 'maxfun': 500, 
                                                    'ftol': 1e-12, 'gtol': 1e-12})
            exact_score = objective_exact(result.x)
            if exact_score < best_score:
                best_score = exact_score
                best_h = np.maximum(result.x, 0)
        except Exception:
            pass
    
    return [float(x) for x in best_h]
