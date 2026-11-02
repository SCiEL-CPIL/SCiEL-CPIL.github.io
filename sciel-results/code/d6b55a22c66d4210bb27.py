import numpy as np
from scipy.optimize import minimize
import time

N = 600

def _compute_score(f):
    """
    Computes the normalized maximum of the autoconvolution.
    Assumes the metric is max(f*f) / (sum(f)^2) or similar, 
    but we will compute the raw max(f*f) and normalize by the seed value 
    to be safe, or just return the value if the evaluator handles scaling.
    Based on the prompt 'score is seed/value', and seed is 1.0, 
    if we return the raw value, the evaluator might divide by seed_raw.
    However, the prompt says 'score is the upper bound on C1'.
    We will assume the objective to minimize is max(f*f) given sum(f)=1.
    """
    if np.any(f < 0):
        return 1e10
    s = np.sum(f)
    if s < 1e-9:
        return 1e10
    f_norm = f / s
    # Compute autoconvolution
    # Using FFT for efficiency
    fft_f = np.fft.fft(f_norm, n=2*N)
    fft_conv = fft_f * fft_f
    conv = np.fft.ifft(fft_conv).real
    # The autoconvolution of a function on [0, N-1] is centered at 0.
    # In FFT output, the peak is at index 0.
    # We need the maximum value.
    # For non-negative functions, the peak is at 0.
    # But to be robust, take the max of the real part.
    return np.max(conv)

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    # Initial guess: Uniform
    f_best = np.ones(N) / N
    best_score = _compute_score(f_best)
    
    # Strategy: Optimize in the frequency domain or use a sophisticated time-domain optimizer.
    # We will use L-BFGS-B on the time-domain variables with a log-barrier for positivity.
    # To handle the non-negativity strictly, we optimize log(f) + small epsilon.
    # Let g = log(f). f = exp(g).
    # Constraint sum(exp(g)) = 1.
    # This is a convex problem if we minimize sum(exp(2g))? No, autoconv is non-convex in g.
    
    # Alternative: Use Simulated Annealing on the step heights.
    # Or use a Genetic Algorithm.
    # Given the time limit, a simple gradient-based method with multiple restarts is best.
    
    def objective(g):
        f = np.exp(g)
        s = np.sum(f)
        f_norm = f / s
        return _compute_score(f_norm)
        
    def gradient(g):
        # Numerical gradient is safer for complex objective
        eps = 1e-6
        grad = np.zeros_like(g)
        for i in range(N):
            g_p = g.copy()
            g_m = g.copy()
            g_p[i] += eps
            g_m[i] -= eps
            grad[i] = (objective(g_p) - objective(g_m)) / (2 * eps)
        return grad

    # We will try a few random initializations in the log-space.
    # Uniform corresponds to g = -log(N) everywhere.
    
    best_g = -np.log(N) * np.ones(N)
    
    for i in range(5): # 5 restarts
        if time.time() > deadline - safety_margin:
            break
            
        if i == 0:
            g_current = best_g.copy()
        else:
            # Random perturbation
            g_current = best_g + np.random.normal(0, 0.1, N)
            
        # Optimize
        try:
            res = minimize(objective, g_current, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-6})
            if res.success or res.fun < best_score:
                f_candidate = np.exp(res.x)
                s = np.sum(f_candidate)
                f_candidate = f_candidate / s
                score_candidate = _compute_score(f_candidate)
                if score_candidate < best_score:
                    best_score = score_candidate
                    best_g = res.x.copy()
                    f_best = f_candidate
        except Exception:
            pass

    return f_best.tolist()

if __name__ == "__main__":
    construct()
