"""Minimize normalized max autoconvolution of nonnegative step function on [-1/4, 1/4]."""
import numpy as np
import time

N = 600
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _autoconv_peak(heights):
    """Compute the maximum of the autoconvolution of the step function.
    heights: array of N nonnegative values on equal intervals of [-1/4, 1/4].
    Returns the peak value of the autoconvolution."""
    # The autoconvolution (f * f)(t) for a step function can be computed
    # via FFT for efficiency
    n_fft = 2 * N + 1
    f = np.zeros(n_fft)
    f[:N] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=n_fft)
    # The autoconvolution is symmetric; peak is at index 0 (lag 0)
    # But we should check the full range to be safe
    return np.max(conv)

def _score(heights):
    """Compute the normalized max autoconvolution (the score to minimize)."""
    integral = np.sum(heights)
    if integral <= 0:
        return np.inf
    # Normalize so that integral = 1 for fair comparison
    h = heights / integral
    peak = _autoconv_peak(h)
    return peak

def construct():
    start_time = time.time()
    
    # Initial guess: raised cosine profile
    x = np.linspace(0, 1, N)
    heights = 0.5 * (1 + np.cos(2 * np.pi * x))
    heights[0] = 0.0
    heights[-1] = 0.0
    # Ensure nonnegative
    heights = np.maximum(heights, 0.0)
    
    best_heights = heights.copy()
    best_score = _score(heights)
    
    # Use scipy's L-BFGS-B for optimization
    try:
        from scipy.optimize import minimize
        from scipy.optimize import LinearConstraint
        
        # We optimize in log-space to ensure nonnegativity
        # h_i = exp(theta_i)
        # Minimize peak autoconv of normalized h
        
        def objective(theta):
            if time.time() - start_time > TIME_LIMIT:
                return best_score
            h = np.exp(theta)
            # Clip to avoid overflow
            h = np.clip(h, 1e-10, 1e10)
            return _score(h)
        
        # Gradient via finite differences or analytic
        def gradient(theta):
            h = np.exp(theta)
            h = np.clip(h, 1e-10, 1e10)
            integral = np.sum(h)
            hn = h / integral
            
            # Compute autoconvolution gradient
            # d(peak)/d(h_i) where peak = max(conv)
            # For simplicity, use the value at lag 0 which is sum(hn^2)
            # d/dh_i [sum(hn^2)] = 2*hn_i*(1/integral - 1/integral^2 * sum(hn))
            # = 2*hn_i*(1/integral - 1/integral) = 0 ... that's not right
            # Let me just use numerical gradient
            
            eps = 1e-7
            grad = np.zeros_like(theta)
            for i in range(0, N, max(1, N // 100)):  # sparse gradient for speed
                theta_p = theta.copy()
                theta_p[i] += eps
                h_p = np.exp(theta_p)
                h_p = np.clip(h_p, 1e-10, 1e10)
                f_p = _score(h_p)
                
                theta_m = theta.copy()
                theta_m[i] -= eps
                h_m = np.exp(theta_m)
                h_m = np.clip(h_m, 1e-10, 1e10)
                f_m = _score(h_m)
                
                grad[i] = (f_p - f_m) / (2 * eps)
            
            # Interpolate gradient
            if N > 100:
                idx = np.arange(0, N, max(1, N // 100))
                grad_full = np.interp(np.arange(N), idx, grad)
                return grad_full
            return grad
        
        # Initial theta
        h_init = np.maximum(heights, 1e-6)
        theta_init = np.log(h_init)
        
        # Bounds for theta
        theta_lower = np.full(N, -10.0)
        theta_upper = np.full(N, 10.0)
        
        result = minimize(
            objective,
            theta_init,
            method='L-BFGS-B',
            jac=gradient,
            bounds=list(zip(theta_lower, theta_upper)),
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() - start_time < TIME_LIMIT:
            h_opt = np.exp(result.x)
            h_opt = np.clip(h_opt, 1e-10, 1e10)
            s_opt = _score(h_opt)
            if s_opt < best_score:
                best_score = s_opt
                best_heights = h_opt.copy()
        
    except Exception:
        pass
    
    # Final polish: local search on the best heights found
    if time.time() - start_time < TIME_LIMIT - 10:
        h = best_heights.copy()
        step_size = 0.1
        for iteration in range(200):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            improved = False
            for i in range(N):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                h_trial = h.copy()
                h_trial[i] = max(0, h[i] + step_size)
                s_trial = _score(h_trial)
                if s_trial < best_score - 1e-12:
                    best_score = s_trial
                    h = h_trial
                    improved = True
                    break
                
                h_trial = h.copy()
                h_trial[i] = max(0, h[i] - step_size)
                s_trial = _score(h_trial)
                if s_trial < best_score - 1e-12:
                    best_score = s_trial
                    h = h_trial
                    improved = True
                    break
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        if best_score < _score(best_heights):
            best_heights = h.copy()
    
    # Ensure nonnegative and return as list
    best_heights = np.maximum(best_heights, 0.0)
    
    return list(best_heights)
