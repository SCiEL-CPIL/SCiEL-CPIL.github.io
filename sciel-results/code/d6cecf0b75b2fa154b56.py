import numpy as np
import time
from scipy.optimize import minimize

N = 600
dx = 1.0 / (2 * N)  # interval width: [-1/4, 1/4] has length 1/2, N intervals

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    n_fft = 2 * N

    def autoconv(h):
        """Compute autoconvolution h*h via FFT. Returns array of length 2N-1."""
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        H = np.fft.fft(h_padded)
        autoconv_circ = np.fft.ifft(H * H).real
        return autoconv_circ[:2*N-1] * dx**2

    def objective_value(h):
        """Just the objective value for evaluation."""
        h = np.maximum(h, 0)
        l2 = np.sum(h**2) * dx
        if l2 < 1e-15:
            return 1e10
        h_norm = h / np.sqrt(l2)
        g = autoconv(h_norm)
        return np.max(g)

    def smooth_objective_with_grad(h, tau=0.01):
        """Objective with gradient for scipy, using smooth max."""
        h = np.maximum(h, 0)
        l2 = np.sum(h**2) * dx
        if l2 < 1e-15:
            return 1e10, np.zeros(N)
        h_norm = h / np.sqrt(l2)
        g = autoconv(h_norm)
        
        max_val = np.max(g)
        if max_val <= 0:
            return 0.0, np.zeros(N)
        
        w = np.exp((g - max_val) / tau)
        W = np.sum(w)
        smooth_max = tau * (max_val + np.log(W))
        
        w_padded = np.zeros(n_fft)
        w_padded[:len(w)] = w
        Wp = np.fft.fft(w_padded)
        Hn = np.fft.fft(h_norm)
        grad_hnorm = np.fft.ifft(Wp * Hn).real[:N] / W
        
        sqrt_l2 = np.sqrt(l2)
        d_hnorm_d_h = (1 - h**2 / l2) / sqrt_l2
        grad = grad_hnorm * d_hnorm_d_h
        
        return smooth_max, grad

    # Generate diverse initial candidates
    centers = np.arange(N) - N // 2
    candidates = []
    
    # Gaussian peaks with various widths
    for sigma in [1, 2, 5, 10, 20, 50, 100, 200]:
        h = np.exp(-centers**2 / (2 * sigma**2))
        candidates.append(h)
    
    # Triangular peaks
    for width in [10, 50, 100, 200, 300]:
        h = np.maximum(0, 1 - np.abs(centers) / width)
        candidates.append(h)
    
    # Box functions
    for half_width in [10, 50, 100, 200, 300]:
        h = np.zeros(N)
        mask = np.abs(centers) <= half_width
        h[mask] = 1.0
        candidates.append(h)
    
    # Gaussian + base
    for sigma in [5, 20, 50]:
        for base in [0.1, 0.3, 0.5]:
            h = base + (1 - base) * np.exp(-centers**2 / (2 * sigma**2))
            candidates.append(h)
    
    # Inverted Gaussian
    for sigma in [50, 100, 200]:
        h = 1.0 - np.exp(-centers**2 / (2 * sigma**2))
        h = np.maximum(h, 0)
        candidates.append(h)
    
    # Random smooth functions
    np.random.seed(42)
    for trial in range(10):
        freqs = np.random.uniform(0.5, 5.0, size=5)
        phases = np.random.uniform(0, 2*np.pi, size=5)
        amps = np.random.uniform(0.5, 2.0, size=5)
        h = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            h += a * np.cos(f * centers / N * 2 * np.pi + p)
        h = np.maximum(h, 0)
        if np.sum(h) > 1e-10:
            candidates.append(h)

    # Evaluate all candidates
    best_score = float('inf')
    best_h = None

    for h in candidates:
        if time.time() > deadline:
            break
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < 0):
            h = np.maximum(h, 0)
        if np.sum(h) < 1e-10:
            continue
        score = objective_value(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Optimize from the best candidate using L-BFGS-B with smooth objective
    if best_h is not None and time.time() < deadline - 30:
        # Try multiple temperatures for smooth max
        for tau in [0.1, 0.05, 0.02, 0.01]:
            if time.time() > deadline - 20:
                break
            
            def obj_grad(h):
                val, grad = smooth_objective_with_grad(h, tau=tau)
                return val, grad
            
            # Start from best_h
            x0 = best_h.copy()
            
            try:
                res = minimize(obj_grad, x0, method='L-BFGS-B', 
                              jac=True,
                              bounds=[(0, None)] * N,
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
                
                h_opt = np.maximum(res.x, 0)
                score_opt = objective_value(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
            except Exception as e:
                pass

    # Final polish with exact max (tau -> 0)
    if best_h is not None and time.time() < deadline - 15:
        for tau in [0.005, 0.001]:
            if time.time() > deadline - 10:
                break
            
            def obj_grad(h):
                val, grad = smooth_objective_with_grad(h, tau=tau)
                return val, grad
            
            try:
                res = minimize(obj_grad, best_h.copy(), method='L-BFGS-B',
                              jac=True,
                              bounds=[(0, None)] * N,
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                
                h_opt = np.maximum(res.x, 0)
                score_opt = objective_value(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
            except Exception as e:
                pass

    if best_h is None:
        return [1.0] * N

    return list(best_h)

# EVOLVE-BLOCK-END
