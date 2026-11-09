import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _ratio(h):
    """Calculate the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Calculate autoconvolution
    # Use FFT for efficiency, though N=50 is small
    # Linear convolution length is 2N-1
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    C = np.fft.ifft(H * H).real
    peak = np.max(C)
    if peak <= 0:
        return 0.0
    return l2_sq / (l1 * peak)

def _objective_from_h(h):
    """Objective function for minimization. h is the vector."""
    l1 = np.sum(h)
    if l1 == 0:
        return 1e10
    l2_sq = np.sum(h**2)
    if l2_sq == 0:
        return 1e10
    
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    C = np.fft.ifft(H * H).real
    peak = np.max(C)
    
    if peak <= 0:
        return 1e10
        
    return (l1 * peak) / l2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # 1. Parametric scans
    # Geometric decay
    for r in np.linspace(0.1, 0.95, 50):
        h = r ** np.arange(N)
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
            
    # Gaussian
    for sigma in np.linspace(1, 10, 50):
        x = np.arange(N)
        h = np.exp(-(x - N/2)**2 / (2*sigma**2))
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()

    # 2. Local optimization from best point
    # We optimize h directly with bounds [0, inf) but SLSQP might struggle with bounds.
    # Let's try Nelder-Mead or Powell on h directly?
    # Or use the y=sqrt(h) trick but with better initializations.
    
    # The directive suggests "continuous spectral optimization... search over Fourier coefficients".
    # However, the constraint h >= 0 makes Fourier coefficient search difficult (inverse FFT must be non-negative).
    # A more robust structural change is to use a better global optimizer like Differential Evolution or CMA-ES if available, 
    # or a more aggressive multi-start local search.
    # Let's try a different representation: optimize log(h) or just use a better local search on h with clipping.
    
    # Let's try to use the insight from "inspiration 1": Nelder-Mead.
    # And "cell findings": SLSQP + Nelder-Mead.
    
    # We will run a loop of random restarts + local optimization.
    # We will use Nelder-Mead on the vector h, but we must ensure h >= 0.
    # Nelder-Mead doesn't handle bounds.
    # So we stick to y = sqrt(h) or use a penalty.
    # Let's use y = sqrt(h) and Nelder-Mead.
    
    np.random.seed(42)
    
    # Initializations
    y0 = np.sqrt(best_h)
    starts = [y0]
    
    # Random starts
    for _ in range(20):
        y_rand = np.random.rand(N) * 0.5 + 0.1
        starts.append(y_rand)
        
    # Sparse random
    for _ in range(10):
        y_struct = np.zeros(N)
        indices = np.random.choice(N, size=np.random.randint(5, 15), replace=False)
        y_struct[indices] = np.random.rand(len(indices))
        starts.append(y_struct)
        
    # Cosine windows
    for alpha in np.linspace(0.1, 2.0, 10):
        x = np.arange(N)
        h_cos = np.cos(np.pi * (x - (N-1)/2) / (N - 1) * alpha) ** 2
        h_cos = np.maximum(h_cos, 0)
        starts.append(np.sqrt(h_cos))
        
    # Run optimization
    for y_init in starts:
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead for robustness against local minima, then polish with Powell?
            # Or just Powell with better options.
            res = minimize(_objective_from_h, y_init, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < 1e10:
                y_opt = res.x
                h_opt = y_opt**2
                s = _ratio(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
                    
                    # Polish with Powell
                    res2 = minimize(_objective_from_h, y_opt, method='Powell',
                                    options={'maxiter': 100, 'ftol': 1e-10, 'xtol': 1e-10})
                    if res2.fun < 1e10:
                        y_opt2 = res2.x
                        h_opt2 = y_opt2**2
                        s2 = _ratio(h_opt2)
                        if s2 > best_score:
                            best_score = s2
                            best_h = h_opt2.copy()
                            
        except Exception:
            pass

    return list(best_h)
