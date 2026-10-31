import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s
    
    def score(h):
        h = np.array(h, dtype=float)
        l2_sq = np.sum(h ** 2)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return -1.0
        # Autoconvolution: (h*h)[k] = sum_i h[i]*h[i+k] for k in -(N-1)..(N-1)
        # Using FFT for efficiency
        n_fft = 1 << (2 * N - 1).bit_length()
        H = np.fft.fft(h, n=n_fft)
        conv = np.fft.ifft(H * H).real
        # The autoconvolution at lag 0 is conv[N-1] if centered, but let's just take max
        sup_conv = np.max(conv)
        if sup_conv < 1e-15:
            return -1.0
        return l2_sq / (l1 * sup_conv)
    
    # Start with seed
    best_h = np.ones(N)
    best_score = score(best_h)
    
    # Try sparse patterns
    for K in range(1, 15):
        if time.time() > deadline:
            break
        # Try random sparse placements
        for trial in range(200):
            if time.time() > deadline:
                break
            idx = np.random.choice(N, size=K, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0
            s = score(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
    
    # Local optimization on best found
    if time.time() < deadline - 5:
        # Use continuous optimization
        def neg_score(x):
            h = np.abs(x)  # enforce nonnegativity
            return -score(h)
        
        x0 = best_h.copy()
        result = minimize(neg_score, x0, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < -best_score:
            h_opt = np.abs(result.x)
            s_opt = score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 2:
        def neg_score_tight(x):
            h = np.abs(x)
            return -score(h)
        
        x0 = best_h.copy()
        result = minimize(neg_score_tight, x0, method='L-BFGS-B',
                         options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.fun < -best_score:
            h_opt = np.abs(result.x)
            s_opt = score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
    
    return [float(x) for x in best_h]
