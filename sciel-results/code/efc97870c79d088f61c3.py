import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0

# Precompute fine grid for fast evaluation
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute the normalized maximum of the autoconvolution."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    f_fine = h[CELL]
    
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    max_conv = np.max(conv)
    l2_sq = np.sum(f_fine ** 2) * DX
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq

def params_to_h(params):
    """
    Convert parameters to step function heights.
    params: [a1, x1, s1, a2, x2, s2, ..., aK, xK, sK]
    """
    K = len(params) // 3
    h = np.zeros(FINE_N)
    for i in range(K):
        a_raw = params[3*i]
        x_c = params[3*i + 1]
        s_raw = params[3*i + 2]
        
        # Ensure positive amplitude and width
        a = np.log1p(np.exp(a_raw))
        
        # Ensure positive width, avoid division by zero
        s = np.exp(s_raw) if s_raw < 20 else 1e10
        s = max(s, 1e-4)
        
        # Clip center to domain to avoid huge values, though exp handles it
        x_c = np.clip(x_c, -0.25, 0.25)
        
        h += a * np.exp(-0.5 * ((X_FINE - x_c) / s) ** 2)
    
    # Map to N bins
    sums = np.zeros(N)
    counts = np.zeros(N)
    np.add.at(sums, CELL, h)
    np.add.at(counts, CELL, 1)
    counts[counts == 0] = 1
    h_bins = sums / counts
    
    return h_bins

def objective(params):
    h = params_to_h(params)
    val = ratio_from_h(h)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    deadline = time.time() + 145.0 # Safety margin
    
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)
    
    # Define number of basis functions
    K = 5
    
    np.random.seed(42)
    
    # Initializations
    inits = []
    
    # 1. Uniform-ish (small random perturbations around center)
    p0 = np.zeros(K * 3)
    for i in range(K):
        p0[3*i] = np.log(1.0 / K) # log amplitude
        p0[3*i+1] = np.random.uniform(-0.2, 0.2)
        p0[3*i+2] = np.log(0.1 + np.random.rand()*0.2) # log width
    
    inits.append(p0)
    
    # 2. Single dominant bump
    for shift in [-0.15, 0, 0.15]:
        p0 = np.zeros(K * 3)
        # First bump strong
        p0[0] = np.log(1.0)
        p0[1] = shift
        p0[2] = np.log(0.15)
        # Other bumps weak
        for i in range(1, K):
            p0[3*i] = np.log(0.01)
            p0[3*i+1] = np.random.uniform(-0.2, 0.2)
            p0[3*i+2] = np.log(0.05)
        inits.append(p0)
        
    # 3. Random smooth shapes
    for _ in range(10):
        p0 = np.zeros(K * 3)
        for i in range(K):
            p0[3*i] = np.random.uniform(-2, 1) # log amp
            p0[3*i+1] = np.random.uniform(-0.25, 0.25)
            p0[3*i+2] = np.random.uniform(np.log(0.05), np.log(0.3)) # log width
        inits.append(p0)
        
    # 4. Asymmetric pairs
    for _ in range(5):
        p0 = np.zeros(K * 3)
        # Two main bumps
        x1 = np.random.uniform(-0.2, -0.05)
        x2 = np.random.uniform(0.05, 0.2)
        p0[0] = np.random.uniform(0, 1)
        p0[1] = x1
        p0[2] = np.random.uniform(np.log(0.1), np.log(0.2))
        
        p0[3] = np.random.uniform(0, 1)
        p0[4] = x2
        p0[5] = np.random.uniform(np.log(0.1), np.log(0.2))
        
        for i in range(2, K):
            p0[3*i] = np.random.uniform(-3, 0)
            p0[3*i+1] = np.random.uniform(-0.25, 0.25)
            p0[3*i+2] = np.random.uniform(np.log(0.05), np.log(0.2))
        inits.append(p0)
        
    # Optimization
    for p0 in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective, 
                p0, 
                method='L-BFGS-B', 
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            if res.success or res.fun < best_val:
                h_candidate = params_to_h(res.x)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
            
    # Local refinement on the best h found
    if time.time() < deadline - 5.0:
        # Use a few random perturbations on the bin heights to fine-tune
        # This is a fallback in case the parametric model is too restrictive
        for it in range(500):
            if time.time() > deadline - 2.0:
                break
            idx = np.random.choice(N, 10, replace=False)
            h_try = best_h.copy()
            h_try[idx] += np.random.randn(10) * 0.01
            h_try = np.clip(h_try, 0, None)
            val = ratio_from_h(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()
                
    return list(best_h)
