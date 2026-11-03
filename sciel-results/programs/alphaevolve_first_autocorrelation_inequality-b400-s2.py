"""Combined program: parametric Gaussian superposition + fine-grid FFT ratio + direct polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0

# Fine grid for accurate autoconvolution evaluation
FINE_N = 6000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Normalization: max((f*f)(t)) / (integral f)^2
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    # Map step heights to fine grid
    f_fine = h[CELL]
    
    # FFT-based autoconvolution
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX  # scale by dx for continuous integral
    
    max_conv = np.max(conv)
    
    # Normalization: (integral f)^2 = (sum(h) * W)^2
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')
    
    return max_conv / integral_sq

def params_to_h(params):
    """
    Convert Gaussian superposition parameters to step function heights.
    params: [a1, x1, s1, a2, x2, s2, ..., aK, xK, sK]
    """
    K = len(params) // 3
    h_fine = np.zeros(FINE_N)
    
    for i in range(K):
        a_raw = params[3 * i]
        x_c = params[3 * i + 1]
        s_raw = params[3 * i + 2]
        
        # Softplus for positive amplitude
        a = np.log1p(np.exp(np.clip(a_raw, -50, 50)))
        # Exp for positive width
        s = np.exp(np.clip(s_raw, -10, 10))
        s = max(s, 1e-4)
        # Clip center to domain
        x_c = np.clip(x_c, -0.25, 0.25)
        
        h_fine += a * np.exp(-0.5 * ((X_FINE - x_c) / s) ** 2)
    
    # Bin the fine values to N intervals (take mean per bin)
    h_bins = np.zeros(N)
    sums = np.zeros(N)
    counts = np.zeros(N)
    np.add.at(sums, CELL, h_fine)
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
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)
    
    # Number of Gaussian basis functions
    K = 5
    
    np.random.seed(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform-ish spread
    p0 = np.zeros(K * 3)
    for i in range(K):
        p0[3 * i] = np.log(1.0 / K)
        p0[3 * i + 1] = np.random.uniform(-0.2, 0.2)
        p0[3 * i + 2] = np.log(0.1 + np.random.rand() * 0.2)
    inits.append(p0)
    
    # 2. Single dominant bump at various positions
    for shift in [-0.15, 0.0, 0.15]:
        p0 = np.zeros(K * 3)
        p0[0] = np.log(1.0)
        p0[1] = shift
        p0[2] = np.log(0.15)
        for i in range(1, K):
            p0[3 * i] = np.log(0.01)
            p0[3 * i + 1] = np.random.uniform(-0.2, 0.2)
            p0[3 * i + 2] = np.log(0.05)
        inits.append(p0)
    
    # 3. Random smooth shapes
    for _ in range(10):
        p0 = np.zeros(K * 3)
        for i in range(K):
            p0[3 * i] = np.random.uniform(-2, 1)
            p0[3 * i + 1] = np.random.uniform(-0.25, 0.25)
            p0[3 * i + 2] = np.random.uniform(np.log(0.05), np.log(0.3))
        inits.append(p0)
    
    # 4. Asymmetric pairs
    for _ in range(5):
        p0 = np.zeros(K * 3)
        x1 = np.random.uniform(-0.2, -0.05)
        x2 = np.random.uniform(0.05, 0.2)
        p0[0] = np.random.uniform(0, 1)
        p0[1] = x1
        p0[2] = np.random.uniform(np.log(0.1), np.log(0.2))
        p0[3] = np.random.uniform(0, 1)
        p0[4] = x2
        p0[5] = np.random.uniform(np.log(0.1), np.log(0.2))
        for i in range(2, K):
            p0[3 * i] = np.random.uniform(-3, 0)
            p0[3 * i + 1] = np.random.uniform(-0.25, 0.25)
            p0[3 * i + 2] = np.random.uniform(np.log(0.05), np.log(0.2))
        inits.append(p0)
    
    # Optimize from each initialization
    for p0 in inits:
        if time.time() > deadline - 15.0:
            break
            
        try:
            res = minimize(
                objective,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < best_val:
                h_candidate = params_to_h(res.x)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Direct coordinate descent polish on the 600 bin heights
    if time.time() < deadline - 5.0:
        # Gradient-free local search: random perturbations
        for it in range(2000):
            if time.time() > deadline - 2.0:
                break
            
            # Perturb a random subset of bins
            n_perturb = np.random.randint(5, 20)
            idx = np.random.choice(N, n_perturb, replace=False)
            h_try = best_h.copy()
            h_try[idx] += np.random.randn(n_perturb) * 0.02
            h_try = np.clip(h_try, 0, None)
            
            val = ratio_from_h(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()
    
    return list(best_h)
