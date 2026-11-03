import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute for fast convolution ratio
# Use a moderate FINE_N to balance accuracy and speed
FINE_N = 10000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute normalized max autoconvolution ratio using FFT."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

# Precompute Fourier basis for speed
# We use a small set of low-frequency harmonics to enforce smoothness
# Domain is [-0.25, 0.25], period T = 0.5
# Basis: 1, cos(4*pi*k*x), sin(4*pi*k*x)
K_MAX = 15
COS_BASIS = np.zeros((N, K_MAX))
SIN_BASIS = np.zeros((N, K_MAX))
for k in range(1, K_MAX + 1):
    COS_BASIS[:, k-1] = np.cos(4 * np.pi * k * X)
    SIN_BASIS[:, k-1] = np.sin(4 * np.pi * k * X)

def fourier_to_h_fast(c0, a, b):
    h = c0 + a @ COS_BASIS.T + b @ SIN_BASIS.T
    return np.clip(h, 0.0, None)

def objective(params):
    c0 = params[0]
    a = params[1:1+K_MAX]
    b = params[1+K_MAX:1+2*K_MAX]
    h = fourier_to_h_fast(c0, a, b)
    if h.sum() < 1e-12:
        return 1e6
    return ratio_from_h(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initial best: uniform
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    # Phase 1: Fourier Domain Optimization with Gradient-Based Methods
    # We use L-BFGS-B which is good for box-constrained problems and uses gradients.
    # Since we can't easily compute the gradient of ratio_from_h (which involves a max and FFT),
    # we will use L-BFGS-B with numerical gradients or just use Nelder-Mead with more iterations
    # but the directive asks for a structural change to gradient-based.
    # Let's try to compute the gradient numerically or use a method that doesn't require explicit gradients
    # but is more global than Nelder-Mead? Or just use L-BFGS-B with finite differences.
    # Actually, L-BFGS-B with finite differences can be slow.
    # Let's try a different approach: use the fact that the objective is differentiable almost everywhere.
    # The max in ratio_from_h is the issue. We can smooth it or just use Nelder-Mead but with a better initialization.
    
    # The directive says: "optimize these coefficients using gradient-based methods"
    # Let's use scipy.optimize.minimize with method='L-BFGS-B' and let it use finite differences.
    # To make it faster, we can reduce the number of parameters or use a coarser grid for the ratio computation during optimization.
    
    # Let's create a faster ratio function for optimization
    FINE_N_OPT = 2000
    X_FINE_OPT = np.linspace(-0.25, 0.25, FINE_N_OPT, endpoint=False)
    DX_OPT = 0.5 / FINE_N_OPT
    CELL_OPT = np.clip(((X_FINE_OPT + 0.25) / W).astype(np.int64), 0, N - 1)
    N_FFT_OPT = 1 << (2 * FINE_N_OPT - 1).bit_length()
    
    def ratio_from_h_fast(h):
        h = np.asarray(h, dtype=np.float64)
        if h.size != N:
            h = np.resize(h, N)
        h = np.clip(h, 0.0, None)
        s = h.sum()
        if s < 1e-15:
            return 1e6
        f = h[CELL_OPT]
        F = np.fft.rfft(f, N_FFT_OPT)
        conv = np.fft.irfft(F * F, N_FFT_OPT)[:2 * FINE_N_OPT - 1] * DX_OPT
        l2 = np.sum(f * f) * DX_OPT
        if l2 < 1e-15:
            return 1e6
        return float(np.max(conv) / l2)
    
    def objective_fast(params):
        c0 = params[0]
        a = params[1:1+K_MAX]
        b = params[1+K_MAX:1+2*K_MAX]
        h = fourier_to_h_fast(c0, a, b)
        if h.sum() < 1e-12:
            return 1e6
        return ratio_from_h_fast(h)
    
    # Try a few random initializations in Fourier space
    rng = np.random.default_rng(42)
    
    best_fourier_h = best_h.copy()
    best_fourier_val = best_val
    
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
            
        # Random initialization
        # c0 around 0.5-1.0
        c0_init = rng.uniform(0.2, 1.0)
        # a, b small random values, decaying with k
        a_init = np.zeros(K_MAX)
        b_init = np.zeros(K_MAX)
        for k in range(K_MAX):
            scale = 1.0 / (k + 1)
            a_init[k] = rng.normal(0, 0.1 * scale)
            b_init[k] = rng.normal(0, 0.1 * scale)
            
        x0 = np.concatenate([[c0_init], a_init, b_init])
        
        # L-BFGS-B optimization with finite differences
        try:
            res = opt.minimize(objective_fast, x0, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-5, 'maxfun': 1000})
            if res.fun < best_fourier_val:
                h_opt = fourier_to_h_fast(res.x[0], res.x[1:1+K_MAX], res.x[1+K_MAX:1+2*K_MAX])
                v_opt = ratio_from_h(h_opt)
                if v_opt < best_fourier_val:
                    best_fourier_val = v_opt
                    best_fourier_h = h_opt
        except Exception:
            pass

    if best_fourier_val < best_val:
        best_val = best_fourier_val
        best_h = best_fourier_h

    # Phase 2: Local Refinement in Spatial Domain
    if time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(12345)
        current_h = best_h.copy()
        current_val = best_val
        
        for it in range(2000):
            if time.time() > deadline - 5.0:
                break
            
            # Random block perturbation
            mode = int(rng2.integers(0, 3))
            if mode == 0:
                # Additive bump
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(5, 30))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                amp = float(rng2.normal(0.0, 0.1))
                h_try = current_h.copy()
                xs = np.linspace(-1, 1, hi - lo)
                bump = np.exp(-xs**2)
                h_try[lo:hi] += amp * bump
                h_try = np.clip(h_try, 0.0, None)
            elif mode == 1:
                # Multiplicative scale
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(5, 30))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                f = float(rng2.uniform(0.8, 1.2))
                h_try = current_h.copy()
                h_try[lo:hi] *= f
                h_try = np.clip(h_try, 0.0, None)
            else:
                # Smoothing / Blurring
                h_try = current_h.copy()
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(3, 15))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                if hi - lo > 2:
                    h_try[lo:hi] = np.convolve(h_try[lo:hi], np.ones(3)/3, mode='same')
            
            v = ratio_from_h(h_try)
            if v < current_val:
                current_val = v
                current_h = h_try
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()

    return list(best_h)
