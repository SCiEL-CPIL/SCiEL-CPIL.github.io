"""Combined program: analytic signal representation + fine-grid FFT ratio + multi-start optimization."""
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

# Precompute bin centers for parametric model evaluation
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * W

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

def params_to_h_analytic(params):
    """
    Convert complex exponential coefficients to step function heights.
    params: [c0_re, c0_im, c1_re, c1_im, ..., cK_re, cK_im]
    f(x) = |sum_k c_k * exp(i * k * 2*pi * x)|^2
    This ensures non-negativity by construction.
    """
    K = len(params) // 2
    h = np.zeros(N)
    
    # Evaluate the analytic signal at bin centers
    signal = np.zeros(N, dtype=np.complex128)
    for k in range(K):
        c_re = params[2*k]
        c_im = params[2*k+1]
        c = c_re + 1j * c_im
        # Frequency k corresponds to k cycles over the domain length 0.5
        # So exp(i * k * 2*pi * x / 0.5) = exp(i * k * 4*pi * x)
        # But we want frequencies that are integer multiples of the fundamental
        # The fundamental frequency for domain [-0.25, 0.25] is 1/0.5 = 2 cycles per unit
        # So frequency k is 2k cycles per unit, i.e., exp(i * 2*pi * 2k * x) = exp(i * 4*pi*k*x)
        signal += c * np.exp(1j * 4 * np.pi * k * BIN_CENTERS)
    
    # Squared magnitude
    h = np.abs(signal) ** 2
    
    return h

def objective_analytic(params):
    h = params_to_h_analytic(params)
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
    
    np.random.seed(42)
    
    # Number of complex exponentials to use
    K = 6
    
    # Generate diverse initializations for the complex coefficients
    inits = []
    
    # 1. Uniform-ish: only DC term
    p0 = np.zeros(2 * K)
    p0[0] = 1.0  # c0_re = 1, c0_im = 0
    inits.append(p0)
    
    # 2. Add first harmonic
    p0 = np.zeros(2 * K)
    p0[0] = 1.0
    p0[2] = 0.5  # c1_re
    p0[3] = 0.0  # c1_im
    inits.append(p0)
    
    # 3. Random smooth shapes with few harmonics
    for _ in range(15):
        p0 = np.zeros(2 * K)
        # DC term
        p0[0] = np.random.uniform(0.5, 1.5)
        p0[1] = np.random.uniform(-0.5, 0.5)
        # Other harmonics
        for k in range(1, K):
            amp = np.random.uniform(0.0, 0.5) * (1.0 / (k + 1))
            phase = np.random.uniform(0, 2 * np.pi)
            p0[2*k] = amp * np.cos(phase)
            p0[2*k+1] = amp * np.sin(phase)
        inits.append(p0)
    
    # 4. Asymmetric patterns
    for _ in range(10):
        p0 = np.zeros(2 * K)
        p0[0] = np.random.uniform(0.5, 1.5)
        p0[1] = 0.0
        # Alternating signs for some harmonics
        for k in range(1, K):
            amp = np.random.uniform(0.1, 0.4) * (1.0 / (k + 1))
            if k % 2 == 0:
                p0[2*k] = amp
            else:
                p0[2*k] = -amp
            p0[2*k+1] = 0.0
        inits.append(p0)
    
    # 5. Sparse random coefficients
    for _ in range(10):
        p0 = np.zeros(2 * K)
        p0[0] = 1.0
        # Pick 2-3 random harmonics
        harmonics = np.random.choice(range(1, K), size=np.random.randint(2, 4), replace=False)
        for k in harmonics:
            amp = np.random.uniform(0.2, 0.6)
            phase = np.random.uniform(0, 2 * np.pi)
            p0[2*k] = amp * np.cos(phase)
            p0[2*k+1] = amp * np.sin(phase)
        inits.append(p0)
    
    # Optimize from each initialization
    for p0 in inits:
        if time.time() > deadline - 15.0:
            break
            
        try:
            res = minimize(
                objective_analytic,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < best_val:
                h_candidate = params_to_h_analytic(res.x)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Direct coordinate descent polish on the 600 bin heights
    if time.time() < deadline - 5.0:
        # Gradient-free local search: random perturbations
        for it in range(3000):
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
