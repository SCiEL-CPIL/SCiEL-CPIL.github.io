"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

K = 24  # Number of Fourier modes
DIM = 2 * K  # Parameter dimension = 48
BETA = 120.0  # Sharpness of soft-max (higher = closer to true max)

# Precompute Fourier basis matrices
_n = np.arange(N)
_cos_basis = np.array([np.cos(2 * np.pi * k * _n / N) for k in range(1, K)])  # (K-1, N)
_sin_basis = np.array([np.sin(2 * np.pi * k * _n / N) for k in range(1, K)])  # (K-1, N)

def _value(h):
    """Compute the evaluation metric."""
    s = np.sum(h)
    if s == 0:
        return np.inf
    ac = np.convolve(h, h)
    return abs(2 * N * np.max(ac) / (s ** 2))

def _reconstruct(x):
    """
    Reconstruct h from parameter vector x (vectorized).
    x = [a0, a1, ..., a_{K-1}, b1, ..., b_{K-1}]
    h[n] = a0 + sum_{k=1}^{K-1} (a_k * cos(2*pi*k*n/N) + b_k * sin(2*pi*k*n/N))
    """
    h = np.full(N, x[0])
    h += _cos_basis @ x[1:K]
    h += _sin_basis @ x[K:2*K]
    return h

def _soft_max(ac, beta=BETA):
    """Compute smooth approximation to max(ac) using log-sum-exp."""
    # Scale for numerical stability
    ac_scaled = ac / beta
    m = np.max(ac_scaled)
    if not np.isfinite(m):
        return 0.0
    log_sum_exp = m + np.log(np.sum(np.exp(ac_scaled - m)))
    return log_sum_exp * beta

def _soft_max_grad(ac, beta=BETA):
    """
    Gradient of soft_max with respect to ac.
    Returns weights w_i = exp(ac_i/beta) / sum_j exp(ac_j/beta)
    """
    ac_scaled = ac / beta
    m = np.max(ac_scaled)
    if not np.isfinite(m):
        return np.zeros_like(ac)
    exp_vals = np.exp(ac_scaled - m)
    weights = exp_vals / np.sum(exp_vals)
    return weights

def _objective_grad(x):
    """
    Compute objective value and analytic gradient using FFT-based autocorrelation.
    """
    h = _reconstruct(x)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(x)
    
    # Compute autocorrelation via FFT for speed
    # Full convolution length is 2*N - 1
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    AC_fft = H * np.conj(H)
    ac = np.real(np.fft.ifft(AC_fft))
    # ac has length n_fft, but we only need the first 2*N-1 entries
    # Actually, np.convolve(h,h) gives length 2*N-1
    # The FFT-based circular convolution of length n_fft=2N gives the same result
    # for the first 2N-1 elements (zero-padded to avoid wrap-around)
    ac = ac[:2*N-1]
    
    # Soft max
    sm = _soft_max(ac, BETA)
    w = _soft_max_grad(ac, BETA)  # weights for soft max
    
    # Objective: abs(2*N*sm / s^2)
    # For simplicity, assume s > 0 (we'll handle sign)
    val = abs(2 * N * sm / (s ** 2))
    
    # Gradient computation:
    # val = 2*N*|sm| / s^2  (assuming sm and s have consistent sign for derivative)
    # Let's compute d(val)/d(x_i)
    
    # d(sm)/d(h_j) = w_j (from soft max)
    # d(sm)/d(x_i) = sum_j w_j * dh_j/dx_i
    # dh_j/dx_0 = 1
    # dh_j/dx_k (k>=1) = cos_basis[k-1, j]
    # dh_j/dx_{K+k-1} (k>=1) = sin_basis[k-1, j]
    
    # d(s)/d(x_i):
    # ds/dx_0 = N
    # ds/dx_k = sum_j cos_basis[k-1, j] = 0 for k>=1
    # ds/dx_{K+k-1} = 0 for k>=1
    
    # So:
    # d(sm)/dx_0 = sum_j w_j = 1
    # d(sm)/dx_k = sum_j w_j * cos_basis[k-1, j] = w @ cos_basis[k-1, :]
    # d(sm)/dx_{K+k-1} = w @ sin_basis[k-1, :]
    
    # d(s)/dx_0 = N, d(s)/dx_i = 0 for i > 0
    
    # val = 2*N*sm/s^2 (assuming positive)
    # d(val)/dx_0 = 2*N * (1/s^2 * d(sm)/dx_0 + sm * d(1/s^2)/dx_0)
    #            = 2*N * (1/s^2 * 1 + sm * (-2/s^3) * N)
    #            = 2*N/s^2 - 4*N^2*sm/s^3
    
    # d(val)/dx_i (i>0) = 2*N/s^2 * d(sm)/dx_i
    
    grad = np.zeros_like(x)
    if abs(s) > 1e-12:
        s2 = s * s
        s3 = s2 * s
        
        # Gradient of sm w.r.t. x
        dsm_dx0 = 1.0
        dsm_dx_cos = w @ _cos_basis  # shape (K-1,)
        dsm_dx_sin = w @ _sin_basis  # shape (K-1,)
        
        # Gradient of val w.r.t. x_0
        grad[0] = 2 * N / s2 - 4 * N * N * sm / s3
        
        # Gradient of val w.r.t. x_k (k >= 1)
        grad[1:K] = 2 * N / s2 * dsm_dx_cos
        grad[K:2*K] = 2 * N / s2 * dsm_dx_sin
        
        # Apply sign if needed (for abs)
        if 2 * N * sm / (s ** 2) < 0:
            grad = -grad
    
    return val, grad

def _objective(x):
    """Objective function for scipy.optimize."""
    val, _ = _objective_grad(x)
    return val

def _generate_starts():
    """Generate diverse starting points."""
    starts = []
    np.random.seed(42)
    
    # Seed start: all ones
    x_seed = np.zeros(DIM)
    x_seed[0] = 1.0
    starts.append(x_seed)
    
    # Block patterns
    for block_size in [10, 20, 40, 50, 80]:
        x_block = np.zeros(DIM)
        x_block[0] = 0.5
        # Add some low-frequency structure
        for k in range(1, min(K, 5)):
            x_block[k] = np.random.randn() * 0.3
            x_block[K + k - 1] = np.random.randn() * 0.3
        starts.append(x_block)
    
    # Gaussian-filtered randoms
    for _ in range(5):
        x_gauss = np.zeros(DIM)
        x_gauss[0] = 1.0 + np.random.randn() * 0.2
        # Low-pass filtered random
        for k in range(1, K):
            decay = np.exp(-k / 5.0)
            x_gauss[k] = np.random.randn() * 0.5 * decay
            x_gauss[K + k - 1] = np.random.randn() * 0.5 * decay
        starts.append(x_gauss)
    
    # Multi-frequency cosine patterns
    for k_main in [2, 5, 10, 15, 20]:
        if k_main >= K:
            continue
        x_cos = np.zeros(DIM)
        x_cos[0] = 1.0
        x_cos[k_main] = 0.8
        # Add some noise
        for k in range(1, K):
            if k != k_main:
                x_cos[k] = np.random.randn() * 0.1
                x_cos[K + k - 1] = np.random.randn() * 0.1
        starts.append(x_cos)
    
    # Pure random with positive DC
    for _ in range(5):
        x_rand = np.random.randn(DIM) * 0.3
        x_rand[0] = 1.0 + np.random.randn() * 0.1
        starts.append(x_rand)
    
    return starts

def construct():
    """
    Construct the optimal step heights using optimization.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    starts = _generate_starts()
    
    best_x = None
    best_val = np.inf
    best_h = np.ones(N)
    
    # Phase 1: Multi-start L-BFGS-B
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(
                _objective,
                x0,
                jac=lambda x: _objective_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            h_candidate = _reconstruct(res.x)
            val = _value(h_candidate)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
                best_h = h_candidate.copy()
        except Exception:
            continue
    
    # Phase 2: Polish the best result
    if best_x is not None and time.time() < deadline - 10:
        try:
            res_final = minimize(
                _objective,
                best_x,
                jac=lambda x: _objective_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            h_candidate = _reconstruct(res_final.x)
            val = _value(h_candidate)
            if val < best_val:
                best_val = val
                best_x = res_final.x.copy()
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Phase 3: Try a few more starts if time permits
    if time.time() < deadline - 20:
        np.random.seed(123)
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            x_extra = np.random.randn(DIM) * 0.2
            x_extra[0] = 1.0 + np.random.randn() * 0.1
            try:
                res = minimize(
                    _objective,
                    x_extra,
                    jac=lambda x: _objective_grad(x)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_candidate = _reconstruct(res.x)
                val = _value(h_candidate)
                if val < best_val:
                    best_val = val
                    best_x = res.x.copy()
                    best_h = h_candidate.copy()
            except Exception:
                continue
    
    if best_x is not None:
        return best_h
    
    # Fallback to seed
    return np.ones(N)

# EVOLVE-BLOCK-END
