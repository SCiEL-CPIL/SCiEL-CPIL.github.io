"""
Structural change: Fourier-mode parameterization (low-dim) + log-sum-exp smooth surrogate
for gradient-based optimization, followed by FFT-based direct polish on the 600 bins.

This combines:
- The low-dimensional Fourier representation from the inspiration (reduces 600-dim to ~21-dim)
- The fine-grid FFT autoconvolution evaluation from the current program (accurate scoring)
- A smooth log-sum-exp surrogate for differentiability
- Direct coordinate polish on the full 600 bins at the end
"""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 600
W = 1.0 / 1200.0  # width of each interval

# Fine grid for accurate autoconvolution
FINE_N = 3000
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


def fourier_to_heights(params, K):
    """
    Convert Fourier parameters to nonnegative step heights.
    params: [a0, a1, b1, a2, b2, ..., aK, bK]
    h[i] = max(0, a0 + sum_{k=1}^{K} a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
    """
    a0 = params[0]
    i = np.arange(N)
    heights = np.full(N, a0, dtype=np.float64)
    for k in range(1, K + 1):
        ak = params[2 * k - 1]
        bk = params[2 * k]
        heights += ak * np.cos(2.0 * np.pi * k * i / N)
        heights += bk * np.sin(2.0 * np.pi * k * i / N)
    heights = np.clip(heights, 0.0, None)
    return heights


def smooth_objective(params, K, tau=0.01):
    """
    Smooth surrogate: log-sum-exp of conv / (integral)^2.
    Uses fine-grid FFT for accuracy.
    """
    h = fourier_to_heights(params, K)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX

    # Log-sum-exp for smooth max
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    return lse / integral_sq


def epigraph_objective(params, K, tau=0.01):
    """
    Epigraph formulation: minimize t + penalty for violations.
    params: [a0, a1, b1, ..., aK, bK, t]
    Objective: t * integral_sq + tau * sum(exp((conv_i - t)/tau))
    This directly optimizes the worst-case level t.
    """
    fourier_params = params[:-1]
    t = params[-1]
    
    h = fourier_to_heights(fourier_params, K)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    # Epigraph: minimize t * integral_sq + tau * sum(exp((conv_i - t)/tau))
    # This is smooth and the optimal t will be near max(conv)
    m = np.max(conv)
    if m < 1e-15:
        return t * integral_sq
    
    # Soft penalty for conv exceeding t
    # Use log-sum-exp centered at t for numerical stability
    shifted = (conv - t) / tau
    max_shifted = np.max(shifted)
    if max_shifted > 700:  # overflow protection
        lse_term = tau * (max_shifted + np.log(np.sum(np.exp(shifted - max_shifted))))
    else:
        lse_term = tau * np.log(np.sum(np.exp(shifted)))
    
    return t * integral_sq + lse_term


def smooth_gradient(params, K, tau=0.01):
    """
    Numerical gradient of the smooth objective (central differences on a few coords).
    For L-BFGS-B we provide the function only; let it use finite differences.
    But to be faster, we compute gradient via finite differences ourselves.
    """
    h = fourier_to_heights(params, K)
    s = h.sum()
    if s < 1e-15:
        return np.zeros_like(params)

    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX

    m = np.max(conv)
    if m < 1e-15:
        return np.zeros_like(params)

    # softmax weights
    exp_vals = np.exp((conv - m) / tau)
    w = exp_vals / np.sum(exp_vals)

    integral_sq = (s * W) ** 2

    # Gradient via finite differences
    grad = np.zeros_like(params)
    eps = 1e-6
    for j in range(len(params)):
        p_plus = params.copy()
        p_plus[j] += eps
        p_minus = params.copy()
        p_minus[j] -= eps
        fp = smooth_objective(p_plus, K, tau)
        fm = smooth_objective(p_minus, K, tau)
        grad[j] = (fp - fm) / (2 * eps)

    return grad


def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)

    np.random.seed(42)

    # Number of Fourier modes
    K = 10

    # Generate diverse initializations in Fourier parameter space
    inits = []

    # 1. Near-uniform with small perturbations
    for _ in range(5):
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0  # base level
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.1)
            p0[2 * k] = np.random.normal(0, 0.1)
        inits.append(p0)

    # 2. Stronger modulation
    for _ in range(5):
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.3)
            p0[2 * k] = np.random.normal(0, 0.3)
        inits.append(p0)

    # 3. Dominant low-frequency modes
    for k_dom in [1, 2, 3]:
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        p0[2 * k_dom - 1] = np.random.uniform(0.2, 0.5)
        p0[2 * k_dom] = np.random.uniform(0.2, 0.5)
        for k in range(1, K + 1):
            if k != k_dom:
                p0[2 * k - 1] = np.random.normal(0, 0.05)
                p0[2 * k] = np.random.normal(0, 0.05)
        inits.append(p0)

    # 4. Asymmetric (more sine than cos)
    for _ in range(5):
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.15)
            p0[2 * k] = np.random.normal(0, 0.3)
        inits.append(p0)

    # 5. Random
    for _ in range(5):
        p0 = np.zeros(2 * K + 1)
        p0[0] = np.random.uniform(0.5, 1.5)
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.4)
            p0[2 * k] = np.random.normal(0, 0.4)
        inits.append(p0)

    # Optimize from each initialization using L-BFGS-B on the smooth surrogate
    for p0 in inits:
        if time.time() > deadline - 30.0:
            break

        try:
            res = minimize(
                smooth_objective,
                p0,
                args=(K, 0.01),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )

            if np.isfinite(res.fun):
                h_candidate = fourier_to_heights(res.x, K)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Try epigraph formulation with auxiliary variable t
    if time.time() < deadline - 25.0:
        for p0 in inits[:8]:
            if time.time() > deadline - 20.0:
                break
            try:
                # Initialize t to current best ratio * integral_sq estimate
                h0 = fourier_to_heights(p0, K)
                s0 = h0.sum()
                f_fine0 = h0[CELL]
                F0 = np.fft.rfft(f_fine0, N_FFT)
                conv0 = np.fft.irfft(F0 * F0, N_FFT)[:2 * FINE_N - 1]
                conv0 *= DX
                t0 = np.max(conv0) if np.max(conv0) > 0 else 0.1
                
                p0_epigraph = np.concatenate([p0, [t0]])
                
                res = minimize(
                    epigraph_objective,
                    p0_epigraph,
                    args=(K, 0.01),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )

                if np.isfinite(res.fun):
                    h_candidate = fourier_to_heights(res.x[:-1], K)
                    val = ratio_from_h(h_candidate)
                    if val < best_val and np.isfinite(val):
                        best_val = val
                        best_h = h_candidate.copy()
            except Exception:
                pass

    # Try with smaller tau for sharper max approximation
    if time.time() < deadline - 15.0:
        for p0 in inits[:5]:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    smooth_objective,
                    p0,
                    args=(K, 0.005),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if np.isfinite(res.fun):
                    h_candidate = fourier_to_heights(res.x, K)
                    val = ratio_from_h(h_candidate)
                    if val < best_val and np.isfinite(val):
                        best_val = val
                        best_h = h_candidate.copy()
            except Exception:
                pass

    # Coordinate descent polish on the 600 bin heights
    if time.time() < deadline - 5.0:
        # Define a helper for 1D minimization
        def minimize_bin(i, current_h, current_val):
            def obj_1d(h_i):
                h_temp = current_h.copy()
                h_temp[i] = max(0.0, h_i)
                return ratio_from_h(h_temp)
            
            try:
                res = minimize_scalar(obj_1d, bounds=(0, 5.0), method='bounded', 
                                      options={'maxiter': 30, 'xatol': 1e-7})
                if res.fun < current_val:
                    return max(0.0, res.x), res.fun, True
            except Exception:
                pass
            return current_h[i], current_val, False

        # Perform passes
        while time.time() < deadline - 3.0:
            improved = False
            for i in range(N):
                if time.time() > deadline - 3.0:
                    break
                
                new_h_i, new_val, is_improved = minimize_bin(i, best_h, best_val)
                if is_improved:
                    best_h[i] = new_h_i
                    best_val = new_val
                    improved = True
            
            if not improved:
                # Try random restarts/perturbations if stuck
                for _ in range(20):
                    if time.time() > deadline - 3.0:
                        break
                    idx = np.random.choice(N, 10, replace=False)
                    h_try = best_h.copy()
                    h_try[idx] += np.random.randn(10) * 0.2
                    h_try = np.clip(h_try, 0, None)
                    val = ratio_from_h(h_try)
                    if val < best_val:
                        best_val = val
                        best_h = h_try.copy()
                        improved = True
                if not improved:
                    break

    # High-precision final polish: tighter tolerances, more iterations
    if time.time() < deadline - 1.0:
        # Multiple passes with very tight xatol
        for pass_num in range(3):
            if time.time() > deadline - 1.0:
                break
            for i in range(N):
                if time.time() > deadline - 1.0:
                    break
                def obj_1d_precise(h_i):
                    h_temp = best_h.copy()
                    h_temp[i] = max(0.0, h_i)
                    return ratio_from_h(h_temp)
                
                try:
                    res = minimize_scalar(obj_1d_precise, bounds=(0, 5.0), method='bounded', 
                                          options={'maxiter': 50, 'xatol': 1e-10})
                    if np.isfinite(res.fun) and res.fun < best_val:
                        best_h[i] = max(0.0, res.x)
                        best_val = res.fun
                except Exception:
                    pass

    return list(best_h)
