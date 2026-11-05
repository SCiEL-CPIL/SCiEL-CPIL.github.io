"""Combined program for third autocorrelation inequality optimization.
Structural change: direct h-space optimization with analytical gradient
(inspiration 1) wrapped in the multi-phase search and DCT/Chebyshev
representation machinery (current program)."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft, dct, idct
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _soft_max_grad(x, beta):
    """Soft maximum and its gradient."""
    m = np.max(x)
    if not np.isfinite(m):
        return m, np.zeros_like(x)
    exp_vals = np.exp(beta * (x - m))
    w = exp_vals / np.sum(exp_vals)
    sm = m + np.log(np.sum(exp_vals)) / beta
    grad = beta * w
    return sm, grad

def _chebyshev_basis(M, x):
    """
    Compute Chebyshev polynomials of the second kind U_0(x), ..., U_{M-1}(x)
    evaluated at points x. Returns array of shape (M, len(x)).
    Uses recurrence: U_0=1, U_1=2x, U_k=2x*U_{k-1} - U_{k-2}
    """
    n = len(x)
    basis = np.zeros((M, n))
    if M > 0:
        basis[0] = 1.0
    if M > 1:
        basis[1] = 2.0 * x
    for k in range(2, M):
        basis[k] = 2.0 * x * basis[k-1] - basis[k-2]
    return basis

def _objective_and_grad(h, beta):
    """
    Soft-max objective with analytical gradient in h-space.
    This is the core structural improvement: direct optimization in h-space
    with exact gradient rather than relying on numerical gradients through
    DCT/Chebyshev transforms.
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = _convolve_fft(h)
    sm, weights = _soft_max_grad(conv, beta)
    
    # Analytical gradient: d(sm)/d(h[i]) = sum_k weights[k] * d(conv[k])/d(h[i])
    # For full convolution: conv[k] = sum_j h[j]*h[k-j]
    # d(conv[k])/d(h[i]) = h[k-i] if 0<=k-i<N, plus h[k-i] if k-i != i and 0<=k-i<N
    # i.e., factor is 2*h[k-i] when k-i != i, and h[k-i] when k-i == i
    # More precisely: if 0 <= k-i < N, the term h[i]*h[k-i] contributes h[k-i]
    # and if k-i != i, the term h[k-i]*h[i] also contributes h[k-i]
    # So d(conv[k])/d(h[i]) = (2 if k!=2*i else 1) * h[k-i] if 0<=k-i<N else 0
    
    n_conv = 2 * N - 1
    grad_h = np.zeros(N)
    
    for i in range(N):
        # k ranges over 0..2N-2, we need 0 <= k-i < N, so k in [i, i+N-1]
        k_idx = np.arange(i, i + N)
        j_idx = np.arange(N)  # j = k - i
        w = weights[k_idx]
        h_j = h[j_idx]
        # Factor: 2 if j != i, 1 if j == i
        factor = np.full(N, 2.0)
        factor[i] = 1.0
        grad_h[i] = np.sum(w * factor * h_j)
    
    # val = 2*N*sm / s^2
    # d(val)/d(h[i]) = (2*N/s^2) * grad_h[i] - (4*N*sm/s^3) * 1
    coeff_sm = 2.0 * N / (s * s)
    coeff_s = -4.0 * N * sm / (s ** 3)
    
    grad = coeff_sm * grad_h + coeff_s
    val = 2.0 * N * sm / (s * s)
    if not np.isfinite(val):
        return 1e10, np.zeros_like(h)
    
    return val, grad

def _objective_h(h, beta):
    """Objective only for h-space optimization."""
    val, _ = _objective_and_grad(h, beta)
    return val

def _objective_h_and_grad(h, beta):
    """Objective and gradient for h-space optimization."""
    return _objective_and_grad(h, beta)

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Combines direct h-space optimization with analytical gradients
    and multi-representation search (DCT, Chebyshev) for diversity.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit

    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0

    def try_update(h):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h)) or abs(np.sum(h)) < 1e-12:
            return
        v = _eval_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Phase 1: Quick parametric families to get a good starting point
    for m in range(1, 21):
        if time.time() > deadline - 90:
            break
        for a in np.linspace(-0.8, 0.8, 17):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            try_update(h)

    # Family 2: Smoothed block
    for beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 90:
            break
        h = np.tanh(beta * (np.arange(N) - N / 2.0))
        if abs(np.sum(h)) > 1e-6:
            try_update(h)

    # Phase 2: Diverse initial candidates (from inspiration 1)
    rng = np.random.default_rng(42)
    
    # Random signals with small mean
    for _ in range(15):
        if time.time() > deadline - 85:
            break
        h = rng.standard_normal(N)
        mean_h = np.mean(h)
        h = h - mean_h + rng.uniform(-0.01, 0.01)
        try_update(h)
    
    # Random walk with drift
    for seed in range(10):
        if time.time() > deadline - 85:
            break
        rng_local = np.random.default_rng(seed)
        drift = rng_local.uniform(0.01, 0.5)
        noise = rng_local.uniform(0.5, 2.0)
        steps = rng_local.standard_normal(N) * noise - drift
        h = np.cumsum(steps)
        h = h - np.mean(h) + rng_local.uniform(-0.01, 0.01)
        try_update(h)
    
    # Sparse signals
    for _ in range(10):
        if time.time() > deadline - 85:
            break
        h = np.zeros(N)
        n_nonzero = rng.integers(5, 30)
        indices = rng.choice(N, size=n_nonzero, replace=False)
        h[indices] = rng.standard_normal(n_nonzero)
        h += rng.uniform(-0.01, 0.01)
        try_update(h)
    
    # Bimodal signals
    for _ in range(10):
        if time.time() > deadline - 85:
            break
        h = np.zeros(N)
        split = rng.integers(N // 3, 2 * N // 3)
        h[:split] = rng.uniform(0.5, 2.0)
        h[split:] = rng.uniform(-2.0, -0.5)
        current_sum = np.sum(h)
        if abs(current_sum) > 0.1:
            h -= current_sum / N
        try_update(h)

    # Phase 3: Direct h-space optimization with analytical gradient (STRUCTURAL CHANGE)
    # This is the key improvement: optimize directly in h-space using exact gradients
    for beta in [5.0, 20.0, 100.0, 400.0, 1000.0]:
        if time.time() > deadline - 30:
            break
        
        for restart in range(3):
            if time.time() > deadline - 28:
                break
            
            if restart == 0:
                init_h = best_h.copy()
            else:
                scale = 0.05 * np.std(best_h) if np.std(best_h) > 0 else 0.05
                init_h = best_h + rng.standard_normal(N) * scale
            
            # Ensure sum is not too small
            s = np.sum(init_h)
            if abs(s) < 0.01:
                init_h = init_h + 0.01 / N * np.ones(N)
            
            try:
                def obj_wrapper(x, beta=beta):
                    return _objective_h(x, beta)
                
                def grad_wrapper(x, beta=beta):
                    _, g = _objective_and_grad(x, beta)
                    return g
                
                result = minimize(
                    obj_wrapper,
                    init_h,
                    jac=grad_wrapper,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue

    # Phase 4: DCT-domain optimization (retained from current program for diversity)
    def _h_to_dct(h):
        return dct(h, type=2, norm='ortho')
    
    def _dct_to_h(coeffs):
        return idct(coeffs, type=2, norm='ortho')
    
    c_best = _h_to_dct(best_h)
    c_dc = c_best[0].copy()

    for beta in [20.0, 100.0, 400.0]:
        if time.time() > deadline - 15:
            break

        def obj_dct(c_ac, beta=beta):
            c_full = np.concatenate([[c_dc], c_ac])
            h_recon = _dct_to_h(c_full)
            if not np.all(np.isfinite(h_recon)):
                return 1e10
            s = np.sum(h_recon)
            if abs(s) < 1e-10:
                return 1e10
            h_norm = h_recon * (N / s)
            conv = _convolve_fft(h_norm)
            sm = _soft_max(conv, beta)
            return abs(2.0 * N * sm / (N * N))

        for trial in range(2):
            if time.time() > deadline - 13:
                break

            if trial == 0:
                c_ac_trial = c_best[1:].copy()
            else:
                c_ac_trial = c_best[1:] + 0.05 * rng.standard_normal(N-1)

            try:
                result = minimize(
                    obj_dct,
                    c_ac_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = _h_to_dct(best_h)
                        c_dc = c_best[0].copy()
            except Exception:
                continue

    # Phase 5: Chebyshev polynomial parameterization
    if time.time() < deadline - 12:
        x_grid = 2.0 * np.arange(N) / (N - 1) - 1.0
        
        for M in [6, 10, 14]:
            if time.time() > deadline - 10:
                break
            
            basis = _chebyshev_basis(M, x_grid)
            
            try:
                a_init, _, _, _ = np.linalg.lstsq(basis.T, best_h, rcond=None)
            except:
                a_init = np.zeros(M)
                a_init[0] = np.mean(best_h)
            
            for beta in [20.0, 100.0]:
                if time.time() > deadline - 8:
                    break
                
                def obj_cheb(a, beta=beta, basis=basis):
                    h_recon = basis.T @ a
                    if not np.all(np.isfinite(h_recon)):
                        return 1e10
                    s = np.sum(h_recon)
                    if abs(s) < 1e-10:
                        return 1e10
                    h_norm = h_recon * (N / s)
                    conv = _convolve_fft(h_norm)
                    sm = _soft_max(conv, beta)
                    return abs(2.0 * N * sm / (N * N))
                
                try:
                    result = minimize(
                        obj_cheb,
                        a_init.copy(),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    a_opt = result.x
                    h_opt = basis.T @ a_opt
                    
                    if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                        v_opt = _eval_value(h_opt)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_opt.copy()
                except Exception:
                    continue

    # Phase 6: Final high-precision polish in h-space
    if time.time() < deadline - 5:
        for beta in [500.0, 1000.0, 2000.0]:
            if time.time() > deadline - 3:
                break
            
            try:
                def obj_polish(x, beta=beta):
                    return _objective_h(x, beta)
                
                def grad_polish(x, beta=beta):
                    _, g = _objective_and_grad(x, beta)
                    return g
                
                result = minimize(
                    obj_polish,
                    best_h.copy(),
                    jac=grad_polish,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
