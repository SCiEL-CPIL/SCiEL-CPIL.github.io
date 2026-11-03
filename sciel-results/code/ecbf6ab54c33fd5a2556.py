"""
Rewritten program: Spectral Subspace Optimization with Symmetric Basis + Local Search.

Strategy:
1. Represent the sequence h in a low-dimensional subspace spanned by symmetric basis functions.
   This enforces approximate time-reversal symmetry and reduces dimensionality.
2. Optimize the coefficients in this subspace using L-BFGS on a smooth surrogate objective.
3. Expand to full dimension and refine using discrete local search (block sign flips).
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# EVOLVE-BLOCK-START

def _autoconv(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2 * N - 1]

def _raw_value(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    return float(np.max(np.abs(ac))) / (s * s)

def _soft_max_abs(ac, beta=30.0):
    """Smooth approximation of max|ac| via log-sum-exp."""
    a = np.abs(ac)
    m = np.max(a)
    if m < 1e-15:
        return 0.0, None
    scaled = beta * (a - m)
    scaled = np.clip(scaled, -60, 60)
    e = np.exp(scaled)
    log_sum = np.log(np.sum(e))
    weights = e / np.sum(e)
    return m + log_sum / beta, weights

def _smooth_objective(h, beta=30.0):
    """Smooth objective: soft_max_abs(ac) / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    sm, _ = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _smooth_gradient(h, beta=30.0):
    """
    Gradient of soft_max_abs(ac) / (sum h)^2 with respect to h.
    Uses the chain rule: grad_h = (1/s^2) * grad_soft - (2*soft/s^3) * ones.
    grad_soft is computed via FFT cross-correlation.
    """
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(h)
    
    ac = _autoconv(h)
    sm, weights = _soft_max_abs(ac, beta)
    
    if weights is None:
        return np.zeros_like(h)
    
    # g_ac[m] = weights[m] * sign(ac[m])
    g_ac = weights * np.sign(ac)
    
    # grad_soft[j] = sum_m g_ac[m] * d(ac[m])/d(h[j])
    # d(ac[m])/d(h[j]) = h[m-j] + h[j-m] (with boundary handling)
    # This is 2 * (g_ac * h)[j] in circular convolution sense, but we have linear.
    # Using FFT: 
    # Let G = rfft(g_ac, _FSIZE)
    # Let H = rfft(h, _FSIZE)
    # The gradient of the autoconvolution sum is related to the cross-correlation of g_ac and h.
    # Specifically, if L = sum_m g[m] ac[m], then dL/dh[j] = sum_m g[m] (h[m-j] + h[j-m]).
    # This is equivalent to 2 * (g * h)[j] if we treat them as circular convolutions of size _FSIZE,
    # provided the linear convolution is embedded correctly.
    
    G = np.fft.rfft(g_ac, _FSIZE)
    H = np.fft.rfft(h, _FSIZE)
    
    # Cross-correlation via FFT: ifft(conj(G) * H) gives (g * h)[j] = sum_k g[k] h[j-k]
    # We need sum_m g[m] h[m-j]. Let k = m-j => m = k+j. sum_k g[k+j] h[k].
    # This is the cross-correlation of g and h at lag j.
    # np.correlate(g, h, 'full') does this, but FFT is faster.
    # corr[j] = sum_k g[k+j] h[k] = ifft(conj(G) * H)[j] ... wait.
    # DFT of h[k] is H. DFT of g[k+j] is G * exp(i 2 pi j k / N).
    # Actually, cross-correlation R_g_h[j] = sum_k g[k+j] h[k].
    # R = ifft(conj(G) * H) computes sum_k g[k] h[k-j]? 
    # Let's just use the property: grad = 2 * ifft(conj(G) * H) might not be exactly right for linear.
    
    # Safer: Direct O(N^2) is 160k ops, which is ~1ms. N=400.
    # Let's stick to the direct method for robustness and simplicity, as it was in the previous version.
    grad_h = np.zeros(N)
    size = 2 * N - 1
    for j in range(N):
        # First term: sum_m g_ac[m] * h[m-j] for m in [j, j+N-1]
        m_start1 = j
        m_end1 = min(j + N, size)
        if m_end1 > m_start1:
            grad_h[j] += np.sum(g_ac[m_start1:m_end1] * h[:m_end1-m_start1])
        
        # Second term: sum_m g_ac[m] * h[j-m] for m in [max(0, j-N+1), min(j, size-1)]
        m_start2 = max(0, j - N + 1)
        m_end2 = min(j + 1, size)
        if m_end2 > m_start2:
            idx = j - np.arange(m_start2, m_end2)
            grad_h[j] += np.sum(g_ac[m_start2:m_end2] * h[idx])
            
    # Total gradient of objective f = sm / s^2
    # df/dh = (grad_sm * s^2 - sm * 2 * s * ds/dh) / s^4
    # df/dh = grad_sm / s^2 - 2 * sm / s^3
    df = grad_h / (s * s) - 2 * sm / (s * s * s) * np.ones(N)
    
    return df

def _generate_symmetric_basis(M, N):
    """
    Generate M symmetric basis vectors of length N.
    We use a cosine-like basis to enforce symmetry h[i] = h[N-1-i].
    Basis function j: cos(pi * j * (i - (N-1)/2) / ((N-1)/2))
    This ensures the basis vectors themselves are symmetric.
    """
    x = np.arange(N)
    center = (N - 1) / 2.0
    basis = np.zeros((M, N))
    for j in range(M):
        # j=0 is constant, j=1 is linear (anti-symmetric? no, cos is symmetric)
        # cos(0) = 1
        # cos(pi * j * (x - center) / center)
        # At x=0: cos(-pi*j) = (-1)^j
        # At x=N-1: cos(pi*j) = (-1)^j
        # So it is symmetric.
        if j == 0:
            basis[j, :] = 1.0
        else:
            basis[j, :] = np.cos(np.pi * j * (x - center) / center)
    return basis

def _optimize_in_subspace(basis, init_coeffs, beta=30.0, max_iter=200, deadline=None):
    """
    Optimize coefficients in the subspace defined by basis.
    h = basis @ coeffs
    """
    M = len(init_coeffs)
    
    def fun(c):
        if deadline is not None and time.time() > deadline:
            return 1e10
        h = basis @ c
        return _smooth_objective(h, beta)
    
    def jac(c):
        if deadline is not None and time.time() > deadline:
            return np.zeros(M)
        h = basis @ c
        grad_h = _smooth_gradient(h, beta)
        # Chain rule: grad_c = basis.T @ grad_h
        return basis.T @ grad_h
        
    try:
        res = minimize(fun, init_coeffs, method='L-BFGS-B', jac=jac,
                       options={'maxiter': max_iter, 'gtol': 1e-5, 'maxfun': 1000})
        return res.x
    except Exception:
        return init_coeffs

def _block_sign_flip_local_search(h, block_sizes=None, max_passes=3, deadline=None):
    """
    Local search: try flipping signs of blocks of consecutive elements.
    """
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    h = h.copy()
    best_val = _raw_value(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for bs in block_sizes:
            if deadline is not None and time.time() > deadline:
                break
            
            for start in range(0, N, bs):
                if deadline is not None and time.time() > deadline:
                    break
                
                end = min(start + bs, N)
                h[start:end] *= -1
                new_val = _raw_value(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[start:end] *= -1  # revert
        
        if not improved:
            break
    
    return h, best_val

def _single_flip_local_search(h, max_passes=5, deadline=None):
    """
    Local search: try flipping each individual element's sign.
    """
    h = h.copy()
    best_val = _raw_value(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            h[i] *= -1
            new_val = _raw_value(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] *= -1  # revert
        
        if not improved:
            break
    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Base score
    base_score = _raw_value(best_h)
    best_score = base_score
    
    # Strategy: Spectral Subspace Optimization
    # We try different subspace dimensions M
    # And different random initializations of the coefficients
    
    rng = np.random.default_rng(42)
    
    # List of subspace dimensions to try
    M_list = [10, 20, 30, 50, 80]
    
    # Generate bases for each M
    # Note: We use the same basis structure (cosine symmetric) for all M, just truncated.
    # This allows us to reuse computations if we were nesting, but here we just run separate optimizations.
    
    for M in M_list:
        if time.time() > deadline - 10:
            break
            
        basis = _generate_symmetric_basis(M, N)
        
        # Try a few random initializations for this M
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            # Random initial coefficients
            # Scale them so that sum(h) is roughly 1
            c_init = rng.standard_normal(M)
            h_init = basis @ c_init
            s_init = np.sum(h_init)
            if abs(s_init) > 1e-6:
                c_init = c_init * (1.0 / s_init)
            
            # Optimize in subspace
            c_opt = _optimize_in_subspace(basis, c_init, beta=30.0, max_iter=300, deadline=deadline - 5)
            
            # Expand to full h
            h_full = basis @ c_opt
            
            # Ensure sum is 1 (normalization helps with the ratio invariance, though the optimizer handles it)
            s_full = np.sum(h_full)
            if abs(s_full) > 1e-6:
                h_full = h_full * (1.0 / s_full)
            
            # Evaluate
            val = _raw_value(h_full)
            if val < best_score:
                best_score = val
                best_h = h_full.copy()
                
            # Refine with local search
            if time.time() > deadline - 15:
                continue
                
            h_refined, val_refined = _block_sign_flip_local_search(h_full, 
                                                                   block_sizes=[1, 2, 4, 8, 16],
                                                                   max_passes=2,
                                                                   deadline=deadline - 10)
            if val_refined < best_score:
                best_score = val_refined
                best_h = h_refined.copy()
                
            # Final single flip polish
            if time.time() > deadline - 5:
                continue
                
            h_polished, val_polished = _single_flip_local_search(best_h, max_passes=3,
                                                                 deadline=deadline - 2)
            if val_polished < best_score:
                best_score = val_polished
                best_h = h_polished.copy()

    # Final check
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N) / N
        best_score = _raw_value(best_h)
        
    return best_h

# EVOLVE-BLOCK-END
