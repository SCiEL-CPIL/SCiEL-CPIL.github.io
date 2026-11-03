"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses FFT-accelerated overlap computation and vectorized gradient via FFT.
Multi-phase optimization with increasing beta for refinement."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective using FFT-based vectorized computation.
    Returns (value, gradient)
    
    The gradient of C_k w.r.t. h_j is:
    dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k else 0 + h_{j+k} if j+k<n else 0)
    
    This can be computed as: for each k, the contribution to grad is
    -(2/n) * w[k] * (shifted_h_down_k + shifted_h_up_k)
    which is equivalent to a convolution-like operation.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        w = np.full(n, 1.0 / n)
        val = m
    else:
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
    
    # Compute gradient using FFT-based approach
    # grad[j] = sum_k w[k] * dC_k/dh_j
    # dC_k/dh_j = -(2/n) * (h_{j-k} * [j>=k] + h_{j+k} * [j+k<n])
    
    # Term 1: sum_k w[k] * h_{j-k} for j>=k
    # This is like a cross-correlation of w with h
    # For each j, sum over k=0..j of w[k] * h[j-k]
    
    # Term 2: sum_k w[k] * h_{j+k} for j+k<n
    # For each j, sum over k=0..n-1-j of w[k] * h[j+k]
    
    # Use FFT for both terms
    n_fft = 2 * n
    
    # Term 1: (w * h)[j] = sum_{k=0}^{j} w[k] * h[j-k] for j=0..n-1
    # This is linear convolution of w and h, taking first n elements
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # Term 2: sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # This is cross-correlation: (w ⋆ h)[j] = sum_k w[k] * h[j+k]
    # = sum_k w[k] * h[j+k] = correlation
    # Using FFT: correlate w with h
    # corr[j] = sum_k w[k] * h[j+k] for valid j
    # This is equivalent to: reverse w, convolve with h, then shift
    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:n] = w_rev
    W_rev = np.fft.rfft(w_rev_pad, n=n_fft)
    conv2_full = np.fft.irfft(W_rev * H, n=n_fft)
    # conv2_full[n-1-j] = sum_k w_rev[k] * h[n-1-j+k] = sum_k w[n-1-k] * h[n-1-j+k]
    # Let k' = n-1-k, then = sum_{k'} w[k'] * h[j-k'+n-1]... this is getting complex
    # 
    # Alternative: Term 2[j] = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # = sum_{m=j}^{n-1} w[m-j] * h[m] where m = j+k
    # = (w flipped) convolved with h, at position n-1-j
    # 
    # Actually, let's use a simpler approach:
    # Term 2[j] = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # This is the cross-correlation of w and h at lag j
    # Using FFT: ifft(fft(w) * conj(fft(h))) gives circular cross-correlation
    # For linear, we need to be careful with padding
    
    # Simpler: compute term2 directly using matrix-free approach with FFT
    # term2[j] = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # = <w[0:n-1-j], h[j:n]>
    
    # Use the fact that this is a Toeplitz-like operation
    # Let's compute it via FFT:
    # Define v[j] = sum_{k=0}^{n-1} w[k] * h[j+k] * [j+k < n]
    # = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # This is the cross-correlation (w ⋆ h)[j]
    
    # Cross-correlation via FFT:
    # corr[j] = sum_k w[k] * h[j+k]
    # = ifft(fft(w) * conj(fft(h))) in circular sense
    # For linear: pad both to 2n, compute ifft(fft(w_pad) * conj(fft(h_pad)))
    
    # Actually, the standard formula:
    # (w ⋆ h)[j] = sum_k w[k] * h[j+k]
    # Using FFT: ifft(fft(w) * conj(fft(h))) gives sum_k w[k] * conj(h)[k-j]... 
    # Let me use a different approach.
    
    # term2[j] = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # = sum_{m=j}^{n-1} w[m-j] * h[m]
    # Let u[m] = w[m] for m=0..n-1, and we want sum_m u[m-j] * h[m] for m=j..n-1
    # = sum_m u[m-j] * h[m] * [m>=j]
    
    # This is the convolution of u and h, but with u indexed as u[m-j]
    # = (h * u_flipped)[n-1-j] where u_flipped[m] = u[n-1-m]
    
    # Let's just compute it directly for n=100 since it's fast enough
    # But we want FFT speed...
    
    # Actually for n=100, the O(n^2) loop is 10000 ops which is trivial.
    # The FFT overhead might not be worth it for n=100.
    # Let's use the direct approach but vectorized.
    
    # term1[j] = sum_{k=0}^{j} w[k] * h[j-k] = conv(w, h)[j]
    # term2[j] = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    
    # For term2, note that:
    # term2[j] = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # Let's define g[j] = term2[j]
    # g[j] = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    
    # This is equivalent to: for the array w, take the first n-1-j elements,
    # and dot with h[j:n]
    
    # We can compute all g[j] using a Toeplitz matrix or via FFT
    # Using FFT: g = ifft(fft(w) * conj(fft(h))) gives circular correlation
    # For linear correlation with proper padding:
    
    n_fft2 = 2 * n
    w_pad2 = np.zeros(n_fft2)
    w_pad2[:n] = w
    h_pad2 = np.zeros(n_fft2)
    h_pad2[:n] = h
    
    W2 = np.fft.rfft(w_pad2, n=n_fft2)
    H2 = np.fft.rfft(h_pad2, n=n_fft2)
    
    # Cross-correlation: sum_k w[k] * h[j+k]
    # = ifft(fft(w) * conj(fft(h))) in the circular case
    # For linear: the result at index j (for j=0..n-1) is what we want
    # But we need to be careful about the indexing
    
    # Standard: corr[j] = sum_k w[k] * h[k+j] for j=0..n-1
    # Using FFT: corr = ifft(fft(w) * conj(fft(h)))
    # This gives circular correlation. For linear, with zero-padding to 2n:
    corr_full = np.fft.irfft(W2 * np.conj(H2), n=n_fft2)
    # corr_full[j] for j=0..n-1 should give sum_k w[k] * h[k+j] (with h[k+j]=0 if k+j>=n)
    # Actually, with zero-padding, this should be the linear cross-correlation
    term2 = corr_full[:n]
    
    # Now compute gradient
    factor = -(2.0 / n)
    grad = factor * (conv1 + term2)
    
    return val, grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Analytical Starts
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _fix_sum(h_sin)
    _update_best(h_sin)
    
    # 2. Optimization with FFT-accelerated gradient
    
    def _run_optimization(h0, beta, maxiter, ftol):
        nonlocal best_h, best_score
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        def obj_func(x):
            val, _ = _smooth_max_grad(x, beta)
            return val
            
        def jac_func(x):
            _, grad = _smooth_max_grad(x, beta)
            return grad
            
        try:
            res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    # Phase 1: Refine Cosine Start with moderate beta
    if time.time() - t0 < _DEADLINE - 50:
        _run_optimization(h_cos, beta=50.0, maxiter=300, ftol=1e-9)
        
    # Phase 2: Refine Sine Start
    if time.time() - t0 < _DEADLINE - 40:
        _run_optimization(h_sin, beta=50.0, maxiter=200, ftol=1e-9)
        
    # Phase 3: Try random starts if time permits
    if time.time() - t0 < _DEADLINE - 30:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 20:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            _run_optimization(h_rand, beta=50.0, maxiter=100, ftol=1e-8)
            
    # Phase 4: Epigraph polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        def obj(x):
            return x[N]
        
        def jac(x):
            _, gh = _smooth_max_grad(x[:N], 200.0)
            return np.concatenate([gh, [1.0]])
        
        cons = []
        for k in range(N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, k=k: x[N] - _linear_overlap_values(x[:N])[k],
                'jac': lambda x, k=k: None
            })
        cons.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x[:N]) - N/2.0,
            'jac': lambda x: np.concatenate([np.ones(N), [0.0]])
        })
        
        try:
            x0 = np.concatenate([best_h, [best_score + 0.01]])
            bounds = [(0.0, 1.0)] * N + [(0.0, 2.0)]
            
            res = minimize(obj, x0, jac=jac, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 500, 'ftol': 1e-12})
            if res.success or res.fun < best_score:
                _update_best(res.x[:N])
        except Exception:
            pass
        
    # Phase 5: Final high-beta polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        _run_optimization(best_h, beta=200.0, maxiter=500, ftol=1e-14)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
