"""Rewritten program: spectral shaping via DCT basis + projected gradient descent."""
import numpy as np
import time

N = 400

def _value(h):
    """Compute the objective value (lower is better)."""
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s * s))

def _grad(h):
    """Gradient of the objective w.r.t. h.
    
    value = |2N * max(conv) / S^2|
    We minimize max(conv) / S^2 (assuming it's positive, which it usually is for our purposes).
    Let M = max(conv), S = sum(h).
    f = M / S^2
    df/dh_i = (dM/dh_i * S^2 - M * 2*S * dh_i/dS) / S^4
    dM/dh_i = sum over k where conv[k] = M of dconv[k]/dh_i
    conv[k] = sum_j h[j] h[k-j] (with appropriate bounds)
    dconv[k]/dh_i = h[k-i] + h[i-k] (if in bounds, with care for full convolution indexing)
    
    Actually, let's use a simpler approach: subgradient method.
    The max is non-differentiable, but we can use the subgradient:
    dM/dh_i = sum_{k: conv[k]=M} (h[k-i] + h[i-k]) with bounds checking.
    
    For full convolution: conv[k] = sum_{j=0}^{N-1} h[j] * h[k-j] where h is zero-extended.
    So conv[k] = sum_j h[j] h[k-j]
    dconv[k]/dh_i = h[k-i] + h[i-k] (when both indices are in [0, N-1], with h[j]=0 outside)
    """
    conv = np.convolve(h, h, mode='full')
    M = np.max(conv)
    S = np.sum(h)
    
    # Find indices where conv equals M (within tolerance)
    mask = np.abs(conv - M) < 1e-12
    idxs = np.where(mask)[0]
    
    # Compute dM/dh_i = sum over k in idxs of (h[k-i] + h[i-k]) with zero extension
    dM = np.zeros(N)
    for k in idxs:
        for i in range(N):
            # h[k-i] is h at index k-i, valid if 0 <= k-i < N
            if 0 <= k - i < N:
                dM[i] += h[k - i]
            # h[i-k] is h at index i-k, valid if 0 <= i-k < N
            if 0 <= i - k < N:
                dM[i] += h[i - k]
    
    # f = M / S^2
    # df/dh_i = (dM_i * S^2 - M * 2*S) / S^4 = dM_i / S^2 - 2*M/S^3
    grad = dM / (S * S) - 2 * M / (S * S * S)
    
    # We want to minimize |f|, so if f > 0, minimize f; if f < 0, maximize f (i.e., minimize -f)
    # But M is the max of signed entries, so M >= 0 always (since conv has at least one non-negative entry? Not necessarily.)
    # Actually, for real h, conv can have negative max if all entries are negative. But typically M >= 0.
    # Let's just minimize the signed value f = M/S^2, and take abs at the end.
    # For gradient descent on |f|, we use sign(f) * grad_f
    f = M / (S * S)
    if f < 0:
        grad = -grad
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s
    
    # Initialize with random heights
    np.random.seed(42)
    h = np.random.randn(N)
    h -= np.mean(h)  # Make sum zero? No, sum must not be zero. Let's keep sum nonzero.
    # Actually, let's start with a small perturbation around ones
    h = np.ones(N) + 0.1 * np.random.randn(N)
    
    best_h = h.copy()
    best_val = _value(h)
    
    # DCT basis: use scipy if available, else numpy
    try:
        from scipy.fft import dct, idct
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    # Use a subset of DCT coefficients for parameterization
    # h = idct(c) for the first K coefficients
    K = 50  # Number of DCT coefficients to use
    
    if use_scipy:
        # Compute DCT of initial h
        c = dct(h, type=2, norm='ortho')
        # Zero out high-frequency components
        c_full = c.copy()
        c_full[K:] = 0
        # Optimize c_full to minimize value
    else:
        # Fallback: direct gradient descent on h
        pass
    
    # Projected gradient descent on the DCT coefficients
    # h = idct(c)
    # We want to minimize value(h(c))
    
    lr = 0.01
    momentum = 0.9
    v = np.zeros(N) if not use_scipy else np.zeros_like(c_full)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        if use_scipy:
            h = idct(c_full, type=2, norm='ortho')
            val = _value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            # Compute gradient w.r.t. c
            grad_h = _grad(h)
            # grad_c = dct(grad_h)
            grad_c = dct(grad_h, type=2, norm='ortho')
            
            # Momentum update
            v = momentum * v + (1 - momentum) * grad_c
            c_full = c_full - lr * v
            # Only update the first K coefficients
            c_full[K:] = 0
        else:
            val = _value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            grad_h = _grad(h)
            v = momentum * v + (1 - momentum) * grad_h
            h = h - lr * v
        
        # Adaptive learning rate
        if iteration % 100 == 0:
            lr *= 0.95
    
    return best_h
