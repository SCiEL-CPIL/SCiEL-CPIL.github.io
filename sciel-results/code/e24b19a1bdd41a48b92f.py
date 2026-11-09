import numpy as np
import time

N = 50
DELTA = 0.01  # interval width

def _linear_convolution(h):
    """Compute linear autoconvolution of h (length N), returning array of length 2N-1."""
    # (h*h)_k for k = -(N-1)..(N-1)
    # Use FFT for speed
    n = len(h)
    f = np.fft.rfft(h, n=2*n)
    conv = np.fft.irfft(f * f, n=2*n)[:2*n-1]
    # conv[k] corresponds to (h*h)_{k-(n-1)} for k=0..2n-2
    # Actually, np.convolve(h, h, 'full') gives the linear convolution
    return np.convolve(h, h, mode='full')

def _ratio_discrete(h):
    """Compute R_disc = sum(h^2) / (sum(h) * max(conv(h,h)))."""
    l2sq = np.sum(h**2)
    l1 = np.sum(h)
    conv = _linear_convolution(h)
    peak = np.max(conv)
    if peak <= 0 or l1 <= 0:
        return 0.0
    return l2sq / (l1 * peak)

def _soft_max_conv(h, beta):
    """Soft-max of the autoconvolution values."""
    conv = _linear_convolution(h)
    # Shift to avoid overflow
    m = np.max(conv)
    exps = np.exp(beta * (conv - m))
    logsumexp = m + np.log(np.sum(exps))
    return logsumexp / beta, exps / np.sum(exps)  # soft-max value, weights

def _grad_soft_max_conv(h, beta):
    """Gradient of the soft-max of autoconvolution."""
    conv = _linear_convolution(h)
    m = np.max(conv)
    exps = np.exp(beta * (conv - m))
    weights = exps / np.sum(exps)  # shape (2N-1,)
    # Gradient of (h*h)_k w.r.t. h_j is h_{j+k} + h_{j-k}
    # conv[k] = (h*h)_{k-(N-1)} for k=0..2N-2
    # So (h*h)_l = conv[l + N - 1] for l = -(N-1)..(N-1)
    # d/dh_j (h*h)_l = h_{j+l} + h_{j-l}
    grad = np.zeros_like(h)
    n = len(h)
    for l in range(-(n-1), n):
        w = weights[l + n - 1]
        if w > 1e-15:
            # h_{j+l} for j=0..n-1, with h_i=0 outside
            idx_plus = np.arange(n) + l
            valid_plus = (idx_plus >= 0) & (idx_plus < n)
            h_plus = np.zeros(n)
            h_plus[valid_plus] = h[idx_plus[valid_plus]]
            idx_minus = np.arange(n) - l
            valid_minus = (idx_minus >= 0) & (idx_minus < n)
            h_minus = np.zeros(n)
            h_minus[valid_minus] = h[idx_minus[valid_minus]]
            grad += w * (h_plus + h_minus)
    return grad

def _project(h):
    """Project h onto {h >= 0, sum(h^2) = 1}."""
    h = np.maximum(h, 0)
    norm = np.sqrt(np.sum(h**2))
    if norm > 0:
        h = h / norm
    else:
        h = np.ones(N) / np.sqrt(N)
    return h

def _optimize(beta=10.0, max_iter=500, lr=0.01):
    """Optimize h to minimize (sum h) * soft_max(conv(h,h)) subject to sum(h^2)=1, h>=0."""
    # Initialize
    h = np.ones(N) / np.sqrt(N)
    best_h = h.copy()
    best_val = _ratio_discrete(h)
    
    for it in range(max_iter):
        # Compute objective and gradient
        l1 = np.sum(h)
        sm_val, weights = _soft_max_conv(h, beta)
        obj = l1 * sm_val
        # Gradient of obj = grad(l1) * sm_val + l1 * grad(sm_val)
        grad_l1 = np.ones(N)
        grad_sm = _grad_soft_max_conv(h, beta)
        grad = grad_l1 * sm_val + l1 * grad_sm
        
        # Gradient step
        h_new = h - lr * grad
        # Project
        h_new = _project(h_new)
        
        # Check improvement
        val = _ratio_discrete(h_new)
        if val > best_val:
            best_val = val
            best_h = h_new.copy()
        h = h_new
    
    return best_h, best_val

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin
    
    # Try a few random initializations
    best_h = None
    best_val = 0.0
    
    # Seed
    h0 = np.ones(N) / np.sqrt(N)
    val0 = _ratio_discrete(h0)
    best_h = h0.copy()
    best_val = val0
    
    # Random restarts
    rng = np.random.default_rng(42)
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        # Random nonnegative vector
        h = rng.random(N)
        h = h / np.sqrt(np.sum(h**2))
        # Optimize
        h_opt, val_opt = _optimize(beta=10.0, max_iter=200, lr=0.01)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Return as list of floats
    return list(best_h)
