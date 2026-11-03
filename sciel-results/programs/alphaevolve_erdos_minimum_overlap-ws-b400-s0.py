"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses FFT-accelerated overlap computation, Nesterov accelerated gradient
descent on a log-sum-exp smooth surrogate, and coordinate-descent polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_T0 = time.time()
_DEADLINE = _T0 + 142  # safety margin before 150s kill


def _overlap_values(h):
    """Compute all overlap values C_k for k in -(N-1)..(N-1).
    C_k = sum_i h[i] * (1 - h[i+k]) for valid i.
    """
    h = np.asarray(h, dtype=np.float64)
    # Use FFT for efficient computation
    # C_k = sum_i h[i]*(1-h[i+k]) = sum_i h[i] - sum_i h[i]*h[i+k]
    # sum_i h[i] over valid range depends on k
    # For k >= 0: sum_{i=0}^{N-k-1} h[i] - sum_{i=0}^{N-k-1} h[i]*h[i+k]
    # For k < 0:  sum_{i=-k}^{N-1} h[i] - sum_{i=-k}^{N-1} h[i]*h[i+k]
    
    # Actually let's just compute directly using convolution
    # autocorr[m] = sum_i h[i] * h[i+m] for m in 0..N-1
    # Use FFT-based autocorrelation
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(H * np.conj(H), n=n_fft)[:N]
    # ac[m] = sum_{i=0}^{N-1-m} h[i]*h[i+m] for m >= 0
    
    # prefix sums of h
    prefix = np.concatenate([[0.0], np.cumsum(h)])
    
    overlaps = np.zeros(2 * N - 1)
    for k in range(-(N-1), N):
        if k >= 0:
            # sum_{i=0}^{N-k-1} h[i] - ac[k]
            s = prefix[N - k] - prefix[0]
            c = s - ac[k]
        else:
            j = -k
            # sum_{i=j}^{N-1} h[i] - ac[j]
            s = prefix[N] - prefix[j]
            c = s - ac[j]
        overlaps[k + N - 1] = c
    return overlaps


def _score(h):
    """Maximum overlap value."""
    return np.max(_overlap_values(h))


def _smooth_max_val_grad(h, beta=200.0):
    """Log-sum-exp smooth approximation of max overlap and its gradient."""
    overlaps = _overlap_values(h)
    max_ov = np.max(overlaps)
    shifted = beta * (overlaps - max_ov)
    exps = np.exp(shifted)
    Z = np.sum(exps)
    val = max_ov + np.log(Z) / beta
    weights = exps / Z
    
    # Gradient: d(val)/d(h_i) = sum_k weights[k] * d(overlap_k)/d(h_i)
    # overlap_k = sum_i h[i]*(1-h[i+k]) for valid i
    # d(overlap_k)/d(h_i) = (1-h[i+k]) if i+k valid, plus -h[i-k] if i-k valid
    # Let's compute this efficiently
    grad = np.zeros(N)
    for idx, k in enumerate(range(-(N-1), N)):
        w = weights[idx]
        if w < 1e-15:
            continue
        if k >= 0:
            # For i in 0..N-k-1: contribution to h[i] is (1-h[i+k])
            # For i in k..N-1: contribution to h[i] is -h[i-k]
            grad[:N - k] += w * (1.0 - h[k:])
            grad[k:] += w * (-h[:N - k])
        else:
            j = -k
            # For i in j..N-1: contribution to h[i] is (1-h[i-j])
            # For i in 0..N-j-1: contribution to h[i] is -h[i+j]
            grad[j:] += w * (1.0 - h[:N - j])
            grad[:N - j] += w * (-h[j:])
    
    return val, grad


def _project(h):
    """Project h onto {0<=h[i]<=1, sum h = N/2}."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        s = np.sum(h)
        diff = target - s
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        else:
            mask = h > 0.0
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h


def _nesterov_pgd(h0, beta, lr, max_iter, deadline):
    """Nesterov accelerated projected gradient descent on smooth max."""
    h = h0.copy()
    y = h0.copy()
    t = 1.0
    best_val = np.inf
    best_h = h0.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        val_y, grad_y = _smooth_max_val_grad(y, beta)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = h + (t / t_new) * (h - y)
        y = _project(y)
        
        h_new = y - lr * grad_y
        h_new = _project(h_new)
        
        val_h = _smooth_max_val_grad(h_new, beta)[0]
        if val_h < best_val:
            best_val = val_h
            best_h = h_new.copy()
        
        h = h_new
        t = t_new
    
    return best_h


def _coordinate_polish(h, deadline, max_iter=5000):
    """Coordinate descent polish: try flipping each coordinate to 0 or 1."""
    h = h.copy()
    best_score = _score(h)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old = h[i]
            # Try setting to 0
            h[i] = 0.0
            h = _project(h)
            s = _score(h)
            if s < best_score - 1e-12:
                best_score = s
                improved = True
            else:
                h[i] = old
            # Try setting to 1
            h[i] = 1.0
            h = _project(h)
            s = _score(h)
            if s < best_score - 1e-12:
                best_score = s
                improved = True
            else:
                h[i] = old
        if not improved:
            break
    
    return h, best_score


def construct():
    start = time.time()
    deadline = start + 140.0
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    rng = np.random.default_rng(123)
    
    # Multiple restarts with different initializations
    max_restarts = 8
    for restart in range(max_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            h0 = np.full(N, 0.5)
        elif restart == 1:
            h0 = rng.uniform(0, 1, N)
            h0 = _project(h0)
        elif restart == 2:
            # Alternating pattern
            h0 = np.where(np.arange(N) % 2 == 0, 0.7, 0.3)
            h0 = _project(h0)
        elif restart == 3:
            # Sine wave
            t = np.linspace(0, 1, N, endpoint=False)
            h0 = 0.5 + 0.3 * np.sin(2 * np.pi * t)
            h0 = _project(h0)
        elif restart == 4:
            # Higher frequency sine
            t = np.linspace(0, 1, N, endpoint=False)
            h0 = 0.5 + 0.25 * np.sin(4 * np.pi * t)
            h0 = _project(h0)
        elif restart == 5:
            # Block pattern
            h0 = np.zeros(N)
            h0[::3] = 1.0
            h0 = _project(h0)
        elif restart == 6:
            # Another structured pattern
            h0 = 0.5 + 0.2 * np.cos(2 * np.pi * np.arange(N) / 7.0)
            h0 = _project(h0)
        else:
            h0 = rng.uniform(0, 1, N)
            h0 = _project(h0)
        
        # Phase 1: Nesterov PGD with moderate beta
        beta1 = 100.0
        lr1 = 0.02
        h1 = _nesterov_pgd(h0, beta1, lr1, 300, deadline - 15.0)
        
        # Phase 2: Nesterov PGD with higher beta for refinement
        if time.time() < deadline - 10.0:
            beta2 = 500.0
            lr2 = 0.005
            h2 = _nesterov_pgd(h1, beta2, lr2, 200, deadline - 5.0)
        else:
            h2 = h1
        
        # Phase 3: Coordinate descent polish
        if time.time() < deadline - 3.0:
            h2, sc2 = _coordinate_polish(h2, deadline - 1.0, max_iter=3000)
        else:
            sc2 = _score(h2)
        
        if sc2 < best_score:
            best_score = sc2
            best_h = h2.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 2.0:
        best_h, best_score = _coordinate_polish(best_h, deadline - 0.5, max_iter=10000)
    
    return best_h
