"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses Nesterov Accelerated Gradient Descent (NAG) with Log-Sum-Exp smoothing
and projection onto the sum-constrained box [0, 1]^N.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


def _autocorr(h):
    """Linear autocorrelation r_k = sum_{i=k}^{N-1} h_i * h_{i-k} for k=0..N-1 via FFT."""
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    return corr[:n]


def _overlap_values(h):
    """C_k = (2/N) * (N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """Maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    return float(np.max(cs))


def _smooth_max_grad(h, beta):
    """
    Gradient of smooth max objective: (1/beta) * log(sum(exp(beta * C_k)))
    Returns (value, gradient).
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    # Gradient of C_k with respect to h
    # C_k = (2/N) * (N/2 - r_k)
    # dC_k/dh = -(2/N) * dr_k/dh
    # dr_k/dh_j = h_{j-k} (if j>=k) + h_{j+k} (if j+k < N)
    
    grad = np.zeros(n)
    # Vectorized computation of gradient contribution
    # For each k, we add w[k] * (-(2/N)) * (shifted h terms)
    
    # Efficiently:
    # grad += sum_k w[k] * (-(2/N)) * (h[-k:] + h[:-k] padded)
    # Actually, let's just loop k, it's N=100, so 100 iterations is fast.
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        # Contribution to grad from r_k
        # d r_k / d h_j
        # j >= k: h_{j-k}
        # j < n-k: h_{j+k}
        
        # This creates a vector v where v[j] is the derivative
        # v[k:] += h[:n-k]
        # v[:n-k] += h[k:]
        
        # We can do this in place on a temporary or directly on grad
        if k > 0:
            grad[k:] += factor * h[:n - k]
        if k < n:
            grad[:n - k] += factor * h[k:]
            
    return val, grad


def _project_sum_box(y, S):
    """
    Project y onto {h : sum(h) = S, 0 <= h_i <= 1}.
    Finds tau such that sum(clip(y_i - tau, 0, 1)) = S.
    """
    # S is N/2
    # y - tau
    # We need tau in [min(y)-1, max(y)]
    lo = np.min(y) - 1.0
    hi = np.max(y)
    
    # Bisection on tau
    # f(tau) = sum(clip(y_i - tau, 0, 1)) - S
    # f is decreasing in tau.
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(y - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > S:
            lo = mid
        else:
            hi = mid
    h_proj = np.clip(y - (lo + hi) / 2.0, 0.0, 1.0)
    # Final correction for floating point
    diff = S - np.sum(h_proj)
    if abs(diff) > 1e-12:
        # Adjust the most central element
        idx = np.argsort(np.abs(h_proj - 0.5))[0]
        h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)
    return h_proj


def _nesterov_pgd(h0, beta, max_iters, lr):
    """
    Nesterov Accelerated Gradient Descent.
    """
    h = h0.copy()
    v = np.zeros_like(h)
    S = N / 2.0
    
    for i in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        val, grad = _smooth_max_grad(h, beta)
        
        # Nesterov step
        # y = h + (t-1)/t * (h - h_prev) -- simplified momentum
        # Standard NAG:
        # t_k = (1 + sqrt(1 + 4 t_{k-1}^2)) / 2
        # y = h + (t_{k-1} - 1) / t_k * (h - h_prev)
        # h_next = project(y - lr * grad(y))
        
        # Simpler momentum version often works well:
        # v = momentum * v + grad
        # h = project(h - lr * v)
        
        # Let's use standard NAG structure for convergence guarantee
        t_prev = 1.0
        t_curr = 1.0
        h_prev = h0.copy()
        
        # Actually, let's just use a simple momentum PGD which is robust
        momentum = 0.9
        v = np.zeros_like(h)
        
        for _ in range(max_iters):
            if time.time() > _DEADLINE - 10:
                break
            val, grad = _smooth_max_grad(h, beta)
            v = momentum * v + grad
            h_new = h - lr * v
            h = _project_sum_box(h_new, S)
            
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_sum_box(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Initial guess: Cosine
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)
    
    # 2. Random restarts to find good basins
    for _ in range(5):
        if time.time() - t0 > _DEADLINE - 60:
            break
        h_rand = np.random.uniform(0, 1, N)
        _update_best(h_rand)
        
    # 3. Nesterov PGD optimization with increasing beta
    # Start with low beta (smooth), increase to high beta (sharp max)
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    h_curr = best_h.copy()
    
    for beta in betas:
        if time.time() - t0 > _DEADLINE - 20:
            break
            
        # Decrease learning rate as beta increases (landscape gets sharper)
        lr = 0.01 / np.sqrt(beta)
        
        # Run NAG for a while
        h_opt = _nesterov_pgd(h_curr, beta, max_iters=2000, lr=lr)
        _update_best(h_opt)
        h_curr = best_h.copy() # Use the best found so far as start for next beta
        
    # 4. Final high-precision polish with very high beta and small steps
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        beta_final = 1000.0
        lr_final = 0.001 / np.sqrt(beta_final)
        h_final = _nesterov_pgd(best_h, beta_final, max_iters=5000, lr=lr_final)
        _update_best(h_final)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    return _project_sum_box(best_h, N / 2.0)
