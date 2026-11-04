"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(h):
    """Compute the upper bound on C3: max absolute autocorrelation at non-zero lags."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))


def _compute_acf(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf


def _psl_value(h):
    """Compute PSL: max |acf[k]| for k=1..n-1."""
    acf = _compute_acf(h)
    n = len(h)
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))


def _soft_max_psl(x, beta=10.0):
    """Smooth surrogate: soft-max of |acf[k]| for k=1..n-1."""
    n = len(x)
    if n <= 1:
        return 0.0
    acf = _compute_acf(x)
    sidelobes = np.abs(acf[1:n])
    m = np.max(sidelobes)
    if m == 0:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (sidelobes - m)))) / beta


def _soft_max_psl_grad(x, beta=10.0):
    """
    Gradient of soft-max PSL w.r.t. x.
    Uses FFT-based convolution for efficiency.
    """
    n = len(x)
    if n <= 1:
        return np.zeros(n)
    acf = _compute_acf(x)
    sidelobes = np.abs(acf[1:n])
    m = np.max(sidelobes)
    if m == 0:
        return np.zeros(n)
    
    # Weights for soft-max
    exp_vals = np.exp(beta * (sidelobes - m))
    weights = exp_vals / np.sum(exp_vals)
    
    # weights[k-1] corresponds to lag k (k=1..n-1)
    # grad[i] = sum_{k=1}^{n-1} weights[k-1] * (x[i-k] + x[i+k]) with valid index handling
    # This is: (w * x)[i] + (w_rev * x)[i] where w is padded appropriately
    
    # Build weight vector for convolution
    # w[k] = weights[k-1] for k=1..n-1, w[0]=0
    w = np.zeros(n)
    w[1:] = weights
    
    # Compute convolution w * x and w_rev * x using FFT
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    W = np.fft.rfft(w, n=fft_size)
    X = np.fft.rfft(x, n=fft_size)
    
    # (w * x)[i] = sum_k w[k] * x[i-k] -> this is cross-correlation
    # Actually, let's use direct convolution via FFT
    # conv(w, x)[i] = sum_k w[k] * x[i-k]
    conv_wx = np.fft.irfft(W * X, n=fft_size)
    
    # For the other term: sum_k w[k] * x[i+k] = sum_k w[k] * x[i+k]
    # Let j = i+k, then k = j-i, sum_j w[j-i] * x[j] = (w_rev * x)[i]
    # where w_rev[k] = w[-k] = w[n-k] (circular)
    # Actually, sum_k w[k] * x[i+k] is the cross-correlation of w and x at lag -i
    # = conv(w_rev, x)[i] where w_rev is w reversed
    
    w_rev = w[::-1]
    W_rev = np.fft.rfft(w_rev, n=fft_size)
    conv_wrev_x = np.fft.irfft(W_rev * X, n=fft_size)
    
    grad = conv_wx[0:n] + conv_wrev_x[0:n]
    
    return grad


def _continuous_optimize(x_init, max_iters=300, lr=0.05, beta=10.0, time_budget=15.0):
    """
    Gradient descent on continuous relaxation with periodic projection to ±1.
    """
    start = time.time()
    x = x_init.copy()
    best_h = np.sign(x).copy()
    best_psl = _psl_value(best_h)
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        # Compute gradient
        grad = _soft_max_psl_grad(x, beta=beta)
        
        # Gradient step
        x = x - lr * grad
        
        # Normalize to prevent drift
        x_norm = np.linalg.norm(x)
        if x_norm > 0:
            x = x / x_norm * np.sqrt(N)
        
        # Project to ±1 and evaluate
        h = np.sign(x)
        psl = _psl_value(h)
        if psl < best_psl:
            best_psl = psl
            best_h = h.copy()
        
        # Adapt learning rate
        if it % 50 == 0 and it > 0:
            lr *= 0.9
    
    return best_h, best_psl


def _local_search(h, time_limit=20.0):
    """
    Local search: try flipping each position to minimize PSL.
    """
    h = h.copy()
    n = len(h)
    current_psl = _psl_value(h)
    best_h = h.copy()
    best_psl = current_psl
    
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        # Try flipping each position
        for i in range(n):
            if time.time() - start > time_limit:
                break
            h[i] = -h[i]
            new_psl = _psl_value(h)
            if new_psl < current_psl:
                current_psl = new_psl
                improved = True
                if new_psl < best_psl:
                    best_psl = new_psl
                    best_h = h.copy()
            else:
                h[i] = -h[i]  # revert
        
        if not improved:
            # Escape local minimum: random perturbation
            for _ in range(5):
                i = np.random.randint(n)
                h[i] = -h[i]
            current_psl = _psl_value(h)
            if current_psl > best_psl * 2.0:
                h = best_h.copy()
                current_psl = best_psl
    
    return best_h, best_psl


def _simulated_annealing(h_init, time_budget=15.0):
    """
    Simulated annealing on ±1 sequences.
    """
    n = len(h_init)
    h = h_init.copy()
    current_psl = _psl_value(h)
    best_h = h.copy()
    best_psl = current_psl
    
    T = 1.0
    cooling = 0.9995
    
    start = time.time()
    while time.time() - start < time_budget:
        i = np.random.randint(n)
        h[i] = -h[i]
        new_psl = _psl_value(h)
        delta = new_psl - current_psl
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_psl = new_psl
            if new_psl < best_psl:
                best_psl = new_psl
                best_h = h.copy()
        else:
            h[i] = -h[i]
        
        T *= cooling
    
    return best_h, best_psl


def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin: 150s limit
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Continuous optimization from multiple random starts
    num_starts = 10
    for seed in range(num_starts):
        if time.time() > deadline - 50:
            break
        rng = np.random.default_rng(seed)
        x_init = rng.normal(size=N)
        h_opt, psl_opt = _continuous_optimize(x_init, max_iters=200, lr=0.1, beta=10.0, time_budget=8.0)
        if psl_opt < best_score:
            best_score = psl_opt
            best_h = h_opt.copy()
    
    # Phase 2: Local search from best solution
    if best_h is not None:
        h_ls, psl_ls = _local_search(best_h, time_limit=25.0)
        if psl_ls < best_score:
            best_score = psl_ls
            best_h = h_ls
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 20:
        h_sa, psl_sa = _simulated_annealing(best_h, time_budget=15.0)
        if psl_sa < best_score:
            best_score = psl_sa
            best_h = h_sa
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 10:
        h_final, psl_final = _local_search(best_h, time_limit=10.0)
        if psl_final < best_score:
            best_score = psl_final
            best_h = h_final
    
    if best_h is None:
        best_h = np.random.choice([-1.0, 1.0], size=N)
    
    return best_h
# EVOLVE-BLOCK-END
