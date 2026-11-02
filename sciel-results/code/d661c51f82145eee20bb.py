"""Second autocorrelation inequality - full-support continuous optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 135.0

def _autoconv_peak(h):
    """Compute max of linear autoconvolution."""
    n = len(h)
    fft_h = np.fft.rfft(h, n=2*n)
    autoconv = np.fft.irfft(fft_h * np.conj(fft_h), n=2*n)[:n]
    return autoconv.max()

def _score(h):
    """Compute the ratio (L2^2)/(L1 * ||autoconv||_inf)."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.dot(h, h)
    peak = _autoconv_peak(h)
    if peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def construct():
    t_start = time.time()
    time_budget = _DEADLINE - t_start - 5.0
    if time_budget < 1.0:
        return [1.0/N] * N

    def objective(x):
        """Negative log of the ratio for minimization."""
        a = np.exp(x)
        a = a / a.sum()  # normalize to sum=1
        l2_sq = np.dot(a, a)
        if l2_sq < 1e-30:
            return 0.0
        peak = _autoconv_peak(a)
        if peak < 1e-30:
            return 0.0
        ratio = l2_sq / (1.0 * peak)  # L1=1 after normalization
        if ratio <= 0:
            return 0.0
        return -np.log(ratio)

    def objective_with_l1(x):
        """Objective without normalization, to allow scaling."""
        a = np.exp(x)
        l1 = a.sum()
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.dot(a, a)
        peak = _autoconv_peak(a)
        if peak < 1e-15:
            return 0.0
        ratio = l2_sq / (l1 * peak)
        if ratio <= 0:
            return 0.0
        return -np.log(ratio)

    best_h = np.ones(N, dtype=float) / N
    best_score = _score(best_h)

    # Generate diverse initializations
    np.random.seed(42)
    inits = []
    
    # Uniform
    inits.append(np.zeros(N))
    
    # Gaussian profiles with different centers and widths
    for c in [N//4, N//2, 3*N//4]:
        for w in [3, 5, 8, 12]:
            h = np.exp(-((np.arange(N) - c)**2) / (2*w*w))
            h = h / h.sum()
            inits.append(np.log(h + 1e-15))
    
    # Triangular
    h = np.maximum(0, 1 - np.abs(np.arange(N) - N//2) / (N//2 + 1))
    h = h / h.sum()
    inits.append(np.log(h + 1e-15))
    
    # Random
    for _ in range(20):
        h = np.random.rand(N) + 0.1
        h = h / h.sum()
        inits.append(np.log(h + 1e-15))
    
    # Sparse-like: concentrate on few points
    for k in [3, 5, 8]:
        support = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[support] = 1.0/k
        inits.append(np.log(h + 1e-15))
    
    # Optimize from each initialization
    n_inits = len(inits)
    time_per_init = time_budget / (n_inits + 1)
    
    for i, x0 in enumerate(inits):
        if time.time() - t_start > _DEADLINE - 10.0:
            break
        
        t0 = time.time()
        max_iter = min(300, max(50, int(time_per_init * 50)))
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
            a = np.exp(res.x)
            a = a / a.sum()
            sc = _score(a)
            if sc > best_score:
                best_score = sc
                best_h = a.copy()
        except Exception:
            pass
        
        if time.time() - t_start > _DEADLINE - 15.0:
            break
    
    # Final polish with more iterations on the best
    if time.time() - t_start < _DEADLINE - 10.0:
        x0 = np.log(best_h + 1e-15)
        t0 = time.time()
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            a = np.exp(res.x)
            a = a / a.sum()
            sc = _score(a)
            if sc > best_score:
                best_score = sc
                best_h = a.copy()
        except Exception:
            pass
    
    best_h = np.maximum(0, best_h)
    if best_h.sum() < 1e-15:
        best_h = np.ones(N) / N
    
    return best_h.tolist()
