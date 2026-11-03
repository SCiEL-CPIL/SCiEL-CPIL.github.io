import time
import numpy as np
from scipy.optimize import minimize, root

N = 100
_DEADLINE = time.time() + 110

def _linear_overlap_values(h):
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
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _fix_sum(h):
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

def _h_from_sparse(x_params, support):
    """Reconstruct h from sparse frequency representation."""
    n = N
    X = np.zeros(n, dtype=np.complex128)
    
    # DC component (real, non-negative)
    if len(support) > 0 and support[0] == 0:
        X[0] = max(0.0, x_params[0])
    
    # Other frequencies (complex conjugate pairs)
    for i, f in enumerate(support[1:]):
        idx = 1 + 2*i
        if idx + 1 >= len(x_params):
            break
        re = x_params[idx]
        im = x_params[idx+1]
        X[f] = re + 1j * im
        X[n-f] = re - 1j * im
    
    h_raw = np.abs(np.fft.ifft(X))**2
    sum_h = np.sum(h_raw)
    if sum_h < 1e-15:
        return np.full(n, 0.5)
    h = h_raw / sum_h * (n / 2.0)
    return np.clip(h, 0.0, 1.0)

def _overlap_gradients(h, x_params, support):
    """Compute gradients of overlap values with respect to sparse parameters."""
    n = N
    cs = _linear_overlap_values(h)
    # Numerical gradient approximation
    eps = 1e-7
    grad = np.zeros_like(x_params)
    for j in range(len(x_params)):
        x_plus = x_params.copy()
        x_plus[j] += eps
        h_plus = _h_from_sparse(x_plus, support)
        cs_plus = _linear_overlap_values(h_plus)
        grad[j] = (np.max(cs_plus) - np.max(cs)) / eps
    return grad

def _smooth_max(cs, tau=5.0):
    """Smooth maximum function."""
    m = np.max(cs)
    if m < 1e-9:
        return 0.0
    return m + tau * np.log(np.sum(np.exp((cs - m) / tau)))

def _objective(params, support):
    """Objective function for optimization."""
    h = _h_from_sparse(params, support)
    cs = _linear_overlap_values(h)
    
    # Penalty for out-of-range
    penalty = 0.0
    if np.any(h > 1.0):
        penalty += np.sum((h[h > 1.0] - 1.0)**2) * 1000.0
    if np.any(h < 0.0):
        penalty += np.sum((-h[h < 0.0])**2) * 1000.0
    
    s = _smooth_max(cs, tau=3.0)
    return s + penalty

def _polish_time_domain(h0):
    """Polish using SLSQP in time domain."""
    h0 = _fix_sum(h0)
    
    def obj(x):
        return _smooth_max(_linear_overlap_values(x), tau=2.0)
        
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    try:
        res = minimize(obj, h0, method='SLSQP', bounds=bounds, constraints=constraints, 
                      options={'maxiter': 300, 'ftol': 1e-12})
        if res.success and res.fun < _score(h0):
            return _fix_sum(res.x)
    except Exception:
        pass
    return h0

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        if np.any(h > 1.0 + 1e-9) or np.any(h < -1e-9):
            return
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Fallbacks
    _update_best(np.full(N, 0.5))
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)
    
    # Try different sparse supports
    supports_to_try = []
    
    # Low frequency supports
    for K in [5, 8, 12, 15, 20]:
        supports_to_try.append([0] + list(range(1, K)))
    
    # Mixed low and high
    for K_low, K_high in [(5, 3), (8, 5), (10, 5), (12, 8)]:
        supports_to_try.append([0] + list(range(1, K_low)) + list(range(N-K_high, N)))
    
    # Bandpass
    for center, width in [(10, 5), (15, 8), (20, 10)]:
        supports_to_try.append([0] + list(range(max(1, center-width), center+width)))
    
    # Random supports
    np.random.seed(123)
    for _ in range(20):
        K = np.random.randint(8, 18)
        idx = np.random.choice(np.arange(1, N//2), K, replace=False)
        supports_to_try.append([0] + list(idx))
    
    for support in supports_to_try:
        if time.time() - t0 > _DEADLINE - 25:
            break
        
        K = len(support)
        num_params = 1 + 2*(K-1)
        
        # Multiple random initializations
        for trial in range(5):
            if time.time() - t0 > _DEADLINE - 20:
                break
            
            # Random phases and small amplitudes
            x0 = np.zeros(num_params)
            x0[0] = np.random.uniform(0.5, 1.5)  # DC component
            for i in range(K-1):
                phase = np.random.rand() * 2 * np.pi
                amp = np.random.uniform(0.05, 0.3)
                x0[1+2*i] = amp * np.cos(phase)
                x0[2+2*i] = amp * np.sin(phase)
            
            # Optimize using L-BFGS-B
            try:
                res = minimize(_objective, x0, args=(support,), method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < best_score:
                    h_opt = _h_from_sparse(res.x, support)
                    _update_best(h_opt)
            except Exception:
                pass
    
    # Polish the best result
    if best_h is not None:
        polished_h = _polish_time_domain(best_h)
        if _score(polished_h) < best_score:
            best_h = polished_h
            best_score = _score(polished_h)
        
        # Additional polishing iterations
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 5:
                break
            polished_h = _polish_time_domain(best_h)
            if _score(polished_h) < best_score:
                best_h = polished_h
                best_score = _score(polished_h)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
        best_h = _polish_time_domain(best_h)
        
    return best_h
