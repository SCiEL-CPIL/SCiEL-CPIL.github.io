import time
import numpy as np
from scipy.optimize import minimize

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

def _smooth_max_score(h, tau=5.0):
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    if m < 1e-9:
        return 0.0
    return m + tau * np.log(np.sum(np.exp((cs - m) / tau)))

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

def _polish_time_domain(h0):
    h0 = _fix_sum(h0)
    
    def obj(x):
        return _smooth_max_score(x, tau=2.0)
        
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    try:
        res = minimize(obj, h0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
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

    # Fallback: Constant
    _update_best(np.full(N, 0.5))
    # Fallback: Cosine
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)

    # Sparse Frequency Optimization
    def _h_from_x(x_params, support):
        x = np.zeros(N, dtype=np.complex128)
        if not support:
            return np.zeros(N)
        
        x[support[0]] = x_params[0]
        
        for i, f in enumerate(support[1:]):
            idx_re = 1 + 2*i
            idx_im = 2 + 2*i
            if idx_re >= len(x_params): break
            x[f] = x_params[idx_re] + 1j * x_params[idx_im]
            x[N-f] = np.conj(x[f])
            
        y = np.fft.ifft(x)
        h_raw = np.abs(y)**2
        sum_h = np.sum(h_raw)
        if sum_h == 0:
            return np.zeros(N)
        h = h_raw / sum_h * (N / 2.0)
        return h

    def _objective(params, support):
        h = _h_from_x(params, support)
        penalty = 0.0
        if np.any(h > 1.0):
            penalty += np.sum((h[h > 1.0] - 1.0)**2) * 100.0
        if np.any(h < 0.0):
            penalty += np.sum((-h[h < 0.0])**2) * 100.0
        
        s = _smooth_max_score(h, tau=5.0)
        return s + penalty

    # Try different supports
    supports = []
    # Low frequencies
    for K in [10, 20, 30, 40, 50]:
        supports.append(list(range(K)))
    # High frequencies
    for K in [10, 20, 30]:
        supports.append([0] + list(range(N-K+1, N)))
    # Bandpass
    supports.append([0] + list(range(5, 15)))
    supports.append([0] + list(range(35, 45)))
    supports.append([0] + list(range(20, 30)))
    # Random
    np.random.seed(123)
    for _ in range(25):
        K = np.random.randint(10, 30)
        idx = np.random.choice(np.arange(1, N//2), K, replace=False)
        supports.append([0] + list(idx))
    # Structured patterns
    supports.append([0] + list(range(1, 11, 2)))  # Odd low freqs
    supports.append([0] + list(range(2, 22, 2)))  # Even low freqs
    supports.append([0] + list(range(1, 21, 3)))  # Every 3rd low freq
    
    for support in supports:
        if time.time() - t0 > _DEADLINE - 30:
            break
        K = len(support)
        num_params = 1 + 2*(K-1)
        
        # Multiple restarts with different initializations
        for restart in range(3):
            if time.time() - t0 > _DEADLINE - 30:
                break
                
            # Initial guess: random phases, varied amplitudes
            x0 = np.zeros(num_params)
            x0[0] = 1.0  # DC component
            
            if restart == 0:
                # Small random amplitudes
                for i in range(K-1):
                    x0[1+2*i] = np.cos(np.random.rand()*2*np.pi) * 0.1
                    x0[2+2*i] = np.sin(np.random.rand()*2*np.pi) * 0.1
            elif restart == 1:
                # Medium amplitudes with some structure
                for i in range(K-1):
                    amp = 0.05 + 0.05 * np.random.rand()
                    x0[1+2*i] = np.cos(np.random.rand()*2*np.pi) * amp
                    x0[2+2*i] = np.sin(np.random.rand()*2*np.pi) * amp
            else:
                # Larger random amplitudes
                for i in range(K-1):
                    amp = 0.1 + 0.1 * np.random.rand()
                    x0[1+2*i] = np.cos(np.random.rand()*2*np.pi) * amp
                    x0[2+2*i] = np.sin(np.random.rand()*2*np.pi) * amp
            
            res = minimize(_objective, x0, args=(support,), method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_score:
                h_opt = _h_from_x(res.x, support)
                _update_best(h_opt)
                
    # Polish the best result in time domain
    if best_h is not None:
        polished_h = _polish_time_domain(best_h)
        if _score(polished_h) < best_score:
            best_h = polished_h
            best_score = _score(polished_h)
            
        # Second polish with tighter tolerance
        polished_h2 = _polish_time_domain(best_h)
        if _score(polished_h2) < best_score:
            best_h = polished_h2
            best_score = _score(polished_h2)
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        best_h = _polish_time_domain(best_h)
        
    return best_h
