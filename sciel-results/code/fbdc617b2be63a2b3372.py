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

    # Improved Sparse Frequency Optimization
    # We represent h as |IFFT(X)|^2.
    # To ensure sum(h) = N/2, we can scale X or penalize.
    # Let's use a penalty method with L-BFGS-B.
    
    def _h_from_params(params, support):
        # params: [A0, R1, P1, R2, P2, ...]
        # A0 is real DC. Rk, Pk are amplitude and phase for frequency k.
        x = np.zeros(N, dtype=np.complex128)
        
        # DC component
        # Sum of h is (1/N) * Sum(|X|^2).
        # We want Sum(h) = N/2 => Sum(|X|^2) = N^2/2.
        # We will optimize X directly and penalize sum deviation.
        
        x[support[0]] = params[0] + 0j
        
        for i, f in enumerate(support[1:]):
            idx_r = 1 + 2*i
            idx_p = 2 + 2*i
            if idx_p >= len(params): break
            amp = params[idx_r]
            phase = params[idx_p]
            x[f] = amp * np.exp(1j * phase)
            x[N-f] = np.conj(x[f])
            
        y = np.fft.ifft(x)
        h_raw = np.abs(y)**2
        
        # Normalize to target sum to help optimization? 
        # No, let's penalize. But scaling helps convergence.
        # Let's scale to sum N/2, then the penalty is 0.
        # But scaling changes the shape.
        # Better: optimize X such that Sum(|X|^2) is close to N^2/2.
        
        sum_h = np.sum(h_raw)
        if sum_h > 1e-12:
            # Scale to exact sum? If we scale, we lose the gradient of the sum constraint.
            # Let's just return h_raw and let the objective penalize the sum and bounds.
            pass
            
        return h_raw

    def _objective(params, support):
        h = _h_from_params(params, support)
        
        # Penalty for sum
        target_sum = N / 2.0
        sum_penalty = (np.sum(h) - target_sum)**2 * 10.0
        
        # Penalty for bounds
        # h must be in [0, 1].
        # h is |y|^2 so h >= 0 always.
        # We only need to penalize h > 1.
        bound_penalty = np.sum(np.maximum(0, h - 1.0)**2) * 10.0
        
        s = _score(h)
        return s + sum_penalty + bound_penalty

    # Try a few specific supports with L-BFGS-B
    # Support: DC and first K harmonics
    for K in [3, 5, 8, 12]:
        if time.time() - t0 > _DEADLINE - 10:
            break
        support = list(range(K))
        # Params: A0, (R1, P1), ..., (RK-1, PK-1)
        # Note: support[0] is 0. support[1:] are 1..K-1.
        num_params = 1 + 2*(K-1)
        
        # Initial guess
        # Target Sum(|X|^2) = N^2/2 = 5000.
        # If we use K components, average |X|^2 ~ 5000/K.
        # A0^2 + 2*Sum(Rk^2) = 5000.
        # Let's set A0 large.
        x0 = np.zeros(num_params)
        # Spread power among components
        power_per_comp = (N**2 / 2.0) / K
        x0[0] = np.sqrt(power_per_comp)
        for i in range(K-1):
            x0[1+2*i] = np.sqrt(power_per_comp / 2.0) # Since Rk^2 appears twice in sum (k and N-k)
            x0[2+2*i] = np.random.rand() * 2 * np.pi
            
        res = minimize(_objective, x0, args=(support,), method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < best_score:
            h_opt = _h_from_params(res.x, support)
            _update_best(h_opt)

    # Also try a few random supports
    for _ in range(3):
        if time.time() - t0 > _DEADLINE - 10:
            break
        K = 5
        idx = np.random.choice(np.arange(1, N//2), K-1, replace=False)
        support = [0] + list(idx)
        num_params = 1 + 2*(K-1)
        
        x0 = np.zeros(num_params)
        power_per_comp = (N**2 / 2.0) / K
        x0[0] = np.sqrt(power_per_comp)
        for i in range(K-1):
            x0[1+2*i] = np.sqrt(power_per_comp / 2.0)
            x0[2+2*i] = np.random.rand() * 2 * np.pi
            
        res = minimize(_objective, x0, args=(support,), method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < best_score:
            h_opt = _h_from_params(res.x, support)
            _update_best(h_opt)

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    return best_h
