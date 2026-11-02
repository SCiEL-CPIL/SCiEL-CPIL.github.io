"""Combined spectral + projected-gradient approach for Erdos minimum overlap."""

import time
import numpy as np
from scipy.fft import fft, ifft

N = 100

# ---- tunable knobs ----
EPS = 1e-4
ALPHA = 0.02
MAX_POLISH_ITERS = 300
DEADLINE_S = 140.0  # safety margin before 150s kill


def _time_remaining():
    return DEADLINE_S - time.time()


def _max_overlap(h):
    """Maximum circular overlap of h with shifted (1-h), via FFT."""
    H = fft(h)
    F = H * (1.0 - np.exp(-2j * np.pi * np.arange(N) / N))
    return float(np.max(np.real(ifft(F))))


def _compute_all_overlaps(h):
    """All circular overlaps of h with shifted (1-h), via FFT."""
    H = fft(h)
    F = H * (1.0 - np.exp(-2j * np.pi * np.arange(N) / N))
    return np.real(ifft(F))


def _project(h, target_sum=None):
    """Project onto {0<=h_i<=1, sum_i h_i = target_sum}."""
    if target_sum is None:
        target_sum = 0.5 * N
    h = np.clip(h, 0.0, 1.0)
    diff = h.sum() - target_sum
    if diff > 1e-12:
        order = np.argsort(h)[::-1]
        for i in order:
            take = min(diff, h[i])
            h[i] -= take
            diff -= take
            if diff <= 1e-12:
                break
    elif diff < -1e-12:
        diff = -diff
        order = np.argsort(h)
        for i in order:
            take = min(diff, 1.0 - h[i])
            h[i] += take
            diff -= take
            if diff <= 1e-12:
                break
    return h


def _spectral_solution():
    """Deterministic circulant construction with low-frequency spectral plan."""
    A = 0.25 * N - (N - 3) * EPS
    A = max(A, 0.0)
    mags = np.full(N, EPS)
    mags[0] = 0.5 * N
    mags[1] = A
    mags[N - 1] = A
    H = mags * np.exp(1j * 2 * np.pi * np.arange(N) / N)
    h = np.real(ifft(H))
    return _project(h)


def _spectral_solution_variational():
    """Try concentrating energy in different low-frequency bins."""
    best_h = None
    best_v = float('inf')
    
    for target_k in [1, 2, 3, 4, 5]:
        A = 0.25 * N - (N - (2*target_k + 1)) * EPS
        A = max(A, 0.0)
        mags = np.full(N, EPS)
        mags[0] = 0.5 * N
        mags[target_k] = A
        mags[N - target_k] = A
        for k in range(1, N):
            if k != target_k and k != N - target_k and k != 0:
                mags[k] = EPS
        H = mags * np.exp(1j * 2 * np.pi * np.arange(N) / N)
        h = np.real(ifft(H))
        h = _project(h)
        v = _max_overlap(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
    
    return best_h, best_v


def _polish(h0):
    """Bounded projected gradient descent on the max-overlap objective."""
    h = h0.copy()
    best = h.copy()
    best_v = _max_overlap(h)
    it = 0
    step = ALPHA
    
    while it < MAX_POLISH_ITERS and _time_remaining() > 5.0:
        H = fft(h)
        F = H * (1.0 - np.exp(-2j * np.pi * np.arange(N) / N))
        Ov = np.real(ifft(F))
        s = int(np.argmax(Ov))
        
        # d O(s)/d h_i = (1 - h_{i-s}) - h_{i+s} (circular)
        idx = np.arange(N)
        grad = (1.0 - h[(idx - s) % N]) - h[(idx + s) % N]
        
        h_new = _project(h - step * grad)
        v_new = _max_overlap(h_new)
        
        if v_new < best_v:
            best_v = v_new
            best = h_new.copy()
            h = h_new
            step = min(step * 1.1, 0.1)
        else:
            step = step * 0.7
            if step < 0.001:
                break
        
        it += 1
    
    return best, best_v


def _soft_min_optimize(h0, beta_schedule=None):
    """Optimize using soft-min formulation with L-BFGS-B."""
    if beta_schedule is None:
        beta_schedule = [50.0, 100.0, 300.0, 1000.0]
    
    from scipy.optimize import minimize
    
    def _obj_grad(x, beta):
        h = x
        n = len(h)
        corrs = _compute_all_overlaps(h)
        
        # soft min of corrs
        min_val = np.min(corrs)
        shifted = corrs - min_val
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        soft_min = min_val - log_sum / beta
        obj = 1.0 - soft_min
        
        exp_neg_beta = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_neg_beta)
        weights = exp_neg_beta / sum_exp
        
        # gradient of soft_min wrt h
        # d(corrs[k])/d(h[i]) for circular correlation
        grad = np.zeros(n)
        for k in range(1, n):
            w = weights[k-1]
            if w > 1e-15:
                # corrs[k] = sum_i h[i] * (1 - h[(i+k)%N])
                # d/dh[i] = (1 - h[(i+k)%N]) - h[(i-k)%N]
                idx = np.arange(n)
                dcorr = (1.0 - h[(idx + k) % n]) - h[(idx - k) % n]
                grad += w * dcorr
        
        grad_obj = -grad
        return obj, grad_obj
    
    best_h = h0.copy()
    best_v = _max_overlap(h0)
    
    current = h0.copy()
    
    for beta in beta_schedule:
        if _time_remaining() < 3.0:
            break
        
        try:
            res = minimize(
                _obj_grad,
                current,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * N,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < _max_overlap(current):
                h_new = _project(res.x)
                v_new = _max_overlap(h_new)
                if v_new < best_v:
                    best_v = v_new
                    best_h = h_new.copy()
                    current = h_new
        except Exception:
            pass
    
    return best_h, best_v


def construct():
    t0 = time.time()
    
    # Start with spectral solution
    h = _spectral_solution()
    v = _max_overlap(h)
    
    best_h = h.copy()
    best_v = v
    
    # Try variational spectral solutions
    if _time_remaining() > 10.0:
        h2, v2 = _spectral_solution_variational()
        if v2 < best_v:
            best_v = v2
            best_h = h2.copy()
    
    # Polish with projected gradient descent
    if _time_remaining() > 15.0:
        h3, v3 = _polish(best_h)
        if v3 < best_v:
            best_v = v3
            best_h = h3.copy()
    
    # Try soft-min optimization from multiple starting points
    if _time_remaining() > 30.0:
        starts = [best_h.copy()]
        
        # Add a few diverse starts
        starts.append(_spectral_solution())
        x = np.linspace(0, 2*np.pi, N)
        for freq, amp in [(1, 0.3), (2, 0.3), (3, 0.2)]:
            starts.append(_project(0.5 + amp*np.sin(freq*x)))
            starts.append(_project(0.5 + amp*np.cos(freq*x)))
        
        for start in starts:
            if _time_remaining() < 10.0:
                break
            h4, v4 = _soft_min_optimize(start)
            if v4 < best_v:
                best_v = v4
                best_h = h4.copy()
    
    # Final polish
    if _time_remaining() > 10.0:
        h5, v5 = _polish(best_h)
        if v5 < best_v:
            best_v = v5
            best_h = h5.copy()
    
    # Ensure valid output
    best_h = _project(best_h)
    
    return best_h
