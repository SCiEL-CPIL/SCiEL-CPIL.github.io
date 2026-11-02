"""Rewritten program for Erdos minimum overlap task."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_min_corr(h):
    """Compute min over shifts of the cross-correlation."""
    n = len(h)
    min_corr = float('inf')
    for k in range(1, n):
        corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
        if corr < min_corr:
            min_corr = corr
    return min_corr

def _compute_all_corrs_fft(h):
    """Compute all cross-correlations using FFT for speed."""
    n = len(h)
    size = 2 * n
    f = np.fft.rfft(h, n=size)
    autocorr = np.fft.irfft(f * np.conj(f), n=size)
    corrs = autocorr[1:n] * (2.0 / n)
    return corrs

def _project(v, target_sum):
    """Project onto {x: 0<=x<=1, sum=target_sum}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_cands = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_cands) == 0:
        rho = n - 1
    else:
        rho = rho_cands[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    w = np.clip(w, 0, 1)
    
    for _ in range(50):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = w < 1.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            mask = w > 0.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    return w

def _soft_min_obj_grad(x, beta=100.0):
    h = x
    n = len(h)
    corrs = _compute_all_corrs_fft(h)
    
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    obj = 1.0 - soft_min
    
    exp_neg_beta = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_neg_beta)
    weights = exp_neg_beta / sum_exp
    
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w > 1e-15:
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    grad_obj = -grad
    return obj, grad_obj

def _subgradient_step(h, k_star, step_size):
    """Perform one subgradient step in the direction that increases the minimum correlation."""
    n = len(h)
    grad = np.zeros(n)
    for j in range(n - k_star):
        grad[j] += (2.0 / n) * h[j + k_star]
        grad[j + k_star] += (2.0 / n) * h[j]
    h_new = h + step_size * grad
    target_sum = n / 2.0
    h_new = _project(h_new, target_sum)
    return h_new

def _coordinate_polish(h, max_iter=2000):
    """
    Coordinate descent polish: for each coordinate, find optimal value
    given others fixed, using the min-correlation objective.
    """
    n = len(h)
    target_sum = n / 2.0
    h = h.copy()
    best_obj = 1.0 - _compute_min_corr(h)
    
    for iteration in range(max_iter):
        if _time_remaining() < 5.0:
            break
        improved = False
        for i in range(n):
            old_val = h[i]
            
            sum_others = np.sum(h) - old_val
            x_min = max(0.0, target_sum - sum_others)
            x_max = min(1.0, target_sum - sum_others)
            
            if x_max - x_min < 1e-15:
                continue
            
            a = np.zeros(n - 1)
            b = np.zeros(n - 1)
            for k in range(1, n):
                ak = 0.0
                if i < n - k:
                    ak += (2.0 / n) * h[i + k]
                if i >= k:
                    ak += (2.0 / n) * h[i - k]
                a[k-1] = ak
                corr_k = np.dot(h[:n-k], h[k:]) * (2.0 / n)
                b[k-1] = corr_k - ak * old_val
            
            candidates = [x_min, x_max]
            for k in range(n - 1):
                for m in range(k + 1, n - 1):
                    da = a[k] - a[m]
                    if abs(da) > 1e-15:
                        x_bp = (b[m] - b[k]) / da
                        if x_min <= x_bp <= x_max:
                            candidates.append(x_bp)
            
            best_x = old_val
            best_min_corr = float('inf')
            for xc in candidates:
                xc = np.clip(xc, x_min, x_max)
                min_c = np.min(a * xc + b)
                if min_c < best_min_corr:
                    best_min_corr = min_c
                    best_x = xc
            
            if best_x != old_val:
                h[i] = best_x
                new_obj = 1.0 - _compute_min_corr(h)
                if new_obj < best_obj - 1e-15:
                    best_obj = new_obj
                    improved = True
        
        if not improved:
            break
    
    return h, best_obj

def construct():
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = 1.0 - _compute_min_corr(best_x)
    
    np.random.seed(42)
    
    # Generate diverse restarts
    restarts = [np.full(n, 0.5)]
    
    # Linear ramps
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    
    # Sinusoidal patterns
    x = np.linspace(0, 2*np.pi, n)
    for freq in [1, 2, 3, 4, 5]:
        for amp in [0.2, 0.3, 0.4]:
            restarts.append(0.5 + amp*np.sin(freq*x))
            restarts.append(0.5 + amp*np.cos(freq*x))
    
    # Random smooth
    for _ in range(15):
        coeffs = np.random.randn(1, 8, 3) * 0.08
        t = np.linspace(0, 2*np.pi, n)
        sig = 0.5
        for j in range(1, 8):
            for k in range(1, 3):
                sig += coeffs[0, j, k] * np.sin(j * k * t)
        restarts.append(np.clip(sig, 0, 1))
    
    # Block patterns
    for k in range(1, n//2):
        restarts.append(np.concatenate([np.zeros(k), np.ones(n-k)*0.5]))
        restarts.append(np.concatenate([np.ones(k)*0.5, np.zeros(n-k)]))
    
    # Alternating
    alt1 = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)])
    alt2 = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)])
    restarts.append(alt1)
    restarts.append(alt2)
    
    # Random
    for _ in range(10):
        restarts.append(np.random.uniform(0, 1, n))
    
    beta_schedule = [20.0, 50.0, 150.0, 400.0, 1000.0]
    
    # Phase 1: L-BFGS-B optimization from multiple restarts
    for x0 in restarts:
        if _time_remaining() < 10.0:
            break
        x0 = _project(x0, target_sum)
        
        for beta in beta_schedule:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_grad, x0, args=(beta,),
                    method='L-BFGS-B', jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 25}
                )
                h_new = _project(res.x, target_sum)
                robj = 1.0 - _compute_min_corr(h_new)
                if robj < best_obj:
                    best_obj = robj
                    best_x = h_new
                x0 = h_new
            except Exception:
                pass
    
    # Phase 2: Subgradient ascent on the active (minimum) shift
    if _time_remaining() > 20.0:
        step_size = 0.01
        for _ in range(500):
            if _time_remaining() < 5.0:
                break
            h = best_x
            min_corr = float('inf')
            k_star = 1
            for k in range(1, n):
                c = np.dot(h[:n-k], h[k:]) * (2.0 / n)
                if c < min_corr:
                    min_corr = c
                    k_star = k
            
            h_new = _subgradient_step(h, k_star, step_size)
            new_obj = 1.0 - _compute_min_corr(h_new)
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = h_new
            else:
                step_size *= 0.8
                if step_size < 1e-6:
                    break
    
    # Phase 3: Coordinate descent polish
    if _time_remaining() > 25.0:
        best_x, best_obj = _coordinate_polish(best_x, max_iter=3000)
    
    # Phase 4: Another round of subgradient + coordinate descent
    if _time_remaining() > 20.0:
        step_size = 0.005
        for _ in range(300):
            if _time_remaining() < 5.0:
                break
            h = best_x
            min_corr = float('inf')
            k_star = 1
            for k in range(1, n):
                c = np.dot(h[:n-k], h[k:]) * (2.0 / n)
                if c < min_corr:
                    min_corr = c
                    k_star = k
            
            h_new = _subgradient_step(h, k_star, step_size)
            new_obj = 1.0 - _compute_min_corr(h_new)
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = h_new
            else:
                step_size *= 0.7
                if step_size < 1e-7:
                    break
        
        if _time_remaining() > 15.0:
            best_x, best_obj = _coordinate_polish(best_x, max_iter=2000)
    
    # Phase 5: High-beta L-BFGS-B polish
    if _time_remaining() > 15.0:
        for beta in [2000.0, 5000.0]:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_grad, best_x, args=(beta,),
                    method='L-BFGS-B', jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-13, 'maxls': 20}
                )
                h_new = _project(res.x, target_sum)
                robj = 1.0 - _compute_min_corr(h_new)
                if robj < best_obj:
                    best_obj = robj
                    best_x = h_new
            except Exception:
                pass
    
    # Phase 6: Final coordinate polish
    if _time_remaining() > 10.0:
        best_x, best_obj = _coordinate_polish(best_x, max_iter=5000)
    
    # Exact sum fix
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-13:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
