"""Rewritten program for Erdos minimum overlap task."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, brentq

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_all_corrs_fft(h):
    """Compute all linear cross-correlations R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1,...,n-1
    using FFT for O(n log n) speed."""
    n = len(h)
    h_padded = np.zeros(2 * n)
    h_padded[:n] = h
    H = np.fft.rfft(h_padded, n=2*n)
    acf = np.fft.irfft(H * np.conj(H), n=2*n)
    # acf[k] = sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=0,...,n-1
    corrs = acf[1:n] * (2.0 / n)
    return corrs

def _compute_min_corr(h):
    """Compute min over shifts of the cross-correlation."""
    corrs = _compute_all_corrs_fft(h)
    return np.min(corrs)

def _project(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
    def f(theta):
        w = np.clip(v - theta, 0, 1)
        return np.sum(w) - target_sum
    
    lo = np.min(v) - 1.0
    hi = np.max(v) + 1.0
    
    if f(lo) < 0:
        lo -= 10.0
    if f(hi) > 0:
        hi += 10.0
        
    try:
        theta = brentq(f, lo, hi, xtol=1e-12)
    except ValueError:
        w = np.clip(v, 0, 1)
        for _ in range(50):
            diff = target_sum - np.sum(w)
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                mask = w < 1.0
                cnt = np.sum(mask)
                if cnt == 0: break
                w[mask] += diff / cnt
            else:
                mask = w > 0.0
                cnt = np.sum(mask)
                if cnt == 0: break
                w[mask] += diff / cnt
            w = np.clip(w, 0, 1)
        return w

    w = np.clip(v - theta, 0, 1)
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-12:
        mask = (w > 1e-9) & (w < 1.0 - 1e-9)
        cnt = np.sum(mask)
        if cnt > 0:
            w[mask] += diff / cnt
            w = np.clip(w, 0, 1)
    return w

def _soft_min_obj_grad_fft(x, beta=100.0):
    """Objective and gradient using soft-min approximation with FFT-based gradient computation.
    
    The gradient of the soft-min of corrs w.r.t. h is:
    grad_h = -sum_k w_k * d(R_k)/dh
    where w_k = exp(-beta*R_k) / sum_j exp(-beta*R_j)
    
    d(R_k)/dh[i] = (2/n) * (h[i+k] if i < n-k else 0) + (2/n) * (h[i-k] if i >= k else 0)
    
    We can compute this efficiently using FFT-based convolution.
    """
    h = x
    n = len(h)
    
    corrs = _compute_all_corrs_fft(h)
    
    # Stable log-sum-exp for soft-min
    max_corr = np.max(corrs)
    exp_vals = np.exp(-beta * (corrs - max_corr))
    sum_exp = np.sum(exp_vals)
    soft_min = max_corr - np.log(sum_exp) / beta
    obj = 1.0 - soft_min
    
    weights = exp_vals / sum_exp  # w_k for k=1,...,n-1
    
    # Compute gradient efficiently
    # grad[i] = -sum_{k=1}^{n-1} w_k * (2/n) * [h[i+k] (if i < n-k) + h[i-k] (if i >= k)]
    # = -(2/n) * sum_{k=1}^{n-1} w_k * h[i+k] (for valid i) - (2/n) * sum_{k=1}^{n-1} w_k * h[i-k] (for valid i)
    
    # First term: for each i, sum over k where i+k < n, i.e., k < n-i
    # Second term: for each i, sum over k where i-k >= 0, i.e., k <= i
    
    # Use cumulative sums for efficiency
    # Term 1: g1[i] = sum_{k=1}^{min(n-1-i, n-1)} w_k * h[i+k]
    # Term 2: g2[i] = sum_{k=1}^{min(i, n-1)} w_k * h[i-k]
    
    # These are cross-correlations of w and h!
    w_padded = np.zeros(2 * n)
    w_padded[:n-1] = weights  # w_1, ..., w_{n-1}
    
    # g1[i] = sum_{k=1}^{n-1-i} w_k * h[i+k] = (w * h)[i] where (w*h)[i] = sum_j w_j h[i+j]
    # This is a cross-correlation. Use FFT.
    
    # Actually, let's just do it directly since n=100 is small
    grad = np.zeros(n)
    for k in range(1, n):
        wk = weights[k-1] * (2.0 / n)
        if wk > 1e-15:
            # d(R_k)/dh[i] = (2/n)*h[i+k] for i=0..n-k-1
            # d(R_k)/dh[i] = (2/n)*h[i-k] for i=k..n-1
            grad[:n-k] -= wk * h[k:]
            grad[k:] -= wk * h[:n-k]
    
    return obj, grad

def _epigraph_obj_grad(x, t, beta=50.0):
    """Epigraph formulation: minimize t s.t. t >= 1 - R_k for all k.
    Smooth approximation: minimize t + (1/beta) * sum_k softplus(beta*(1 - R_k - t))
    """
    h = x
    n = len(h)
    corrs = _compute_all_corrs_fft(h)
    
    # softplus(z) = log(1 + exp(z))
    # gradient: sigmoid(z) = 1/(1+exp(-z))
    
    z = beta * (1.0 - corrs - t)
    # Stable softplus
    softplus_vals = np.where(z > 20, z, np.log1p(np.exp(z)))
    obj = t + np.sum(softplus_vals) / beta
    
    # Gradient w.r.t. h
    sigmoid_vals = np.where(z > 20, 1.0, 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))))
    
    grad_h = np.zeros(n)
    for k in range(1, n):
        wk = sigmoid_vals[k-1] * (2.0 / n)
        if wk > 1e-15:
            grad_h[:n-k] -= wk * h[k:]
            grad_h[k:] -= wk * h[:n-k]
    
    # Gradient w.r.t. t
    grad_t = 1.0 - np.sum(sigmoid_vals) / beta
    
    return obj, (grad_h, grad_t)

def _subgradient_step(h, target_sum, step_size=0.01):
    """Take a subgradient step on the min-correlation constraint."""
    n = len(h)
    corrs = _compute_all_corrs_fft(h)
    min_idx = np.argmin(corrs)
    k_star = min_idx + 1
    
    grad = np.zeros(n)
    for j in range(n - k_star):
        grad[j] += (2.0 / n) * h[j + k_star]
        grad[j + k_star] += (2.0 / n) * h[j]
    
    # We want to increase min corr, so move in direction of grad
    h_trial = h + step_size * grad
    h_trial = _project(h_trial, target_sum)
    return h_trial

def _coordinate_polish(h, target_sum, max_iters=100):
    """Polish using coordinate descent on the min-correlation with adaptive step sizes."""
    n = len(h)
    current_corr = _compute_min_corr(h)
    for _ in range(max_iters):
        if _time_remaining() < 5.0:
            break
        improved = False
        for i in range(n):
            if _time_remaining() < 5.0:
                break
            # Try adaptive step sizes
            for delta in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
                h_trial = h.copy()
                h_trial[i] += delta
                h_trial = _project(h_trial, target_sum)
                trial_corr = _compute_min_corr(h_trial)
                if trial_corr > current_corr + 1e-12:
                    h = h_trial
                    current_corr = trial_corr
                    improved = True
                    break
        if not improved:
            break
    return h

def _local_refine(h, target_sum, max_iters=200):
    """Local refinement using simultaneous perturbation and projection."""
    n = len(h)
    current_corr = _compute_min_corr(h)
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        # Try random perturbations with decreasing step size
        step = 0.02 / (1 + it * 0.1)
        np.random.seed(it)
        
        best_trial = None
        best_trial_corr = current_corr
        
        for _ in range(10):
            perturbation = np.random.normal(0, step, n)
            h_trial = h + perturbation
            h_trial = _project(h_trial, target_sum)
            trial_corr = _compute_min_corr(h_trial)
            
            if trial_corr > best_trial_corr + 1e-12:
                best_trial = h_trial
                best_trial_corr = trial_corr
        
        if best_trial is not None:
            h = best_trial
            current_corr = best_trial_corr
        else:
            break
    
    return h

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_corr = _compute_min_corr(best_x)
    
    # Generate diverse initializations
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n, 0.5))
    # Block structures
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        k = int(n * frac)
        if 0 < k < n:
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
            restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    # Random
    for _ in range(20):
        restarts.append(np.random.uniform(0, 1, n))
    # Smooth gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    for freq in [1, 2, 3, 4, 5, 6, 7, 8]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]:
            restarts.append(0.5 + 0.35*np.sin(freq*x + phase))
            restarts.append(0.5 + 0.35*np.cos(freq*x + phase))
    # Mixed sinusoidal
    np.random.seed(123)
    for _ in range(15):
        f1 = np.random.randint(1, 9)
        f2 = np.random.randint(1, 9)
        p1 = np.random.uniform(0, 2*np.pi)
        p2 = np.random.uniform(0, 2*np.pi)
        a1 = np.random.uniform(0.2, 0.45)
        a2 = np.random.uniform(0.2, 0.45)
        restarts.append(0.5 + a1*np.sin(f1*x + p1) + a2*np.sin(f2*x + p2))
    # Wavelet-like patterns
    for scale in [2, 4, 8]:
        for shift in range(scale):
            pattern = []
            for i in range(n):
                pos = (i + shift) % scale
                if pos < scale // 2:
                    pattern.append(0.0)
                else:
                    pattern.append(1.0)
            restarts.append(np.array(pattern))
    
    # Phase 1: Soft-min optimization with increasing beta per restart
    beta_schedule = [10.0, 30.0, 80.0, 200.0, 500.0]
    
    for x0 in restarts:
        if _time_remaining() < 8.0:
            break
        
        x = _project(x0, target_sum)
        
        for beta in beta_schedule:
            if _time_remaining() < 4.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_grad_fft,
                    x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                res_x = _project(res.x, target_sum)
                res_corr = _compute_min_corr(res_x)
                if res_corr > best_corr:
                    best_corr = res_corr
                    best_x = res_x
                x = res_x
            except Exception:
                pass
    
    # Phase 2: Epigraph formulation polish
    if _time_remaining() > 12.0:
        for beta in [30.0, 80.0, 200.0, 500.0]:
            if _time_remaining() < 4.0:
                break
            try:
                # Start with t = 1 - best_corr
                t0 = 1.0 - best_corr
                x0 = best_x.copy()
                
                def epig_func(vt):
                    v, t = vt[:n], vt[n]
                    obj, (gv, gt) = _epigraph_obj_grad(v, t, beta)
                    return obj, np.concatenate([gv, [gt]])
                
                vt0 = np.concatenate([x0, [t0]])
                bounds = [(0, 1)] * n + [(0, 1)]
                
                res = minimize(
                    epig_func,
                    vt0,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                res_x = _project(res.x[:n], target_sum)
                res_corr = _compute_min_corr(res_x)
                if res_corr > best_corr:
                    best_corr = res_corr
                    best_x = res_x
            except Exception:
                pass
    
    # Phase 3: High-beta soft-min polish on best solution
    if _time_remaining() > 10.0:
        for beta in [800.0, 1500.0, 3000.0]:
            if _time_remaining() < 4.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_grad_fft,
                    best_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                res_x = _project(res.x, target_sum)
                res_corr = _compute_min_corr(res_x)
                if res_corr > best_corr:
                    best_corr = res_corr
                    best_x = res_x
            except Exception:
                pass
    
    # Phase 4: Subgradient and coordinate polish
    if _time_remaining() > 8.0:
        h = best_x.copy()
        for step_size in [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]:
            if _time_remaining() < 4.0:
                break
            for _ in range(40):
                if _time_remaining() < 4.0:
                    break
                h_trial = _subgradient_step(h, target_sum, step_size)
                trial_corr = _compute_min_corr(h_trial)
                if trial_corr > _compute_min_corr(h) + 1e-10:
                    h = h_trial
                else:
                    break
        best_x = h
        best_corr = _compute_min_corr(best_x)
    
    # Phase 5: Local refinement
    if _time_remaining() > 6.0:
        best_x = _local_refine(best_x, target_sum, max_iters=150)
        best_corr = _compute_min_corr(best_x)
    
    # Phase 6: Coordinate polish
    if _time_remaining() > 4.0:
        best_x = _coordinate_polish(best_x, target_sum, max_iters=80)
        best_corr = _compute_min_corr(best_x)
    
    # Ensure exact sum
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
