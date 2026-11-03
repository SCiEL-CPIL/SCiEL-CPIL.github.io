"""Erdos minimum overlap – piecewise-constant (impulse-train) parameterization."""

import numpy as np
import time

N = 100

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _all_corrs_fft(h):
    """Compute all non-circular cross-correlations R(k) for k=1..n-1 using FFT."""
    n = len(h)
    size = 1
    while size < 2 * n:
        size *= 2
    H = np.fft.rfft(h, size)
    corr_full = np.fft.irfft(H * np.conj(H), size)
    corrs = (2.0 / n) * corr_full[1:n]
    return corrs


def _max_overlap(h):
    """Maximum overlap = 1 - min_k R(k)."""
    corrs = _all_corrs_fft(h)
    return 1.0 - np.min(corrs)


def _build_from_impulses(breaks, vals, n):
    """
    Build a length-n array from a piecewise-constant description.
    breaks: sorted array of interior breakpoints (positions where value changes), length m-1
    vals: values for each segment, length m
    Returns: length-n array.
    """
    m = len(vals)
    if m == 1:
        return np.full(n, vals[0])
    # breaks has length m-1
    # Segment i covers indices [ceil(breaks[i-1]), ceil(breaks[i])-1] approximately
    # More precisely: segment i is [breaks[i-1], breaks[i]) in continuous [0,n)
    # For discrete: segment i covers indices where floor(idx) >= breaks[i-1] and < breaks[i]
    # Actually let's think of it as: the function is constant on intervals
    # [0, breaks[0]), [breaks[0], breaks[1]), ..., [breaks[m-2], n)
    # For index j (0-indexed), it belongs to segment i if breaks[i-1] <= j < breaks[i]
    # with breaks[-1] = 0 and breaks[m-1] = n
    h = np.empty(n)
    # Use searchsorted
    # For each index j, find which segment it's in
    # Segment boundaries: 0, breaks[0], breaks[1], ..., breaks[m-2], n
    boundaries = np.concatenate([[0.0], breaks, [float(n)]])
    # For index j, segment = number of boundaries <= j
    segs = np.searchsorted(boundaries, np.arange(n), side='right') - 1
    segs = np.clip(segs, 0, m - 1)
    h = vals[segs]
    return h


def _project(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    w = np.clip(w, 0, 1)
    for _ in range(80):
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


def _soft_min_obj_grad_fft(h, tau):
    """Soft-min of correlations and its gradient using FFT."""
    n = len(h)
    corrs = _all_corrs_fft(h)
    min_corr = np.min(corrs)
    exp_vals = np.exp(-(corrs - min_corr) / tau)
    Z = np.sum(exp_vals)
    soft_min_val = min_corr - tau * np.log(Z)
    
    weights = exp_vals / Z
    grad = np.zeros(n)
    for k in range(1, n):
        w_k = weights[k - 1]
        if w_k < 1e-15:
            continue
        grad[:n-k] += (2.0/n) * w_k * h[k:]
        grad[k:] += (2.0/n) * w_k * h[:n-k]
    return soft_min_val, grad


def _optimize_impulse_train(breaks, vals, n, target_sum, n_iter=5000, lr_init=0.02, tau_init=0.05):
    """
    Optimize the impulse-train parameterization (breaks, vals) using gradient ascent on soft-min.
    """
    m = len(vals)
    # Work with log-spaced or direct breaks
    # Use direct optimization of breaks and vals
    breaks = breaks.copy()
    vals = vals.copy()
    
    best_h = _build_from_impulses(breaks, vals, n)
    best_obj = _max_overlap(best_h)
    patience = 0
    lr = lr_init
    tau = tau_init
    
    for it in range(n_iter):
        if _time_remaining() < 3.0:
            break
        
        h = _build_from_impulses(breaks, vals, n)
        h = _project(h, target_sum)
        
        # Compute gradient w.r.t. h
        _, grad_h = _soft_min_obj_grad_fft(h, tau)
        
        # Map gradient back to breaks and vals
        # d h[j] / d vals[i] = 1 if j is in segment i
        # d h[j] / d breaks[k] = -delta(j - breaks[k]) approximately (discrete)
        
        # Gradient w.r.t. vals: for each segment i, sum of grad_h[j] over j in segment i
        boundaries = np.concatenate([[0.0], breaks, [float(n)]])
        segs = np.searchsorted(boundaries, np.arange(n), side='right') - 1
        segs = np.clip(segs, 0, m - 1)
        
        grad_vals = np.zeros(m)
        for i in range(m):
            mask = segs == i
            grad_vals[i] = np.sum(grad_h[mask])
        
        # Gradient w.r.t. breaks: derivative of h[j] w.r.t. breaks[k]
        # h[j] changes when breaks[k] crosses j
        # For a small perturbation of breaks[k] by db, h[j] changes by:
        #   if j was in segment k and moves to segment k+1: h[j] changes by vals[k+1]-vals[k]
        #   if j was in segment k+1 and moves to segment k: h[j] changes by vals[k]-vals[k+1]
        # In the discrete case, this is a step function. Use a smoothed version.
        # Approximate: grad_breaks[k] = sum over j near breaks[k] of grad_h[j] * (vals[j's old seg] - vals[j's new seg])
        # For simplicity, use a finite-difference-like approach with small epsilon
        grad_breaks = np.zeros(m - 1)
        eps = 0.5
        for k in range(m - 1):
            # Perturb breaks[k] by +eps
            b_plus = breaks.copy()
            b_plus[k] += eps
            b_plus = np.sort(b_plus)
            h_plus = _build_from_impulses(b_plus, vals, n)
            h_plus = _project(h_plus, target_sum)
            _, grad_plus = _soft_min_obj_grad_fft(h_plus, tau)
            obj_plus = -np.log(np.sum(np.exp(-_all_corrs_fft(h_plus) / tau)))  # proxy
            
            # Perturb breaks[k] by -eps
            b_minus = breaks.copy()
            b_minus[k] -= eps
            b_minus = np.sort(b_minus)
            h_minus = _build_from_impulses(b_minus, vals, n)
            h_minus = _project(h_minus, target_sum)
            
            # Use central difference on the soft-min objective
            corrs_plus = _all_corrs_fft(h_plus)
            corrs_minus = _all_corrs_fft(h_minus)
            min_p = np.min(corrs_plus)
            min_m = np.min(corrs_minus)
            # Simple: use the change in min correlation
            grad_breaks[k] = (min_p - min_m) / (2.0 * eps)
        
        # Update
        vals_new = vals + lr * grad_vals
        breaks_new = breaks + lr * grad_breaks
        
        # Keep breaks sorted and within (0, n)
        breaks_new = np.clip(breaks_new, 1.0, n - 1.0)
        breaks_new = np.sort(breaks_new)
        vals_new = np.clip(vals_new, 0.0, 1.0)
        
        h_new = _build_from_impulses(breaks_new, vals_new, n)
        h_new = _project(h_new, target_sum)
        obj_new = _max_overlap(h_new)
        
        if obj_new < best_obj - 1e-12:
            best_obj = obj_new
            best_h = h_new.copy()
            breaks = breaks_new.copy()
            vals = vals_new.copy()
            patience = 0
        else:
            patience += 1
            if patience > 20:
                lr *= 0.9
                patience = 0
        
        # Decay tau
        tau = max(0.005, tau_init * (1.0 - it / n_iter) + 0.005)
    
    return best_h, best_obj


def _direct_optimize(h0, n, target_sum, n_iter=10000, lr_init=0.005, tau_init=0.05):
    """Direct soft-min subgradient ascent on the full h vector."""
    h = h0.copy()
    lr = lr_init
    tau = tau_init
    best_obj = _max_overlap(h)
    best_h = h.copy()
    patience = 0
    
    for it in range(n_iter):
        if _time_remaining() < 3.0:
            break
        
        tau = max(0.005, tau_init * (1.0 - it / n_iter) + 0.005)
        val, grad = _soft_min_obj_grad_fft(h, tau)
        
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-10:
            step = grad / gnorm
        else:
            break
        
        h_new = h + lr * step
        h_new = _project(h_new, target_sum)
        obj_new = _max_overlap(h_new)
        
        if obj_new < best_obj - 1e-12:
            best_obj = obj_new
            best_h = h_new.copy()
            h = h_new
            patience = 0
        else:
            patience += 1
            if patience > 30:
                lr *= 0.95
                patience = 0
    
    return best_h, best_obj


def _coordinate_polish(h, n_sweeps, target_sum):
    """Coordinate descent polish."""
    n = len(h)
    h = h.copy()
    best_obj = _max_overlap(h)
    
    for sweep in range(n_sweeps):
        if _time_remaining() < 3.0:
            break
        improved = False
        for j in range(n):
            if _time_remaining() < 3.0:
                break
            old_val = h[j]
            candidates = [0.0, 1.0]
            if 0 < old_val < 1:
                candidates.extend([old_val * 0.5, old_val * 1.5, 0.25, 0.75, 0.1, 0.9])
            
            best_cand_obj = best_obj
            best_cand_val = old_val
            for cv in candidates:
                h[j] = cv
                h_proj = _project(h, target_sum)
                obj_proj = _max_overlap(h_proj)
                if obj_proj < best_cand_obj - 1e-13:
                    best_cand_obj = obj_proj
                    best_cand_val = cv
                    h = h_proj.copy()
                h[j] = old_val
            
            if best_cand_val != old_val:
                h[j] = best_cand_val
                h = _project(h, target_sum)
                best_obj = _max_overlap(h)
                improved = True
        
        if not improved and sweep > 1:
            break
    
    return h, best_obj


def construct():
    n = N
    target_sum = n / 2.0
    np.random.seed(42)
    
    best_x = np.full(n, 0.5)
    best_obj = _max_overlap(best_x)
    
    # Phase 1: Impulse-train optimization with various numbers of segments
    for m in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        if _time_remaining() < 10.0:
            break
        
        # Multiple random initializations for each m
        for trial in range(3):
            if _time_remaining() < 5.0:
                break
            
            # Generate random breakpoints
            breaks = np.sort(np.random.uniform(1, n - 1, m - 1))
            # Generate random values
            vals = np.random.uniform(0.1, 0.9, m)
            
            h_res, obj_res = _optimize_impulse_train(breaks, vals, n, target_sum, n_iter=3000, lr_init=0.02, tau_init=0.05)
            if obj_res < best_obj - 1e-10:
                best_obj = obj_res
                best_x = h_res.copy()
    
    # Phase 2: Direct optimization from diverse starting points
    restarts = []
    restarts.append(np.full(n, 0.5))
    
    # Block structures
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        k = int(n * frac)
        if 0 < k < n:
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 1.0)]))
            restarts.append(np.concatenate([np.full(k, 1.0), np.full(n-k, 0.0)]))
    
    # Alternating
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    for freq in [1, 2, 3, 4, 5]:
        restarts.append(np.clip(0.5 + 0.4*np.sin(freq*x), 0, 1))
        restarts.append(np.clip(0.5 + 0.4*np.cos(freq*x), 0, 1))
    
    # Random
    for _ in range(15):
        restarts.append(np.random.uniform(0, 1, n))
    
    for x0 in restarts:
        if _time_remaining() < 8.0:
            break
        x = _project(x0, target_sum)
        h_res, obj_res = _direct_optimize(x, n, target_sum, n_iter=5000, lr_init=0.005, tau_init=0.05)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Phase 3: Coordinate descent polish
    if _time_remaining() > 10.0:
        h_res, obj_res = _coordinate_polish(best_x, n_sweeps=30, target_sum=target_sum)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Phase 4: Final direct optimization with small steps
    if _time_remaining() > 8.0:
        h_res, obj_res = _direct_optimize(best_x, n, target_sum, n_iter=8000, lr_init=0.002, tau_init=0.02)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Phase 5: Final coordinate polish
    if _time_remaining() > 5.0:
        h_res, obj_res = _coordinate_polish(best_x, n_sweeps=15, target_sum=target_sum)
        if obj_res < best_obj - 1e-10:
            best_obj = obj_res
            best_x = h_res.copy()
    
    # Ensure exact sum
    best_x = _project(best_x, target_sum)
    
    return best_x
