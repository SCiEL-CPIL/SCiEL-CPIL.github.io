"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b, minimize_scalar

N = 100


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _overlap_vector(h):
    """Compute the overlap vector h[i]*(1-h[i+1])."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    return h * (1.0 - shifted)


def _fix_sum(h, target=None):
    """Fix sum of h to be exactly N/2, keeping elements in [0,1]."""
    if target is None:
        target = N / 2.0
    h = h.copy()
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        s = h.sum()
        diff2 = target - s
        if abs(diff2) > 1e-15:
            mask = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                s = h.sum()
                diff3 = target - s
                if abs(diff3) > 1e-15:
                    mask2 = (h > 1e-12) & (h < 1.0 - 1e-12)
                    if np.any(mask2):
                        h[mask2] += diff3 / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max(x, beta=100.0):
    """Soft max using log-sum-exp."""
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-12:
        return 0.0
    shifted = x - max_x
    return max_x + math.log(np.sum(np.exp(beta * shifted))) / beta


def _smooth_max_grad(x, beta=100.0):
    """Soft max and its gradient with respect to x."""
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-12:
        return 0.0, np.zeros_like(x)
    shifted = x - max_x
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    value = max_x + math.log(wsum) / beta
    grad = weights / wsum
    return value, grad


def _overlap_smooth(h, beta=100.0):
    """Smooth surrogate for max overlap."""
    overlap = _overlap_vector(h)
    return _smooth_max(overlap, beta)


def _overlap_smooth_grad(h, beta=100.0):
    """Smooth surrogate for max overlap and its gradient."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    val, grad_overlap = _smooth_max(overlap, beta)
    grad_h = np.zeros(n)
    for i in range(n):
        grad_h[i] += grad_overlap[i] * (1.0 - (h[i+1] if i+1 < n else 0.0))
        if i + 1 < n:
            grad_h[i+1] += grad_overlap[i] * (-h[i])
    return val, grad_h


def _sigmoid_h(w, c):
    """Build h from sigmoid: h[i] = 1/(1+exp(-(i-c)/w))."""
    i = np.arange(N, dtype=float)
    if w < 1e-6:
        w = 1e-6
    h = 1.0 / (1.0 + np.exp(-(i - c) / w))
    return h


def _solve_c(w):
    """Find c such that sum(sigmoid_h(w,c)) = N/2 via bisection."""
    target_sum = N / 2.0
    
    def sum_diff(c):
        h = _sigmoid_h(w, c)
        return np.sum(h) - target_sum
    
    lo = -20.0 * w
    hi = N + 20.0 * w
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s_mid = np.sum(_sigmoid_h(w, mid))
        if s_mid > target_sum:
            lo = mid
        else:
            hi = mid
            
    c_opt = (lo + hi) / 2.0
    return c_opt


def _sigmoid_score(w):
    """Score for given sigmoid width w."""
    c = _solve_c(w)
    h = _sigmoid_h(w, c)
    h = _fix_sum(h)
    return _score(h)


def _optimize_sigmoid(deadline):
    """Optimize sigmoid width using grid + refinement."""
    best_score = float('inf')
    best_h = None
    
    # Coarse grid on log(w)
    log_w_min = math.log(0.5)
    log_w_max = math.log(50.0)
    n_grid = 150
    
    w_grid = np.linspace(log_w_min, log_w_max, n_grid)
    best_w_log = w_grid[0]
    
    for lw in w_grid:
        if time.time() > deadline - 10:
            break
        w = np.exp(lw)
        try:
            s = _sigmoid_score(w)
            if s < best_score:
                best_score = s
                best_w_log = lw
        except Exception:
            continue
    
    # Refine with minimize_scalar
    if time.time() < deadline - 5:
        delta = (w_grid[1] - w_grid[0]) * 2
        bounds = (max(log_w_min, best_w_log - delta), min(log_w_max, best_w_log + delta))
        
        try:
            def obj(lw):
                w = np.exp(lw)
                return _sigmoid_score(w)
            
            res = minimize_scalar(obj, bounds=bounds, method='bounded', 
                                   options={'xatol': 1e-8, 'maxiter': 100})
            if res.success:
                w_opt = np.exp(res.x)
                c_opt = _solve_c(w_opt)
                h_opt = _sigmoid_h(w_opt, c_opt)
                h_opt = _fix_sum(h_opt)
                s_opt = _score(h_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt
        except Exception:
            pass
    
    if best_h is None:
        w_opt = np.exp(best_w_log)
        c_opt = _solve_c(w_opt)
        best_h = _sigmoid_h(w_opt, c_opt)
        best_h = _fix_sum(best_h)
        best_score = _score(best_h)
    
    return best_h, best_score


def _optimize_2block(deadline):
    """Optimize 2-block construction."""
    n = N
    best_score = float('inf')
    best_h = None
    
    for k in range(1, n):
        if time.time() > deadline - 5:
            break
        a_min = max(0.0, (n/2.0 - n + k) / k)
        a_max = min(1.0, n / (2.0 * k))
        if a_min >= a_max:
            continue
        
        def obj_2block(x, k=k):
            a, b = x
            if a < 0 or a > 1 or b < 0 or b > 1:
                return 1.0
            s = k * a + (n - k) * b
            if abs(s - n/2.0) > 0.1:
                return 1.0 + abs(s - n/2.0)
            h = np.zeros(n)
            h[:k] = a
            h[k:] = b
            h = _fix_sum(h)
            return _overlap_smooth(h, beta=200.0)
        
        for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            a_init = a_min + frac * (a_max - a_min)
            b_init = (n/2.0 - k * a_init) / (n - k)
            if b_init < 0 or b_init > 1:
                continue
            try:
                res = minimize(obj_2block, np.array([a_init, b_init]),
                               method='SLSQP',
                               bounds=[(0, 1), (0, 1)],
                               options={'maxiter': 200, 'ftol': 1e-12})
                if res.fun < 1.0:
                    a_opt, b_opt = res.x
                    h = np.zeros(n)
                    h[:k] = a_opt
                    h[k:] = b_opt
                    h = _fix_sum(h)
                    actual = _score(h)
                    if actual < best_score:
                        best_score = actual
                        best_h = h.copy()
            except Exception:
                pass
    
    return best_h, best_score


def _optimize_full_lbfgsb(h_init, deadline):
    """Optimize full h vector with L-BFGS-B using smooth objective."""
    n = N
    best_score = float('inf')
    best_h = None
    beta = 300.0
    
    def obj_and_grad(x):
        h = np.clip(x, 0.0, 1.0)
        val, grad = _overlap_smooth_grad(h, beta)
        s = h.sum()
        penalty = (s - n/2.0) ** 2 * 50.0
        penalty_grad = 2.0 * (s - n/2.0) * 50.0
        grad_total = grad + penalty_grad
        return val + penalty, grad_total
    
    try:
        res = fmin_l_bfgs_b(obj_and_grad, h_init,
                             fprime=None,
                             pgtol=1e-10, maxiter=300, iprint=0)
        h = _fix_sum(res)
        actual = _score(h)
        if actual < best_score:
            best_score = actual
            best_h = h.copy()
    except Exception:
        pass
    
    return best_h, best_score


def _local_search(h_init, deadline):
    """Coordinate descent local search on the full vector."""
    n = N
    h = h_init.copy()
    best_score = _score(h)
    step_size = 0.05
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for i in range(n):
            for delta in [-step_size, -step_size/2, step_size/2, step_size]:
                h_trial = h.copy()
                h_trial[i] = np.clip(h_trial[i] + delta, 0.0, 1.0)
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < best_score - 1e-12:
                    best_score = s
                    h = h_trial
                    improved = True
                    break
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    return h, best_score


def construct():
    deadline = time.time() + 135.0  # safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Sigmoid optimization (primary structural model)
    if time.time() < deadline - 60:
        hs, ss = _optimize_sigmoid(deadline - 60)
        if hs is not None and ss < best_score:
            best_score = ss
            best_h = hs
    
    # Phase 2: 2-block optimization (alternative structure)
    if time.time() < deadline - 40:
        h2, s2 = _optimize_2block(deadline - 40)
        if h2 is not None and s2 < best_score:
            best_score = s2
            best_h = h2
    
    # Phase 3: Full vector L-BFGS-B from best found
    if time.time() < deadline - 15 and best_h is not None:
        hd, sd = _optimize_full_lbfgsb(best_h, deadline - 15)
        if hd is not None and sd < best_score:
            best_score = sd
            best_h = hd
    
    # Phase 4: Local search polish
    if time.time() < deadline - 3 and best_h is not None:
        hp, sp = _local_search(best_h, deadline - 2)
        if sp < best_score:
            best_score = sp
            best_h = hp
    
    # Fallback
    if best_h is None:
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = _fix_sum(h)
    
    return best_h
