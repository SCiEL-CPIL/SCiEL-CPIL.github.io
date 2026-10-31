"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _normalize(h):
    """Enforce exact sum N/2 and bounds [0,1]."""
    h = h.copy()
    for _ in range(50):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final exact correction
    s = np.sum(h)
    diff = N/2.0 - s
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    return h

def _make_periodic(base, period):
    """Create a periodic version of h with given period, normalized to sum N/2."""
    h_p = np.tile(base[:period], N // period + 1)[:N]
    h_p = _normalize(h_p)
    return h_p

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _solve_periodic(base_init, period, alpha=500.0, maxiter=500):
    """Optimize a periodic pattern with given period using SLSQP."""
    n = period
    def obj(params):
        h = np.tile(params, N // n + 1)[:N]
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        smax = _smooth_max(overlap, alpha=alpha)
        return smax
    
    def con_sum(params):
        h = np.tile(params, N // n + 1)[:N]
        return np.sum(h) - N/2.0
    
    bounds = [(0.0, 1.0)] * n
    x0 = base_init.copy()
    
    try:
        result = minimize(
            obj, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'eq', 'fun': con_sum}],
            options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False}
        )
        if result.fun < 100.0:
            h_full = _normalize(np.tile(result.x, N // n + 1)[:N])
            return h_full, _score(h_full)
    except Exception:
        pass
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    def update(h, score):
        nonlocal best_h, best_score
        if h is not None and score < best_score:
            if np.all(h >= -1e-10) and np.all(h <= 1.0 + 1e-10) and abs(np.sum(h) - N/2.0) < 1e-6:
                best_h = h.copy()
                best_score = score
    
    # Phase 1: Try various periodic patterns
    periods_to_try = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 50]
    
    for period in periods_to_try:
        if time.time() > deadline - 10.0:
            break
        
        # Try multiple initial patterns for this period
        n_init = 3
        for init_idx in range(n_init):
            if time.time() > deadline - 5.0:
                break
            
            if init_idx == 0:
                # Uniform-ish start
                base = np.full(period, 0.5)
            elif init_idx == 1:
                # Alternating high/low
                base = np.zeros(period)
                for i in range(period):
                    if i % 2 == 0:
                        base[i] = 0.7
                    else:
                        base[i] = 0.3
            else:
                # Sinusoidal pattern
                base = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(period) / period)
            
            base = np.clip(base, 0.0, 1.0)
            
            h_opt, score_opt = _solve_periodic(base, period, alpha=500.0, maxiter=300)
            update(h_opt, score_opt)
    
    # Phase 2: Full optimization with smooth max, starting from best periodic
    if best_h is not None:
        h_init = best_h.copy()
        t_init = best_score + 0.01
        
        bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
        
        def obj_full(params):
            h = params[:N]
            t = params[N]
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            smax = _smooth_max(overlap, alpha=1000.0)
            penalty = 10000.0 * np.maximum(0.0, smax - t)**2
            return t + penalty
        
        def con_sum_full(params):
            return np.sum(params[:N]) - N/2.0
        
        def con_epi_full(params):
            h = params[:N]
            t = params[N]
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            return t - np.max(overlap)
        
        constraints = [
            {'type': 'eq', 'fun': con_sum_full},
            {'type': 'ineq', 'fun': con_epi_full}
        ]
        
        x0 = np.concatenate([h_init, [t_init]])
        
        try:
            if time.time() < deadline - 5.0:
                result = minimize(
                    obj_full, x0,
                    method='SLSQP',
                    bounds=bounds_ext,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-15, 'disp': False}
                )
                if result.fun < 100.0:
                    h_opt = _normalize(result.x[:N])
                    score_opt = _score(h_opt)
                    update(h_opt, score_opt)
        except Exception:
            pass
    
    # Phase 3: Local search - perturb and re-optimize
    if best_h is not None and time.time() < deadline - 5.0:
        np.random.seed(123)
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            
            # Small perturbation
            h_pert = best_h + np.random.normal(0, 0.01, N)
            h_pert = _normalize(h_pert)
            
            # Quick local optimization
            t_init = _score(h_pert) + 0.005
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def obj_local(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=2000.0)
                penalty = 50000.0 * np.maximum(0.0, smax - t)**2
                return t + penalty
            
            def con_sum_local(params):
                return np.sum(params[:N]) - N/2.0
            
            def con_epi_local(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return t - np.max(overlap)
            
            constraints = [
                {'type': 'eq', 'fun': con_sum_local},
                {'type': 'ineq', 'fun': con_epi_local}
            ]
            
            x0 = np.concatenate([h_pert, [t_init]])
            
            try:
                result = minimize(
                    obj_local, x0,
                    method='SLSQP',
                    bounds=bounds_ext,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-16, 'disp': False}
                )
                if result.fun < 100.0:
                    h_opt = _normalize(result.x[:N])
                    score_opt = _score(h_opt)
                    update(h_opt, score_opt)
            except Exception:
                pass
    
    if best_h is None:
        # Fallback: 2-block solution
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h0 = np.zeros(N)
        half = N // 2
        h0[:half] = A
        h0[half:] = B
        best_h = h0
    
    return best_h
