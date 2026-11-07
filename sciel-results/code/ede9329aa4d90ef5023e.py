import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
import time

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 135.0  # safety margin before 150s kill

def _parametric_h(params):
    """
    Generate h from low-dimensional parameters using Poisson-like smoothing.
    params: [a1, f1, p1, a2, f2, p2, a3, f3, p3]
    h[i] = 0.5 + sum_k a_k * sin(2*pi*f_k*i/N + p_k)
    Then project to [0,1] and fix sum.
    """
    if len(params) < 9:
        params = np.pad(params, (0, 9 - len(params)))
    h = np.full(N, 0.5)
    for k in range(3):
        a = params[3*k]
        f = params[3*k+1]
        p = params[3*k+2]
        h += a * np.sin(2.0 * np.pi * f * np.arange(N) / N + p)
    return _project_feasible(h)

def _parametric_objective(params):
    """Objective for parametric search."""
    h = _parametric_h(params)
    return _correlation_max(h)

def _corr_vector(h):
    """Compute all lags of correlation."""
    return np.correlate(h, 1.0 - h, mode='full')

def _correlation_max(h):
    """Compute the raw value: max correlation / N * 2."""
    corr = _corr_vector(h)
    return np.max(corr) / N * 2.0

def _project_feasible(h):
    """Project h to satisfy 0 <= h <= 1 and sum(h) = N/2."""
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(1000):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-14:
            break
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        n_free = np.sum(mask)
        if n_free == 0:
            # If all are at bounds, distribute equally among those at bounds
            # This shouldn't happen with proper initialization
            break
        h[mask] += diff / n_free
        h = np.clip(h, 0.0, 1.0)
    # Final adjustment for numerical precision
    diff = TARGET_SUM - np.sum(h)
    if abs(diff) > 1e-15:
        # Distribute to the element closest to 0.5
        idx = np.argmin(np.abs(h - 0.5))
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _logsumexp(x, beta=1.0):
    """Compute log(sum(exp(x/beta))) * beta, which approximates max(x) as beta->0."""
    x_max = np.max(x)
    if x_max == -np.inf:
        return x_max
    shifted = x - x_max
    return x_max + beta * np.log(np.sum(np.exp(shifted / beta)))

def _smooth_objective(h, beta=0.05):
    """Smooth approximation of the max correlation using log-sum-exp."""
    corr = _corr_vector(h)
    ls = _logsumexp(corr, beta)
    return ls / N * 2.0

def _smooth_objective_grad(h, beta=0.05):
    """Gradient of the smooth objective."""
    corr = _corr_vector(h)
    x_max = np.max(corr)
    shifted = corr - x_max
    weights = np.exp(shifted / beta) / np.sum(np.exp(shifted / beta))
    
    # Gradient of corr w.r.t. h
    # corr[lag] = sum_i h[i] * (1-h[i+lag]) for lag >= 0 (with appropriate indexing)
    # Using the full correlation: corr[k] = sum_i h[i] * (1-h[i+k-N+1]) for k in [0, 2N-2]
    # More precisely, np.correlate(h, 1-h, 'full')[k] = sum_i h[i] * (1-h[i + k - (N-1)])
    
    # Let's compute gradient numerically for reliability
    dh = np.zeros(N)
    eps = 1e-8
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        h_minus = h.copy()
        h_minus[i] -= eps
        corr_plus = _corr_vector(h_plus)
        corr_minus = _corr_vector(h_minus)
        dc = (corr_plus - corr_minus) / (2 * eps)
        dh[i] = np.sum(weights * dc)
    
    return 2.0 * dh / N

def _coordinate_polish(h, max_iters=100, tol=1e-12):
    """
    Coordinate descent: for each i, optimize h[i] while maintaining sum constraint.
    We fix all other h[j] and adjust h[i] and one other element to maintain sum.
    Actually, simpler: for each i, try to optimize h[i] by adjusting it and compensating
    with a small change to another element, or use the fact that the sum constraint
    means we can express one variable in terms of others.
    
    Better approach: for each i, we can compute the optimal h[i] given the sum constraint
    by noting that the correlation is linear in h[i] for fixed other h[j].
    But the max over lags makes it piecewise linear.
    
    Simpler: use minimize_scalar on h[i] with the constraint handled by adjusting
    another element. Let's pick element j != i to adjust.
    """
    h = _project_feasible(h).copy()
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            # Choose j to compensate (pick j with h[j] closest to 0.5 for flexibility)
            j = np.argmin(np.abs(h - 0.5))
            if j == i:
                j = (i + 1) % N
            
            def obj_1d(x_i):
                # h[j] = TARGET_SUM - sum(h) + x_i (to maintain sum)
                h_j = TARGET_SUM - (np.sum(h) - h[i]) + x_i
                if h_j < 0.0 or h_j > 1.0:
                    return 1e10
                h_temp = h.copy()
                h_temp[i] = x_i
                h_temp[j] = h_j
                return _correlation_max(h_temp)
            
            # Determine valid range for x_i
            # h[j] = TARGET_SUM - sum(h) + x_i must be in [0, 1]
            sum_others = np.sum(h) - h[i]
            x_min = max(0.0, -(TARGET_SUM - sum_others))  # h[j] >= 0 => x_i >= sum_others - TARGET_SUM
            x_max = min(1.0, 1.0 - (TARGET_SUM - sum_others))  # h[j] <= 1 => x_i <= 1 - TARGET_SUM + sum_others
            x_min = max(0.0, x_min)
            x_max = min(1.0, x_max)
            
            if x_min >= x_max:
                continue
            
            try:
                res = minimize_scalar(obj_1d, bounds=(x_min, x_max), method='bounded',
                                     options={'xatol': tol})
                if res.success:
                    h_j = TARGET_SUM - (np.sum(h) - h[i]) + res.x
                    if 0.0 <= h_j <= 1.0:
                        h_temp = h.copy()
                        h_temp[i] = res.x
                        h_temp[j] = h_j
                        val_temp = _correlation_max(h_temp)
                        if val_temp < _correlation_max(h) - tol:
                            h = h_temp
                            improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return _project_feasible(h)

def construct():
    start_time = time.time()
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)
    
    # Phase 1: Differential Evolution for global search (fast, parallelizable)
    if time.time() - start_time < DEADLINE - 80:
        bounds = [(0.0, 1.0) for _ in range(N)]
        
        def de_objective(x):
            h = _project_feasible(x)
            return _smooth_objective(h, beta=0.1)
        
        try:
            res = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                init='latinhypercube',
                update_interval=10,
                polish=True,
                workers=-1
            )
            h_res = _project_feasible(res.x)
            val = _correlation_max(h_res)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass
    
    # Phase 2: SLSQP with smooth objective from best solution
    if time.time() - start_time < DEADLINE - 40:
        bounds = [(0.0, 1.0) for _ in range(N)]
        constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM, 'jac': lambda x: np.ones(N)}
        
        # Try from current best
        try:
            res = minimize(
                lambda x: _smooth_objective(x, 0.05),
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=[constraint],
                jac=lambda x: _smooth_objective_grad(x, 0.05),
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            h_res = _project_feasible(res.x)
            val = _correlation_max(h_res)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass
        
        # Try from alternating patterns
        for base in [0.1, 0.2, 0.3, 0.4]:
            if time.time() - start_time > DEADLINE - 30:
                break
            alt = np.where(np.arange(N) % 2 == 0, base, 1.0 - base)
            h_alt = _project_feasible(alt)
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, 0.05),
                    h_alt,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[constraint],
                    jac=lambda x: _smooth_objective_grad(x, 0.05),
                    options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
                )
                h_res = _project_feasible(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
        
        # Try from sinusoidal patterns
        for freq in [1, 2, 3, 5]:
            if time.time() - start_time > DEADLINE - 25:
                break
            t = np.arange(N)
            for amp in [0.2, 0.3, 0.4]:
                for phase in [0, np.pi/2, np.pi]:
                    h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * t / N + phase)
                    h_sin = _project_feasible(h_sin)
                    try:
                        res = minimize(
                            lambda x: _smooth_objective(x, 0.05),
                            h_sin,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=[constraint],
                            jac=lambda x: _smooth_objective_grad(x, 0.05),
                            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
                        )
                        h_res = _project_feasible(res.x)
                        val = _correlation_max(h_res)
                        if val < best_val - 1e-10:
                            best_val = val
                            best_h = h_res.copy()
                    except Exception:
                        pass
    
    # Phase 3: Parametric search with multiple restarts
    if time.time() - start_time < DEADLINE - 30:
        # Coarse grid over frequency parameters
        freq_choices = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]
        amp_choices = [0.15, 0.25, 0.35, 0.45]
        
        best_param_val = best_val
        best_param_h = best_h.copy()
        
        for f1 in freq_choices:
            if time.time() - start_time > DEADLINE - 25:
                break
            for f2 in freq_choices[freq_choices.index(f1):]:
                if time.time() - start_time > DEADLINE - 20:
                    break
                for f3 in freq_choices[max(0, freq_choices.index(f2)-1):]:
                    if time.time() - start_time > DEADLINE - 15:
                        break
                    for a1 in amp_choices:
                        if time.time() - start_time > DEADLINE - 10:
                            break
                        for a2 in amp_choices:
                            if time.time() - start_time > DEADLINE - 5:
                                break
                            # Quick eval with fixed phases
                            for p1 in [0, np.pi/2]:
                                for p2 in [0, np.pi/2]:
                                    for p3 in [0, np.pi/2]:
                                        params = [a1, f1, p1, a2, f2, p2, 0.1, f3, p3]
                                        h_p = _parametric_h(params)
                                        val_p = _correlation_max(h_p)
                                        if val_p < best_param_val - 1e-10:
                                            best_param_val = val_p
                                            best_param_h = h_p.copy()
        
        if best_param_val < best_val - 1e-10:
            best_val = best_param_val
            best_h = best_param_h.copy()
        
        # Local refinement of best parametric solution using SLSQP on full h
        if time.time() - start_time < DEADLINE - 10:
            bounds = [(0.0, 1.0) for _ in range(N)]
            constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM, 'jac': lambda x: np.ones(N)}
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, 0.03),
                    best_h,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[constraint],
                    options={'maxiter': 400, 'ftol': 1e-13, 'disp': False}
                )
                h_res = _project_feasible(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-12:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
    
    # Phase 3b: Coordinate descent polish
    if time.time() - start_time < DEADLINE - 10:
        h_cd = _coordinate_polish(best_h, max_iters=50, tol=1e-12)
        val_cd = _correlation_max(h_cd)
        if val_cd < best_val - 1e-12:
            best_val = val_cd
            best_h = h_cd.copy()
    
    # Phase 4: Final SLSQP with tighter tolerance (no numerical grad for speed)
    if time.time() - start_time < DEADLINE - 5:
        bounds = [(0.0, 1.0) for _ in range(N)]
        constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM, 'jac': lambda x: np.ones(N)}
        
        try:
            res = minimize(
                lambda x: _smooth_objective(x, 0.02),
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=[constraint],
                options={'maxiter': 400, 'ftol': 1e-14, 'disp': False}
            )
            h_polish = _project_feasible(res.x)
            val_polish = _correlation_max(h_polish)
            if val_polish < best_val - 1e-12:
                best_val = val_polish
                best_h = h_polish.copy()
        except Exception:
            pass
    
    # Final coordinate polish
    if time.time() - start_time < DEADLINE - 2:
        h_final = _coordinate_polish(best_h, max_iters=30, tol=1e-14)
        val_final = _correlation_max(h_final)
        if val_final < best_val - 1e-14:
            best_val = val_final
            best_h = h_final.copy()
    
    # Ensure final validity
    best_h = _project_feasible(best_h)
    
    return best_h
