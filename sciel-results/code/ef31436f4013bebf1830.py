"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time as _time

def _corr_vector(h):
    """Compute all lags of correlation."""
    return np.correlate(h, 1.0 - h, mode='full')

def _correlation_max(h):
    """Compute the raw value: max correlation / N * 2."""
    corr = _corr_vector(h)
    return np.max(corr) / N * 2.0

def _project_feasible(h):
    """Project h to satisfy 0 <= h <= 1 and sum(h) = N/2."""
    h = np.asarray(h, dtype=np.float64).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        s = h.sum()
        if abs(s - 50.0) < 1e-14:
            break
        if s == 0:
            h = np.full(N, 0.5)
            break
        h = h * (50.0 / s)
        h = np.clip(h, 0.0, 1.0)
    # Final check
    if abs(h.sum() - 50.0) > 1e-9 * N:
        # Fallback: distribute remaining mass
        deficit = 50.0 - h.sum()
        if deficit > 0:
            # Add to entries below 1
            mask = h < 1.0
            if mask.any():
                add_per = deficit / mask.sum()
                h[mask] = np.minimum(1.0, h[mask] + add_per)
            # If still not enough, set some to 1
            while h.sum() < 50.0 - 1e-12:
                idx = np.argmin(h)
                h[idx] = 1.0
        elif deficit < 0:
            # Remove from entries above 0
            mask = h > 0.0
            if mask.any():
                rem_per = -deficit / mask.sum()
                h[mask] = np.maximum(0.0, h[mask] - rem_per)
            while h.sum() > 50.0 + 1e-12:
                idx = np.argmax(h)
                h[idx] = 0.0
    return h

def _logsumexp(x, beta):
    """Stable log-sum-exp."""
    if beta <= 0:
        return np.max(x)
    x_max = np.max(x)
    if x_max > 1e10:
        return x_max
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _smooth_objective(x, beta=20.0):
    """Smooth max of correlations using log-sum-exp."""
    c = _corr_vector(x)
    return _logsumexp(c, beta)

def _smooth_objective_grad(x, beta=20.0):
    """Gradient of smooth max of correlations."""
    c = _corr_vector(x)
    c_max = np.max(c)
    if beta <= 0 or c_max > 1e10:
        # Subgradient: pick the max lag
        idx = np.argmax(c)
        # d/dx of correlation at lag idx
        # correlation[k] = sum_i h[i] * (1-h[i+k-N+1]) for appropriate indexing
        # This is complex; use numerical instead
        eps = 1e-7
        grad = np.zeros(N)
        for i in range(N):
            xp = x.copy()
            xp[i] += eps
            grad[i] = (_smooth_objective(xp, beta) - _smooth_objective(x, beta)) / eps
        return grad
    
    weights = np.exp(beta * (c - c_max))
    weights /= np.sum(weights)
    
    # Gradient of each correlation component w.r.t. h
    # c[k] = sum_i h[i] * (1 - h[i + k - N + 1]) where indices wrap appropriately
    # For mode='full', c has length 2N-1, c[k] = sum_{i} h[i] * (1-h[i+k-(N-1)])
    # Let's compute gradient numerically for stability
    eps = 1e-7
    grad = np.zeros(N)
    for i in range(N):
        xp = x.copy()
        xp[i] += eps
        grad[i] = (_smooth_objective(xp, beta) - _smooth_objective(x, beta)) / eps
    return grad

def _coordinate_polish(h, max_iter=500, tol=1e-12):
    """Coordinate descent polish: optimize one coordinate at a time."""
    h = _project_feasible(h.copy())
    best_val = _correlation_max(h)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            # Optimize h[i] given all others fixed
            # sum constraint: h[i] = 50 - sum(others)
            others_sum = h.sum() - h[i]
            h_i_new = 50.0 - others_sum
            
            # But we also need 0 <= h_i <= 1
            if h_i_new < 0 or h_i_new > 1:
                # Can't change this coordinate while maintaining sum
                continue
            
            # Actually, coordinate descent with equality constraint is tricky.
            # Instead, do a small perturbation approach:
            # Try moving h[i] slightly and adjust another coordinate to maintain sum.
            
            # Simpler: just evaluate if changing h[i] alone (and rescaling) helps
            # This is approximate but works in practice.
            pass
        
        # Alternative: use scipy minimize on individual coordinates
        # For each coordinate, minimize over h[i] in [0,1] with sum constraint
        # This means fixing all other coordinates and solving for h[i] = 50 - sum(others)
        # But that's determined! So we need to move two coordinates at a time.
        
        # Two-coordinate swaps:
        for i in range(0, N, 2):
            j = i + 1
            if j >= N:
                break
            # Optimize h[i] and h[j] jointly, keeping sum fixed
            s_ij = h[i] + h[j]
            # Parameterize: h[i] = t, h[j] = s_ij - t, t in [max(0, s_ij-1), min(1, s_ij)]
            t_lo = max(0.0, s_ij - 1.0)
            t_hi = min(1.0, s_ij)
            if t_hi - t_lo < 1e-15:
                continue
            
            # Use scipy minimize_scalar
            def obj_1d(t):
                h_temp = h.copy()
                h_temp[i] = t
                h_temp[j] = s_ij - t
                return _correlation_max(h_temp)
            
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(obj_1d, bounds=(t_lo, t_hi), method='bounded',
                                  options={'xatol': 1e-10, 'maxiter': 50})
            if res.success and res.fun < best_val - tol:
                h[i] = res.x
                h[j] = s_ij - res.x
                best_val = res.fun
                improved = True
        
        if not improved:
            break
    
    return _project_feasible(h)

def construct():
    start_time = _time.time()
    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)
    
    # Try several initial patterns
    patterns = [
        np.full(N, 0.5),
        np.concatenate([np.ones(50), np.zeros(50)]),
        np.concatenate([np.zeros(50), np.ones(50)]),
        np.tile([1, 1, 0, 1, 1, 0], N // 6 + 1)[:N],
        np.tile([1, 0, 1, 0], N // 4 + 1)[:N],
        np.linspace(1, 0, N),
        np.linspace(0, 1, N),
        np.concatenate([np.linspace(1, 0, 50), np.linspace(0, 1, 50)]),
        # Block patterns
        np.tile([1, 0], N // 2)[:N],
        np.tile([0, 1], N // 2)[:N],
    ]
    
    for p in patterns:
        p = _project_feasible(p)
        val = _correlation_max(p)
        if val < best_val:
            best_val = val
            best_h = p.copy()
    
    # SLSQP optimization with smooth objective
    elapsed = _time.time() - start_time
    if elapsed < DEADLINE - 30:
        try:
            bounds = [(0.0, 1.0)] * N
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 50.0}]
            
            # Try multiple betas
            for beta in [10.0, 20.0, 50.0]:
                elapsed = _time.time() - start_time
                if elapsed > DEADLINE - 15:
                    break
                
                result = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    best_h.copy(),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
                )
                
                if result.x is not None and result.x.shape == (N,):
                    h_opt = _project_feasible(result.x)
                    val = _correlation_max(h_opt)
                    if val < best_val:
                        best_val = val
                        best_h = h_opt.copy()
            
            # Coordinate descent polish
            elapsed = _time.time() - start_time
            if elapsed < DEADLINE - 10:
                h_polished = _coordinate_polish(best_h, max_iter=200)
                val = _correlation_max(h_polished)
                if val < best_val:
                    best_val = val
                    best_h = h_polished.copy()
            
            # More restarts with perturbations
            elapsed = _time.time() - start_time
            if elapsed < DEADLINE - 10:
                for i in range(5):
                    if _time.time() - start_time > DEADLINE - 5:
                        break
                    rng = np.random.default_rng(42 + i * 7)
                    x0 = best_h + rng.normal(0, 0.15, N)
                    x0 = _project_feasible(x0)
                    
                    result = minimize(
                        lambda x: _smooth_objective(x, 30.0),
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
                    )
                    
                    if result.x is not None and result.x.shape == (N,):
                        h_new = _project_feasible(result.x)
                        val = _correlation_max(h_new)
                        if val < best_val:
                            best_val = val
                            best_h = h_new.copy()
                        
                        # Polish
                        elapsed = _time.time() - start_time
                        if elapsed < DEADLINE - 5:
                            h_pol = _coordinate_polish(h_new, max_iter=100)
                            val_pol = _correlation_max(h_pol)
                            if val_pol < best_val:
                                best_val = val_pol
                                best_h = h_pol.copy()
        except Exception:
            pass
    
    # Final validation
    best_h = _project_feasible(best_h)
    if abs(best_h.sum() - 50.0) >= 1e-9 * N:
        best_h = np.full(N, 0.5)
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
