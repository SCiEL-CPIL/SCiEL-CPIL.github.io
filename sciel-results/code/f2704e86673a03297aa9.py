"""Model-and-optimize approach for Erdos minimum overlap.

Key insight: the max correlation value is driven by the peaks of the
correlation function. We can parameterize h as a smooth function of a
few basis coefficients, then directly optimize those parameters.
"""
N = 100
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _smooth_max_correlation(h, lam):
    """Smooth surrogate for max(corr) using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    if m > 1e15:
        return m
    # log-sum-exp for numerical stability
    shifted = corr - m
    log_sum = m + np.log(np.sum(np.exp(shifted * lam)) / len(corr)) / lam
    return log_sum


def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(20):
        if abs(diff) < 1e-14:
            break
        # Only adjust elements that can absorb the correction
        if diff > 0:
            # Need to decrease sum: reduce elements that are > 0
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            # Need to increase sum: increase elements that are < 1
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)  # diff is negative, so this adds
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h


def _construct_from_basis(coeffs):
    """
    Build h from piecewise linear control points.
    coeffs are the values at len(coeffs) equally spaced points.
    """
    n_ctrl = len(coeffs)
    if n_ctrl < 2:
        return np.full(N, 0.5)
    
    # Control points are at indices 0, N/(n_ctrl-1), ..., N-1
    x_ctrl = np.linspace(0, N - 1, n_ctrl)
    y_ctrl = np.asarray(coeffs)
    
    x_full = np.arange(N)
    h = np.interp(x_full, x_ctrl, y_ctrl)
    
    return h


def _objective_from_coeffs(coeffs):
    """Objective: compute correlation value from basis coefficients."""
    h = _construct_from_basis(coeffs)
    h = _project_to_feasible(h)
    return _correlation_value(h)


def _local_polish(h, max_iters=200, deadline=None, rng=None):
    """
    Fast stochastic local search: try random pair adjustments.
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    
    # Step sizes to try
    base_steps = [0.01, 0.05, 0.1]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Try a few random moves
        for _ in range(50):
            if deadline is not None and time.time() > deadline:
                break
                
            i, j = rng.choice(n, size=2, replace=False)
            d = base_steps[rng.integers(0, len(base_steps))]
            
            # Try h[i] += d, h[j] -= d
            if best_h[i] + d <= 1.0 and best_h[j] - d >= 0.0:
                new_h = best_h.copy()
                new_h[i] += d
                new_h[j] -= d
                val = _correlation_value(new_h)
                if val < best_val - 1e-12:
                    best_val = val
                    best_h = new_h
            else:
                # Try h[i] -= d, h[j] += d
                if best_h[i] - d >= 0.0 and best_h[j] + d <= 1.0:
                    new_h = best_h.copy()
                    new_h[i] -= d
                    new_h[j] += d
                    val = _correlation_value(new_h)
                    if val < best_val - 1e-12:
                        best_val = val
                        best_h = new_h
                        
    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential evolution over piecewise linear control points
    n_basis = 10
    bounds = [(0.0, 1.0)] * n_basis
    
    def de_objective(coeffs):
        return _objective_from_coeffs(coeffs)
    
    # Run differential evolution with time budget
    de_deadline = start_time + 45.0
    
    try:
        def timed_de_objective(coeffs):
            if time.time() > de_deadline:
                raise TimeoutError()
            return de_objective(coeffs)
        
        result = differential_evolution(
            timed_de_objective,
            bounds=bounds,
            maxiter=80,
            popsize=12,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.x is not None:
            h_de = _project_to_feasible(_construct_from_basis(result.x))
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Phase 2: Multi-start SLSQP with smooth surrogate
    rng = np.random.default_rng(123)
    
    # Generate diverse starting points
    starts = [best_h.copy()]
    
    # Random feasible starts
    for _ in range(6):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_feasible(h_rand)
        starts.append(h_rand)
    
    # Structured: alternating 0s and 1s
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))
    
    # Structured: blocks
    for block_size in [4, 5, 10]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h_block))
    
    # Sine waves at various frequencies
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        amp = min(0.4, 0.5 / freq)
        h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
        starts.append(_project_to_feasible(h_sin))
    
    # Triangular waves
    for period in [10, 20, 25]:
        h_tri = np.zeros(N)
        for i in range(N):
            t = (i % period) / period
            if t < 0.5:
                h_tri[i] = 2 * t
            else:
                h_tri[i] = 2 - 2 * t
        starts.append(_project_to_feasible(h_tri))
    
    # Optimize from each start with SLSQP using smooth surrogate
    slsqp_deadline = start_time + 115.0
    
    for idx, h_start in enumerate(starts):
        if time.time() > slsqp_deadline - 8.0:
            break
        
        h_start = _project_to_feasible(h_start)
        current_h = h_start
        
        # Smooth objective with increasing lambda for better convergence
        for lam in [50.0, 200.0, 1000.0]:
            if time.time() > slsqp_deadline - 4.0:
                break
            
            def smooth_obj(x, lam=lam):
                return _smooth_max_correlation(x, lam)
            
            try:
                res = minimize(
                    smooth_obj,
                    current_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                    options={'maxiter': 150, 'ftol': 1e-12}
                )
                h_opt = _project_to_feasible(res.x)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_h = h_opt
            except Exception:
                pass
        
        # Final exact optimization with tight tolerance
        if time.time() > slsqp_deadline - 2.0:
            break
        
        try:
            def exact_obj(x):
                return np.max(np.correlate(x, 1 - x, mode="full"))
            
            res_exact = minimize(
                exact_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass
    
    # Phase 3: Local search polish with tight tolerance
    if time.time() < deadline - 3.0:
        try:
            polish_deadline = deadline - 1.0
            best_h = _local_polish(best_h, max_iters=200, deadline=polish_deadline)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
    
    # Final projection and validation
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
