"""Model-and-optimize approach for Erdos minimum overlap.

Key insight: the max correlation value is driven by the peaks of the
correlation function. We parameterize h as a sum of a few sinusoidal
components with incommensurate frequencies, optimize the parameters
with differential evolution, then polish with SLSQP and local search.
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
    shifted = corr - m
    log_sum = m + np.log(np.sum(np.exp(shifted * lam)) / len(corr)) / lam
    return log_sum


def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(30):
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h


def _construct_from_sinusoids(params):
    """
    Build h from a sum of sinusoidal components.
    params = [a1, p1, a2, p2, ..., aK, pK, offset]
    where K = len(params) // 2 - 1 (last param is offset)
    Frequencies are fixed incommensurate values.
    """
    n_params = len(params)
    n_sinusoids = (n_params - 1) // 2
    offset = params[-1]
    
    # Fixed incommensurate frequencies
    freqs = np.array([1.0, 1.618, 2.236, 3.142, 4.123, 5.031, 6.283, 7.141, 8.091, 9.231])[:n_sinusoids]
    
    x = np.arange(N) / N
    h = np.full(N, offset)
    
    for k in range(n_sinusoids):
        amp = params[2 * k]
        phase = params[2 * k + 1]
        h += amp * np.sin(2 * np.pi * freqs[k] * x + phase)
    
    return h


def _objective_from_sinusoids(params):
    """Objective: compute correlation value from sinusoid parameters."""
    h = _construct_from_sinusoids(params)
    h = _project_to_feasible(h)
    return _correlation_value(h)


def _local_polish(h, max_iters=300, deadline=None, rng=None):
    """
    Fast stochastic local search: try random pair adjustments.
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    
    # Adaptive step sizes
    base_steps = [0.005, 0.01, 0.02, 0.05, 0.1]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        for _ in range(80):
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
    
    # Phase 1: Differential evolution over sinusoidal parameters
    n_sinusoids = 6
    n_params = 2 * n_sinusoids + 1  # amplitudes, phases, offset
    bounds = []
    for k in range(n_sinusoids):
        bounds.append((0.0, 0.3))  # amplitude
        bounds.append((0.0, 2 * np.pi))  # phase
    bounds.append((0.3, 0.7))  # offset
    
    def de_objective(params):
        return _objective_from_sinusoids(params)
    
    de_deadline = start_time + 40.0
    
    try:
        def timed_de_objective(params):
            if time.time() > de_deadline:
                raise TimeoutError()
            return de_objective(params)
        
        result = differential_evolution(
            timed_de_objective,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            tol=1e-10,
            mutation=(0.4, 1.2),
            recombination=0.8,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.x is not None:
            h_de = _project_to_feasible(_construct_from_sinusoids(result.x))
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
    for _ in range(8):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_feasible(h_rand)
        starts.append(h_rand)
    
    # Structured: alternating 0s and 1s
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))
    
    # Structured: blocks
    for block_size in [3, 4, 5, 7, 10]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h_block))
    
    # Sine waves at various frequencies
    for freq in [1, 2, 3, 4, 5, 7, 10, 13]:
        amp = min(0.4, 0.5 / freq)
        h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
        starts.append(_project_to_feasible(h_sin))
    
    # Multi-frequency sinusoids
    for freqs in [(1, 3, 7), (2, 5, 11), (1, 4, 9), (3, 5, 8)]:
        h_multi = np.full(N, 0.5)
        for f in freqs:
            amp = 0.15 / len(freqs)
            h_multi += amp * np.sin(2 * np.pi * f * np.arange(N) / N)
        starts.append(_project_to_feasible(h_multi))
    
    # Triangular waves
    for period in [8, 10, 12, 16, 20, 25]:
        h_tri = np.zeros(N)
        for i in range(N):
            t = (i % period) / period
            if t < 0.5:
                h_tri[i] = 2 * t
            else:
                h_tri[i] = 2 - 2 * t
        starts.append(_project_to_feasible(h_tri))
    
    # Sawtooth waves
    for period in [10, 15, 20]:
        h_saw = np.zeros(N)
        for i in range(N):
            t = (i % period) / period
            h_saw[i] = t
        starts.append(_project_to_feasible(h_saw))
    
    # Optimize from each start with SLSQP using smooth surrogate
    slsqp_deadline = start_time + 118.0
    
    for idx, h_start in enumerate(starts):
        if time.time() > slsqp_deadline - 8.0:
            break
        
        h_start = _project_to_feasible(h_start)
        current_h = h_start
        
        # Smooth objective with increasing lambda for better convergence
        for lam in [50.0, 200.0, 1000.0, 5000.0]:
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
                    options={'maxiter': 200, 'ftol': 1e-13}
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
                options={'maxiter': 300, 'ftol': 1e-15}
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
            best_h = _local_polish(best_h, max_iters=300, deadline=polish_deadline)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 1.0:
        try:
            def exact_obj_final(x):
                return np.max(np.correlate(x, 1 - x, mode="full"))
            
            res_final = minimize(
                exact_obj_final,
                best_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            h_final = _project_to_feasible(res_final.x)
            val_final = _correlation_value(h_final)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final.copy()
        except Exception:
            pass
    
    # Final projection and validation
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
