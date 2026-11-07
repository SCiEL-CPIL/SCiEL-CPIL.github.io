"""Combined model-and-optimize with targeted local search for Erdos minimum overlap.

Key structural insight:
  corr[lag] = sum_i h[i]*(1-h[i-lag]) = sum(h) - autocorr[lag] = N/2 - autocorr[lag]
  So minimizing max(corr) = maximizing min(autocorr).

We use:
  1. Differential evolution over low-dimensional basis coefficients (global)
  2. SLSQP with smooth log-sum-exp surrogate (medium-scale local)
  3. Targeted pair-swap local search with autocorrelation-aware heuristics (fine polish)
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _raw_max_corr(h):
    """Raw max correlation (unnormalized)."""
    return np.max(np.correlate(h, 1 - h, mode="full"))


def _autocorr_min(h):
    """Compute min autocorrelation (since corr = N/2 - autocorr, max corr = N/2 - min autocorr)."""
    # autocorr[lag] = sum_i h[i] * h[i-lag]
    # We want to maximize the minimum autocorrelation
    ac = np.correlate(h, h, mode="full")
    return np.min(ac)


def _project_to_feasible(h):
    """Project h onto feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(50):
        if abs(diff) < 1e-15:
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


def _construct_from_basis(coeffs):
    """Build h from basis expansion."""
    n = len(coeffs)
    i = np.arange(N, dtype=float)
    h = np.full(N, 0.5)
    
    basis_funcs = [
        (i / N - 0.5),
        ((i / N - 0.5) ** 2 - 1/12),
        np.sin(2 * np.pi * i / N),
        np.cos(2 * np.pi * i / N),
        np.sin(4 * np.pi * i / N),
        np.cos(4 * np.pi * i / N),
        np.sin(6 * np.pi * i / N),
        np.cos(6 * np.pi * i / N),
        np.sin(10 * np.pi * i / N),
        np.cos(10 * np.pi * i / N),
        np.sin(14 * np.pi * i / N),
        np.cos(14 * np.pi * i / N),
    ]
    
    for k in range(min(n, len(basis_funcs))):
        h += coeffs[k] * basis_funcs[k]
    
    return h


def _objective_from_coeffs(coeffs):
    """Objective from basis coefficients."""
    h = _construct_from_basis(coeffs)
    h = _project_to_feasible(h)
    return _correlation_value(h)


def _smooth_max_corr(h, lam=50.0):
    """Smooth approximation of max(corr) via log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    # Numerically stable soft-max
    shifted = (corr - m) * lam
    # Clip to avoid overflow
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m + log_sum / lam


def _targeted_local_search(h, max_iters=5000, time_limit=None):
    """
    Targeted local search using autocorrelation insight.
    
    Since max(corr) = N/2 - min(autocorr), we want to maximize min(autocorr).
    The min autocorrelation occurs at some lag k*. We try to increase autocorr[k*]
    by moving mass to positions that increase h[i]*h[i+k*] terms.
    """
    best_h = h.copy()
    best_val = _raw_max_corr(best_h)
    
    start_time = time.time()
    if time_limit is None:
        time_limit = start_time + 25.0
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if time.time() > time_limit:
            break
        
        # Find the critical lag
        corr = np.correlate(best_h, 1 - best_h, mode="full")
        max_idx = np.argmax(corr)
        current_max = corr[max_idx]
        
        improved = False
        
        # Strategy 1: Binarization push - push h[i] toward 0 or 1
        # This tends to increase autocorrelation peaks
        for _ in range(40):
            i = rng.integers(N)
            j = rng.integers(N)
            if i == j:
                continue
            
            # Determine direction: push h[i] toward extreme
            if best_h[i] < 0.5:
                d = -best_h[i]  # set to 0
            else:
                d = 1.0 - best_h[i]  # set to 1
            
            # Check feasibility
            new_h_j = best_h[j] - d
            if new_h_j < -1e-12 or new_h_j > 1 + 1e-12:
                # Scale d to fit
                if d > 0:
                    d_max = min(1.0 - best_h[i], best_h[j])
                    d = min(d, d_max)
                else:
                    d_min = max(-best_h[i], best_h[j] - 1.0)
                    d = max(d, d_min)
                if abs(d) < 1e-8:
                    continue
            
            new_h = best_h.copy()
            new_h[i] += d
            new_h[j] -= d
            new_h = np.clip(new_h, 0, 1)
            
            # Quick check: does this improve?
            val = _raw_max_corr(new_h)
            if val < best_val - 1e-12:
                best_val = val
                best_h = new_h
                improved = True
                break
        
        # Strategy 2: Small random perturbations
        if not improved:
            for _ in range(15):
                i = rng.integers(N)
                j = rng.integers(N)
                if i == j:
                    continue
                d = rng.uniform(-0.03, 0.03)
                
                new_h_i = best_h[i] + d
                new_h_j = best_h[j] - d
                if 0 <= new_h_i <= 1 and 0 <= new_h_j <= 1:
                    new_h = best_h.copy()
                    new_h[i] = new_h_i
                    new_h[j] = new_h_j
                    val = _raw_max_corr(new_h)
                    if val < best_val - 1e-12:
                        best_val = val
                        best_h = new_h
                        improved = True
                        break
        
        # Strategy 3: Targeted move - increase autocorr at critical lag
        if not improved:
            # Find lag k* corresponding to max_idx
            # corr index max_idx corresponds to lag = max_idx - (N-1)
            k_star = max_idx - (N - 1)
            
            # Try to increase P_k* = sum h[i]*h[i+k*]
            # Find pairs (i, i+k*) where both can be increased
            for _ in range(10):
                i = rng.integers(N)
                # j should be i + k_star or i - k_star (within bounds)
                if k_star >= 0:
                    j = i + k_star
                else:
                    j = i - k_star
                if j < 0 or j >= N:
                    continue
                
                # Try to increase both h[i] and h[j] slightly
                # by taking from a random k
                k_src = rng.integers(N)
                if k_src == i or k_src == j:
                    continue
                
                delta = rng.uniform(0.005, 0.02)
                if best_h[k_src] < delta:
                    continue
                if best_h[i] + delta > 1 or best_h[j] + delta > 1:
                    continue
                
                new_h = best_h.copy()
                new_h[i] += delta
                new_h[j] += delta
                new_h[k_src] -= 2 * delta
                if new_h[k_src] < 0:
                    continue
                
                val = _raw_max_corr(new_h)
                if val < best_val - 1e-12:
                    best_val = val
                    best_h = new_h
                    improved = True
                    break
        
        if not improved and it > 200:
            # Stuck for a while, try a different random restart within local region
            if it > 400:
                break
    
    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)

    # Phase 1: Differential evolution over basis coefficients
    n_basis = 12
    bounds = [(-0.5, 0.5)] * n_basis

    de_deadline = start_time + 45.0

    try:
        def timed_de_objective(coeffs):
            if time.time() > de_deadline:
                raise TimeoutError()
            return _objective_from_coeffs(coeffs)

        result = differential_evolution(
            timed_de_objective,
            bounds=bounds,
            maxiter=80,
            popsize=15,
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

    starts = [best_h.copy()]

    # Random feasible starts
    for _ in range(6):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_feasible(h_rand)
        starts.append(h_rand)

    # Structured: alternating
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))

    # Structured: blocks
    for block_size in [4, 5, 10]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h_block))

    # Sine waves
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

    slsqp_deadline = start_time + 115.0

    for idx, h_start in enumerate(starts):
        if time.time() > slsqp_deadline - 8.0:
            break

        h_start = _project_to_feasible(h_start)
        current_h = h_start

        # Smooth surrogate optimization with increasing lambda
        for lam in [20.0, 100.0, 300.0]:
            if time.time() > slsqp_deadline - 4.0:
                break

            def smooth_obj(x, lam=lam):
                return _smooth_max_corr(x, lam)

            try:
                res = minimize(
                    smooth_obj,
                    current_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                    options={'maxiter': 150, 'ftol': 1e-14}
                )
                h_opt = _project_to_feasible(res.x)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_h = h_opt
            except Exception:
                pass

        # Final exact objective optimization
        if time.time() > slsqp_deadline - 2.0:
            break

        try:
            def exact_obj(x):
                return _raw_max_corr(x)

            res_exact = minimize(
                exact_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 200, 'ftol': 1e-15}
            )
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass

    # Phase 3: Targeted local search polish
    if time.time() < deadline - 10.0:
        try:
            local_time_limit = deadline - 5.0
            best_h = _targeted_local_search(best_h, max_iters=8000, time_limit=local_time_limit)
            best_val = _correlation_value(best_h)
        except Exception:
            pass

    # Final projection and validation
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val

    return best_h
