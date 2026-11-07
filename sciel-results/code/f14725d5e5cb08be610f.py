"""
Combined Erdos minimum overlap solver.
Strategy:
1. Global search: Differential Evolution over incommensurate-frequency sinusoidal basis.
2. Local Refinement: SLSQP with increasing smoothness (log-sum-exp) to navigate the max-correlation landscape.
3. High-Precision Polish: Tighter SLSQP tolerances to squeeze out remaining value.
4. Local Search: Exact 1-D coordinate-pair descent for fine-tuning.
"""

import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE_OFFSET = 145.0  # Safety margin for 150s limit


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(20):
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


def _construct_from_basis(coeffs):
    """
    Build h from incommensurate-frequency sinusoidal components.
    coeffs = [a1, p1, a2, p2, ..., aK, pK] where K = len(coeffs)//2
    h[i] = 0.5 + sum_k a_k * sin(2*pi*f_k*i/N + p_k)
    """
    i = np.arange(N, dtype=float)
    h = np.full(N, 0.5)
    
    # Incommensurate frequencies (golden-ratio related) to avoid periodicity
    freqs = [1.0, 1.618, 2.618, 4.236, 0.618, 3.236, 5.854, 1.236, 6.854, 2.236, 7.854, 3.854]
    
    K = len(coeffs) // 2
    for k in range(K):
        amp = coeffs[2 * k]
        phase = coeffs[2 * k + 1]
        f = freqs[k % len(freqs)]
        h += amp * np.sin(2 * np.pi * f * i / N + phase)
        
    return h


def _objective_from_coeffs(coeffs):
    """Objective: compute correlation value from basis coefficients."""
    h = _construct_from_basis(coeffs)
    h = _project_to_feasible(h)
    return _correlation_value(h)


def _fast_local_search(h, time_limit=10.0):
    """
    Exact 1-D coordinate-pair descent.
    For each pair (i, j), optimize the delta that moves h[i] by +delta and h[j] by -delta,
    keeping sum constant. Use golden-section search on the smooth surrogate.
    """
    start_time = time.time()
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    
    def _smooth_corr(h_local, lam=500.0):
        corr = np.correlate(h_local, 1 - h_local, mode="full")
        m = np.max(corr)
        if m > 1e15:
            return m
        shifted = (corr - m) * lam
        shifted -= np.max(shifted)  # numerical stability
        return m + np.log(np.sum(np.exp(shifted)) / len(corr)) / lam
    
    # Golden-section search for optimal delta in [delta_lo, delta_hi]
    def _golden_search(obj_fn, lo, hi, max_iter=40):
        gr = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
        a, b = lo, hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = obj_fn(c)
        fd = obj_fn(d)
        for _ in range(max_iter):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = obj_fn(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = obj_fn(d)
        return (a + b) / 2.0
    
    # Precompute partner pairs: for each i, consider a few j values
    # Use j = (i + offset) % n for several offsets to cover different correlation lags
    offsets = [1, 2, 3, 5, 10, 25, 50]
    
    for sweep in range(50):
        if time.time() > start_time + time_limit:
            break
        improved = False
        for i in range(n):
            if time.time() > start_time + time_limit:
                break
            for off in offsets:
                j = (i + off) % n
                if j == i:
                    continue
                # Determine feasible delta range
                # h[i] + delta in [0,1] => delta in [-h[i], 1-h[i]]
                # h[j] - delta in [0,1] => delta in [h[j]-1, h[j]]
                delta_lo = max(-best_h[i], best_h[j] - 1.0)
                delta_hi = min(1.0 - best_h[i], best_h[j])
                if delta_hi - delta_lo < 1e-10:
                    continue
                
                # Define objective as function of delta
                def obj_delta(delta):
                    new_h = best_h.copy()
                    new_h[i] = best_h[i] + delta
                    new_h[j] = best_h[j] - delta
                    return _smooth_corr(new_h, lam=500.0)
                
                # Golden-section search
                opt_delta = _golden_search(obj_delta, delta_lo, delta_hi, max_iter=30)
                
                # Evaluate at optimum and endpoints
                candidates = [delta_lo, opt_delta, delta_hi]
                best_delta = delta_lo
                best_cand_val = obj_delta(delta_lo)
                for cand in candidates[1:]:
                    cv = obj_delta(cand)
                    if cv < best_cand_val:
                        best_cand_val = cv
                        best_delta = cand
                
                # Apply if improved
                new_h = best_h.copy()
                new_h[i] = best_h[i] + best_delta
                new_h[j] = best_h[j] - best_delta
                new_val = _correlation_value(new_h)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    best_h = new_h
                    improved = True
        
        if not improved:
            break
    
    return best_h


def _high_precision_polish(h, time_budget=20.0):
    """
    High-precision polish using SLSQP with tighter tolerances.
    Uses log-sum-exp smoothing with increasing lambda to approximate max.
    """
    start_time = time.time()
    current_h = h.copy()
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    # Increasing lambda makes the surrogate closer to the true max
    for lam in [100.0, 200.0, 500.0, 1000.0, 2000.0]:
        if time.time() - start_time > time_budget - 5.0:
            break
        
        def smooth_obj(x, lam=lam):
            h_local = x
            corr = np.correlate(h_local, 1 - h_local, mode="full")
            m = np.max(corr)
            if m > 1e15:
                return m
            shifted = corr - m
            log_sum_exp = m + np.log(np.sum(np.exp(shifted)) / len(corr))
            return log_sum_exp
        
        try:
            res = minimize(
                smooth_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            h_opt = _project_to_feasible(res.x)
            val = _correlation_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
            current_h = h_opt
        except Exception:
            pass
    
    # Final exact optimization
    if time.time() - start_time < time_budget - 2.0:
        try:
            def exact_obj(x):
                h_local = x
                corr = np.correlate(h_local, 1 - h_local, mode="full")
                return np.max(corr)
            
            res_exact = minimize(
                exact_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 1000, 'ftol': 1e-18}
            )
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass
    
    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_OFFSET
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential Evolution over incommensurate-frequency sinusoidal basis
    # coeffs = [a1, p1, a2, p2, ..., aK, pK] with K=8 components
    n_components = 8
    n_params = 2 * n_components
    # Amplitude bounds: keep sum of max contributions < 0.5 to stay in [0,1]
    amp_bound = 0.15
    phase_bound = np.pi
    bounds = []
    for _ in range(n_components):
        bounds.append((-amp_bound, amp_bound))  # amplitude
        bounds.append((-phase_bound, phase_bound))  # phase
    
    def de_objective(coeffs):
        return _objective_from_coeffs(coeffs)
    
    de_deadline = start_time + 50.0
    try:
        def timed_de_objective(coeffs):
            if time.time() > de_deadline:
                raise TimeoutError()
            return de_objective(coeffs)
        
        result = differential_evolution(
            timed_de_objective,
            bounds=bounds,
            maxiter=150,
            popsize=20,
            tol=1e-10,
            mutation=(0.4, 1.0),
            recombination=0.8,
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
    
    # Phase 2: Multi-start SLSQP
    rng = np.random.default_rng(123)
    starts = [best_h.copy()]
    
    # Random starts
    for _ in range(5):
        h_rand = rng.uniform(0, 1, N)
        starts.append(_project_to_feasible(h_rand))
    
    # Structured starts
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))
    
    for block_size in [4, 5, 10]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h_block))
        
    # Incommensurate-frequency sinusoidal starts
    i_arr = np.arange(N, dtype=float)
    freqs_incomm = [1.0, 1.618, 2.618, 0.618, 3.236, 5.854]
    for f in freqs_incomm:
        amp = 0.3
        for ph in [0.0, np.pi/4, np.pi/2, 3*np.pi/4]:
            h_sin = 0.5 + amp * np.sin(2 * np.pi * f * i_arr / N + ph)
            starts.append(_project_to_feasible(h_sin))
    
    # Multi-frequency starts
    for f1, f2 in [(1.0, 1.618), (1.0, 2.618), (1.618, 0.618)]:
        h_multi = 0.5 + 0.2 * np.sin(2 * np.pi * f1 * i_arr / N) + 0.2 * np.sin(2 * np.pi * f2 * i_arr / N + 1.0)
        starts.append(_project_to_feasible(h_multi))

    slsqp_deadline = start_time + 110.0
    
    for idx, h_start in enumerate(starts):
        if time.time() > slsqp_deadline - 10.0:
            break
        
        h_start = _project_to_feasible(h_start)
        current_h = h_start
        
        # Coarse to fine smoothing
        for lam in [20.0, 100.0, 500.0]:
            if time.time() > slsqp_deadline - 5.0:
                break
            
            def smooth_obj(x, lam=lam):
                h = x
                corr = np.correlate(h, 1 - h, mode="full")
                m = np.max(corr)
                if m > 1e15:
                    return m
                return m + np.log(np.sum(np.exp((corr - m) * lam)) / len(corr)) / lam
            
            try:
                res = minimize(
                    smooth_obj,
                    current_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                h_opt = _project_to_feasible(res.x)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_h = h_opt
            except Exception:
                pass
        
        # Final exact optimization for this start
        if time.time() > slsqp_deadline - 3.0:
            break
        
        try:
            def exact_obj(x):
                h = x
                corr = np.correlate(h, 1 - h, mode="full")
                return np.max(corr)
            
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

    # Phase 3: High Precision Polish on the global best
    if time.time() < deadline - 30.0:
        try:
            best_h = _high_precision_polish(best_h, time_budget=25.0)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
            
    # Phase 4: Exact coordinate-pair local search
    if time.time() < deadline - 5.0:
        try:
            best_h = _fast_local_search(best_h, time_limit=15.0)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
            
    # Phase 5: One more polish after local search
    if time.time() < deadline - 3.0:
        try:
            best_h = _high_precision_polish(best_h, time_budget=5.0)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
            
    # Final Projection
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
        
    return best_h
