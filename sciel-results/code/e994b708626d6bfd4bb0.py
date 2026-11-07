"""Erdos minimum overlap - direct 100-dim optimization with multi-start and polish."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE = 140.0  # safety margin before 150s kill


def _correlation_value(h):
    """The evaluator's value: max(corr)/N*2. Lower is better."""
    corr = np.correlate(h, 1 - h, mode="full")
    return float(np.max(corr)) / N * 2.0


def _correlation_array(h):
    return np.correlate(h, 1 - h, mode="full")


def _project_to_feasible(h, target=None):
    """Project h to satisfy 0<=h<=1 and sum(h)=target within tolerance."""
    if target is None:
        target = N / 2.0
    h = np.array(h, dtype=float)
    # Iterative water-filling style projection
    for _ in range(100):
        h = np.clip(h, 0.0, 1.0)
        diff = target - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-12
        else:
            mask = h > 1e-12
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
    return h


def _objective_direct(h):
    """Direct objective on 100-dim h with projection."""
    h_feas = _project_to_feasible(h)
    return _correlation_value(h_feas)


def _soft_max_surrogate(h, lam):
    """Soft-max surrogate for the max of correlation."""
    corr = _correlation_array(h)
    m = np.max(corr)
    if lam <= 0:
        return m
    shifted = corr - m
    return m + np.log(np.sum(np.exp(lam * shifted)) / len(corr)) / lam


def _local_polish(h, max_iters=50, step_sizes=None):
    """Local polish using pairwise swaps that preserve sum exactly."""
    best_h = np.array(h, dtype=float)
    best_val = _correlation_value(best_h)
    if step_sizes is None:
        step_sizes = [0.005, 0.01, 0.02, 0.05, 0.1]
    n = len(best_h)
    corr = _correlation_array(best_h)
    
    for it in range(max_iters):
        improved = False
        max_idx = int(np.argmax(corr))
        max_lag = max_idx - (n - 1)
        
        # Try to reduce the max correlation by adjusting pairs
        for d in step_sizes:
            for sign in (+1, -1):
                best_pair_val = best_val
                best_pair = None
                best_i, best_j = -1, -1
                
                # Focus on indices involved in the max lag
                if abs(max_lag) < n:
                    involved = list(range(n - abs(max_lag)))
                    involved += [i + abs(max_lag) for i in range(n - abs(max_lag))]
                    involved = list(set(involved))
                else:
                    involved = list(range(n))
                
                # Sample a subset for speed
                if len(involved) > 30:
                    rng_local = np.random.default_rng(it * 1000 + int(d * 1000))
                    involved = list(rng_local.choice(involved, size=30, replace=False))
                
                for ii in range(len(involved)):
                    i = involved[ii]
                    for jj in range(ii + 1, len(involved)):
                        j = involved[jj]
                        ni = best_h[i] + sign * d
                        nj = best_h[j] - sign * d
                        if ni < 0.0 or ni > 1.0 or nj < 0.0 or nj > 1.0:
                            continue
                        new_h = best_h.copy()
                        new_h[i] = ni
                        new_h[j] = nj
                        val = _correlation_value(new_h)
                        if val < best_pair_val - 1e-14:
                            best_pair_val = val
                            best_pair = new_h.copy()
                            best_i, best_j = i, j
                            if val < best_val - 1e-13:
                                best_val = val
                                best_h = new_h
                                corr = _correlation_array(best_h)
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
        if not improved:
            break
    return best_h


def construct():
    start_time = time.time()

    def _remaining():
        return DEADLINE - (time.time() - start_time)

    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)  # seed value = 1.0

    # ---------------- Phase 1: Differential Evolution in 100-dim ----------------
    if _remaining() > 60.0:
        de_deadline = start_time + min(50.0, _remaining() - 10.0)
        try:
            def de_obj(h):
                if time.time() > de_deadline:
                    raise TimeoutError()
                return _objective_direct(h)

            bounds = [(0.0, 1.0)] * N
            res_de = differential_evolution(
                de_obj,
                bounds=bounds,
                maxiter=40, popsize=15, tol=1e-6,
                mutation=(0.5, 1.0), recombination=0.8,
                seed=42, polish=False, init='sobol')
            if res_de.x is not None:
                h_de = _project_to_feasible(res_de.x)
                v_de = _correlation_value(h_de)
                if v_de < best_val:
                    best_val = v_de
                    best_h = h_de.copy()
        except TimeoutError:
            pass
        except Exception:
            pass

    if _remaining() < 25.0:
        return _project_to_feasible(best_h)

    # ---------------- Phase 2: Multi-start L-BFGS-B with soft-max surrogate ----
    rng = np.random.default_rng(2024)
    starts = []
    
    # Current best
    starts.append(best_h.copy())
    
    # Random starts
    for _ in range(8):
        starts.append(rng.uniform(0.0, 1.0, N))
    
    # Bimodal patterns
    for phase in (0, 1):
        starts.append(np.array([0.8 if (i + phase) % 2 == 0 else 0.2 for i in range(N)]))
    
    # Sine patterns
    for freq in (1, 2, 3, 5):
        starts.append(np.clip(0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N), 0.0, 1.0))
    
    # Ramp patterns
    starts.append(np.linspace(0.0, 1.0, N))
    starts.append(np.linspace(1.0, 0.0, N))
    
    lbfgs_deadline = start_time + max(0.0, min(80.0, _remaining() - 20.0))
    for idx, h_start in enumerate(starts):
        if time.time() > lbfgs_deadline - 8.0:
            break
        h_start = np.clip(np.asarray(h_start, dtype=float), 0.0, 1.0)
        current_h = _project_to_feasible(h_start)
        
        for lam in (20.0, 80.0, 300.0):
            if time.time() > lbfgs_deadline - 4.0:
                break
            
            def soft_obj(h, lam=lam):
                h_feas = _project_to_feasible(h)
                return _soft_max_surrogate(h_feas, lam)
            
            try:
                res = minimize(soft_obj, current_h, method='L-BFGS-B',
                               bounds=[(0.0, 1.0)] * N,
                               options={'maxiter': 150, 'ftol': 1e-14})
                h_opt = _project_to_feasible(res.x)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_h = h_opt
            except Exception:
                pass
        
        # Exact SLSQP polish
        if time.time() > lbfgs_deadline - 3.0:
            break
        
        def exact_obj(h):
            return _objective_direct(h)
        
        try:
            res_exact = minimize(exact_obj, current_h, method='SLSQP',
                                 bounds=[(0.0, 1.0)] * N,
                                 options={'maxiter': 200, 'ftol': 1e-14})
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass

    # ---------------- Phase 3: Targeted local polish on top lags ---------------
    if _remaining() > 20.0:
        try:
            corr = _correlation_array(best_h)
            top_lags = np.argsort(corr)[-5:]  # 5 largest lags
            top_lags = sorted(top_lags.tolist())
            
            for lag_idx in top_lags:
                if time.time() > start_time + DEADLINE - 10.0:
                    break
                lag = lag_idx - (N - 1)
                for d in [0.01, 0.02, 0.05, 0.1]:
                    for sign in (+1, -1):
                        improved = False
                        for i in range(N - abs(lag)):
                            j = i + abs(lag)
                            ni = best_h[i] + sign * d
                            nj = best_h[j] - sign * d
                            if ni < 0.0 or ni > 1.0 or nj < 0.0 or nj > 1.0:
                                continue
                            new_h = best_h.copy()
                            new_h[i] = ni
                            new_h[j] = nj
                            val = _correlation_value(new_h)
                            if val < best_val - 1e-14:
                                best_val = val
                                best_h = new_h
                                improved = True
                        if not improved:
                            break
        except Exception:
            pass

    # ---------------- Phase 4: General local polish ----------------------------
    if _remaining() > 12.0:
        try:
            best_h = _local_polish(best_h, max_iters=30)
            best_val = _correlation_value(best_h)
        except Exception:
            pass

    # ---------------- Phase 5: Micro-polish on most influential elements --------
    if _remaining() > 6.0:
        try:
            corr = _correlation_array(best_h)
            max_lag_idx = int(np.argmax(corr))
            max_lag = max_lag_idx - (N - 1)
            involved = set()
            for i in range(N - abs(max_lag)):
                involved.add(i)
                involved.add(i + abs(max_lag))
            involved = sorted(involved)
            
            for d in [0.001, 0.003, 0.005, 0.01]:
                for sign in (+1, -1):
                    for i in involved:
                        if best_h[i] > 0.5:
                            candidates = [j for j in range(N) if j != i and best_h[j] < 0.3]
                        else:
                            candidates = [j for j in range(N) if j != i and best_h[j] > 0.7]
                        for j in candidates[:8]:
                            ni = best_h[i] - sign * d
                            nj = best_h[j] + sign * d
                            if ni < 0.0 or ni > 1.0 or nj < 0.0 or nj > 1.0:
                                continue
                            new_h = best_h.copy()
                            new_h[i] = ni
                            new_h[j] = nj
                            val = _correlation_value(new_h)
                            if val < best_val - 1e-15:
                                best_val = val
                                best_h = new_h
                                break
                    if time.time() > start_time + DEADLINE - 3.0:
                        break
                if time.time() > start_time + DEADLINE - 3.0:
                    break
        except Exception:
            pass

    # Final projection + safety
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val

    # Guard: ensure feasibility
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    best_h = np.clip(best_h, 0.0, 1.0)
    if abs(np.sum(best_h) - N / 2.0) > 1e-9 * N:
        best_h = _project_to_feasible(best_h)

    return best_h
