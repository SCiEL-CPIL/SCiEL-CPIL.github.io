import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)

    def update_best(h, sc):
        nonlocal best_h, best_score
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    # Phase 1: Quick structured evaluations
    x = np.arange(N)
    center = N // 2

    # Delta (baseline, should give 1.0)
    h_delta = np.zeros(N); h_delta[0] = 1.0
    update_best(h_delta, compute_ratio(h_delta))

    # Uniform
    h_unif = np.ones(N)
    update_best(h_unif, compute_ratio(h_unif))

    # Sparse configurations with few nonzero entries
    for n_nonzero in [2, 3, 4, 5, 8, 10]:
        for trial in range(200):
            idx = np.random.choice(N, size=n_nonzero, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0 / n_nonzero
            sc = compute_ratio(h)
            update_best(h, sc)
        if time.time() > deadline - 60.0:
            break

    # Phase 2: Differential Evolution for global search
    def objective_neg(h_flat):
        h = np.maximum(h_flat, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        h2 = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        cinf = np.max(conv)
        if cinf < 1e-15:
            return 1e10
        return -h2 / cinf

    # Run DE with time-limited iterations
    try:
        if time.time() < deadline - 50.0:
            result = differential_evolution(
                objective_neg,
                bounds=[(0, 1)] * N,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=False
            )
            h_de = np.maximum(result.x, 0.0)
            s_de = np.sum(h_de)
            if s_de > 1e-15:
                h_de = h_de / s_de
                sc_de = compute_ratio(h_de)
                update_best(h_de, sc_de)
    except Exception:
        pass

    # Phase 3: Local refinement from best
    if time.time() < deadline - 20.0:
        h_start = best_h.copy()
        try:
            res = minimize(objective_neg, h_start, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            h_opt = np.maximum(res.x, 0.0)
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt = h_opt / s
                sc_opt = compute_ratio(h_opt)
                update_best(h_opt, sc_opt)
        except Exception:
            pass

    # Phase 4: Try sparse patterns with optimization
    if time.time() < deadline - 10.0:
        # Try all pairs
        for i in range(N):
            for j in range(i+1, N):
                h = np.zeros(N)
                h[i] = 0.5
                h[j] = 0.5
                sc = compute_ratio(h)
                update_best(h, sc)
            if time.time() > deadline - 10.0:
                break

    return list(best_h)
