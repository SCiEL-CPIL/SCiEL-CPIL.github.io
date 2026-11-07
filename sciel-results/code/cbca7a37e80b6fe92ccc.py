import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _true_max_corr(h):
    h = np.asarray(h, dtype=np.float64)
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr))

def _value(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    hc = np.clip(h, 0.0, 1.0)
    return _true_max_corr(hc) / N * 2.0

def _smooth_max(c, tau):
    c = np.asarray(c, dtype=np.float64)
    m = float(np.max(c))
    if not np.isfinite(m):
        return np.inf
    z = c / tau - m / tau
    return tau * (m / tau + float(np.log(np.sum(np.exp(z)))))

def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    best_h = np.full(N, 0.5)
    best_val = _value(best_h)

    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}]

    def project_sum(h):
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-12:
            h = h + diff / N
            h = np.clip(h, 0.0, 1.0)
            diff = N / 2.0 - np.sum(h)
            if abs(diff) > 1e-12:
                h = h + diff / N
                h = np.clip(h, 0.0, 1.0)
        return h

    def smooth_objective(x, tau):
        h = np.clip(x, 0.0, 1.0)
        corr = np.correlate(h, 1.0 - h, mode="full")
        return _smooth_max(corr, tau)

    def try_optimize(start_h, tau_coarse=1.0, tau_fine=0.05, maxiter=300):
        nonlocal best_h, best_val
        x = project_sum(np.asarray(start_h, dtype=np.float64)).copy()
        try:
            # Coarse pass
            res = minimize(
                lambda v: smooth_objective(v, tau_coarse), x,
                method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            x = project_sum(res.x)
            # Fine pass
            res = minimize(
                lambda v: smooth_objective(v, tau_fine), x,
                method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            x = project_sum(res.x)
        except Exception:
            pass

        # Cheap coordinate-descent polish on the TRUE max
        try:
            x = _polish(x, deadline)
        except Exception:
            pass

        x = project_sum(x)
        val = _value(x)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_h = x.copy()
        return x

    def _polish(x, dl):
        x = np.array(x, dtype=np.float64)
        cur = _true_max_corr(x)
        n_imp = 0
        for _ in range(4000):
            if time.time() > dl:
                break
            if n_imp > 300:
                break
            improved = False
            step = 0.02
            for i in range(N):
                if time.time() > dl:
                    break
                orig = x[i]
                for d in (step, -step, step * 0.5, -step * 0.5):
                    x[i] = np.clip(orig + d, 0.0, 1.0)
                    # keep sum ~50 by adjusting a free coordinate
                    diff = N / 2.0 - np.sum(x)
                    if abs(diff) > 1e-9:
                        # adjust a different index
                        j = (i + 1) % N
                        xj = x[j]
                        x[j] = np.clip(xj - diff, 0.0, 1.0)
                        diff2 = N / 2.0 - np.sum(x)
                        if abs(diff2) > 1e-9:
                            x[:] = orig + np.where(np.arange(N) == i, 0, 0)  # revert
                            x[i] = orig
                            x = project_sum(x)
                            break
                    v = _true_max_corr(x)
                    if v < cur - 1e-12:
                        cur = v
                        improved = True
                        break
                if not improved:
                    x[i] = orig
                else:
                    n_imp += 1
            if not improved:
                break
        return x

    # --- Diverse initializations ---
    rng = np.random.default_rng(100)
    inits = []

    # Seed
    inits.append(np.full(N, 0.5))

    # Alternating
    alt = np.tile([0.0, 1.0], N // 2)
    inits.append(project_sum(alt))

    # Sine waves at several frequencies
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 25]:
        for phase in [0.0, np.pi / 2, np.pi]:
            t = np.arange(N)
            s = 0.5 + 0.45 * np.sin(2 * np.pi * freq * t / N + phase)
            inits.append(project_sum(s))

    # Blocks
    for bl in [2, 3, 5, 10, 20]:
        blk = np.tile([1.0, 0.0], N // bl)
        if len(blk) < N:
            blk = np.concatenate([blk, np.zeros(N - len(blk))])
        inits.append(project_sum(blk))

    # Random starts
    for _ in range(12):
        r = rng.uniform(0, 1, N)
        inits.append(project_sum(r))

    # Perturbations of seed
    for _ in range(4):
        p = np.full(N, 0.5) + rng.normal(0, 0.2, N)
        inits.append(project_sum(p))

    # Run SLSQP on each, with adaptive tau (coarser early, finer later)
    for idx, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
        tau_c = 1.0 if idx < 8 else 0.5
        tau_f = 0.15 if idx < 8 else 0.05
        cand = try_optimize(init, tau_c, tau_f)

        # Occasionally re-optimize the best with a finer tau
        if idx % 4 == 0 and time.time() < deadline - 15:
            try_optimize(best_h, tau_c=0.3, tau_f=0.03)

    # Final fine polish on the best
    if time.time() < deadline - 5:
        try_optimize(best_h, tau_c=0.2, tau_f=0.02)

    best_h = project_sum(best_h)
    # Final validity check
    if not (np.all(best_h >= 0.0) and np.all(best_h <= 1.0)):
        best_h = np.clip(best_h, 0.0, 1.0)
        best_h = project_sum(best_h)
    if not np.isfinite(_value(best_h)):
        best_h = np.full(N, 0.5)

    return best_h.tolist()
