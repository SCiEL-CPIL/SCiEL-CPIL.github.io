"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 590.0  # safety margin before the 630 s wall-clock kill

    def remaining():
        return deadline - (time.time() - start_time)

    # ------------------------------------------------------------------
    # Objective: weighted L2 energy of Q(x) = prod_i (x - r_i)^2
    # under the Laguerre weight w(x) = x * exp(-x)  (alpha = 1).
    # The Cohn-Goncalves Fourier sign uncertainty upper bound C4 is
    # controlled by this energy (lower energy -> lower bound), so we
    # minimise it.  Computed with Gauss-Laguerre quadrature (weight
    # exp(-x)); we multiply in the extra factor x for alpha = 1.
    # ------------------------------------------------------------------
    # Precompute quadrature nodes/weights once (weight exp(-x)).
    try:
        from numpy.polynomial.laguerre import laggauss
        xq, wq = laggauss(250)
        xq = np.asarray(xq, dtype=float)
        wq = np.asarray(wq, dtype=float)
    except Exception:
        xq = np.linspace(0.0, 200.0, 4000)
        wq = np.exp(-xq) * (200.0 / (4000 - 1))

    # Precompute log weights for the x*exp(-x) weight
    # xq[0] is very small but positive in laggauss; handle log carefully
    log_xw = np.full_like(xq, -700.0)
    mask = xq > 1e-12
    log_xw[mask] = np.log(xq[mask]) + np.log(wq[mask])

    def energy(roots):
        r = np.asarray(roots, dtype=float)
        if np.any(r <= 0.0):
            return 1e30
        # Q(x) = prod (x - r_i)^2 ; log Q = 2 * sum_i log|x - r_i|
        # We need integral of Q(x)^2 * x * exp(-x) dx
        # log(Q^2 * x * exp(-x)) = 4*sum log|x-ri| + log(x) + log(wq)
        diffs = np.abs(xq[None, :] - r[:, None])
        diffs = np.maximum(diffs, 1e-15)
        log_Q_sq = 4.0 * np.sum(np.log(diffs), axis=0)
        log_integrand = log_Q_sq + log_xw
        log_integrand = np.clip(log_integrand, -700.0, 700.0)
        val = np.sum(np.exp(log_integrand))
        if not np.isfinite(val):
            return 1e30
        return float(val)

    # ------------------------------------------------------------------
    # Candidate starting points - diverse initializations
    # ------------------------------------------------------------------
    candidates = []

    # 1. Seed (equally spaced 4,8,...,48)
    candidates.append(np.array([4.0 * (i + 1) for i in range(12)], dtype=float))

    # 2. Roots of L_12^(1) (generalized Laguerre, alpha=1)
    try:
        from scipy.special import roots_genlaguerre
        r_lag1, _ = roots_genlaguerre(12, 1)
        r_lag1 = np.sort(np.asarray(r_lag1, dtype=float))
        candidates.append(r_lag1)
        # Scaled variants
        for s in [0.5, 0.7, 1.0, 1.3, 2.0]:
            candidates.append(r_lag1 * s)
    except Exception:
        pass

    # 3. Roots of L_12^(0) (standard Laguerre)
    try:
        from scipy.special import roots_genlaguerre
        r_lag0, _ = roots_genlaguerre(12, 0)
        r_lag0 = np.sort(np.asarray(r_lag0, dtype=float))
        candidates.append(r_lag0)
        for s in [0.5, 1.0, 2.0]:
            candidates.append(r_lag0 * s)
    except Exception:
        pass

    # 4. Geometric progression
    geo = 4.0 * np.power(12.0, np.arange(12) / 11.0)
    candidates.append(geo)

    # 5. Asymptotic distribution: Laguerre zeros ~ (4n+2alpha+2)*sin^2((k+1/4)*pi/(2n+alpha+1))
    # For alpha=1, n=12: use this as another starting point
    n = 12
    alpha = 1
    k = np.arange(n)
    asymptotic = (4*n + 2*alpha + 2) * np.sin((k + 0.25) * np.pi / (2*n + alpha + 1))**2
    asymptotic = np.sort(asymptotic)
    candidates.append(asymptotic)
    for s in [0.5, 1.0, 2.0]:
        candidates.append(asymptotic * s)

    # 6. Random perturbations of seed
    np.random.seed(42)
    for trial in range(5):
        perturbed = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
        perturbed += np.random.normal(0, 1.0, 12)
        perturbed = np.sort(np.maximum(perturbed, 0.1))
        candidates.append(perturbed)

    # ------------------------------------------------------------------
    # Pick the best starting candidate by energy
    # ------------------------------------------------------------------
    best_val = float("inf")
    best_roots = candidates[0].copy()
    for c in candidates:
        if remaining() < 5.0:
            break
        v = energy(c)
        if v < best_val:
            best_val = v
            best_roots = c.copy()

    # ------------------------------------------------------------------
    # Local refinement with multiple optimization methods and starts
    # ------------------------------------------------------------------
    try:
        from scipy.optimize import minimize

        def objective(roots_flat):
            r = np.asarray(roots_flat, dtype=float)
            if np.any(r <= 1e-6):
                return 1e30
            return energy(r)

        # Refine from top candidates
        # Sort candidates by energy and take top few
        cand_energies = []
        for c in candidates:
            if remaining() < 10.0:
                break
            e = energy(c)
            cand_energies.append((e, c))
        cand_energies.sort(key=lambda x: x[0])

        # Try Nelder-Mead from best 5 distinct starting points
        nm_starts = []
        seen = set()
        for e, c in cand_energies:
            key = tuple(np.round(c, 2))
            if key not in seen:
                nm_starts.append(c.copy())
                seen.add(key)
            if len(nm_starts) >= 5:
                break

        for sp in nm_starts:
            if remaining() < 20.0:
                break
            try:
                res = minimize(
                    objective,
                    sp,
                    method="Nelder-Mead",
                    options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10,
                             "adaptive": True, "initial_simplex": None},
                )
                if res.fun < best_val and np.all(res.x > 0.0):
                    best_val = float(res.fun)
                    best_roots = np.sort(res.x).copy()
            except Exception:
                pass

        # Also try L-BFGS-B with bounds for a few starts
        bounds = [(1e-4, None)] * 12
        for sp in nm_starts[:3]:
            if remaining() < 15.0:
                break
            try:
                res = minimize(
                    objective,
                    sp,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
                )
                if res.fun < best_val and np.all(res.x > 0.0):
                    best_val = float(res.fun)
                    best_roots = np.sort(res.x).copy()
            except Exception:
                pass

    except Exception:
        pass

    # ------------------------------------------------------------------
    # Final polish: coordinate descent on sorted roots
    # ------------------------------------------------------------------
    if remaining() > 30.0:
        out = np.sort(np.asarray(best_roots, dtype=float))
        for iteration in range(50):
            if remaining() < 10.0:
                break
            improved = False
            for i in range(12):
                if remaining() < 5.0:
                    break
                # Try small perturbations
                base = out.copy()
                best_local = energy(base)
                for delta in [-0.5, -0.1, -0.01, 0.01, 0.1, 0.5]:
                    trial = base.copy()
                    trial[i] = max(1e-4, base[i] + delta)
                    # Ensure sorted
                    trial = np.sort(trial)
                    v = energy(trial)
                    if v < best_local - 1e-12:
                        best_local = v
                        out = trial.copy()
                        improved = True
            if not improved:
                break
        best_roots = out

    # ------------------------------------------------------------------
    # Final validity: 12 positive, strictly increasing double roots
    # ------------------------------------------------------------------
    out = np.sort(np.asarray(best_roots, dtype=float))
    out = np.maximum(out, 1e-6)
    # ensure strict increase
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1e-6

    return [float(v) for v in out]
# EVOLVE-BLOCK-END
