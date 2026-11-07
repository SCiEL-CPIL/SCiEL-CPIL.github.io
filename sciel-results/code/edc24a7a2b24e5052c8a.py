"""Erdos minimum overlap - combined control-point + soft-max + epigraph.

Structural idea: represent the 100 step heights as a piecewise-linear function
of K control points (interpolated to N samples, then sum-projected). This
low-dimensional smooth parameterization lets the optimizer explore global
shapes (bimodal, oscillatory) far more effectively than 100 independent vars.

Refinement machinery:
  Phase 1  global search (differential evolution) over control points
  Phase 2  ramping-lambda log-sum-exp soft-max surrogate + L-BFGS-B multi-start
  Phase 3  exact (nonsmooth) SLSQP on control points
  Phase 4  epigraph formulation: auxiliary t + smooth max-of-affine constraint
  Phase 5  pairwise-swap local polish (provably preserves sum)
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
K = 10
DEADLINE = 142.0  # safety margin before 150s kill


# ---------------------------------------------------------------------------
# Core evaluation / feasibility
# ---------------------------------------------------------------------------
def _correlation_value(h):
    """The evaluator's value: max(corr)/N*2. Lower is better."""
    corr = np.correlate(h, 1 - h, mode="full")
    return float(np.max(corr)) / N * 2.0


def _correlation_array(h):
    return np.correlate(h, 1 - h, mode="full")


def _interp_from_control(c):
    """Linearly interpolate K control points (bin centers) to N samples."""
    c = np.asarray(c, dtype=float)
    cx = (np.arange(K) + 0.5) * (N / K)
    x = np.arange(N, dtype=float)
    return np.interp(x, cx, c)


def _enforce_sum(h, target=None):
    """Offset + clip + iterative re-correction to hit sum==target within tol."""
    if target is None:
        target = N / 2.0
    h = np.array(h, dtype=float)
    diff = target - np.sum(h)
    h = h + diff / N
    for _ in range(80):
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


def _feasible_from_control(c, target=None):
    h = _interp_from_control(c)
    return _enforce_sum(h, target)


def _feasible_from_h(h, target=None):
    return _enforce_sum(h, target)


# ---------------------------------------------------------------------------
# Smooth surrogates
# ---------------------------------------------------------------------------
def _soft_max(corr, lam):
    """Numerically stable log-sum-exp soft-max."""
    m = np.max(corr)
    if lam <= 0:
        return m
    shifted = corr - m
    return m + np.log(np.sum(np.exp(lam * shifted)) / len(corr)) / lam


def _objective_soft(c, lam):
    h = _feasible_from_control(c)
    corr = _correlation_array(h)
    return _soft_max(corr, lam)


def _objective_exact(c):
    h = _feasible_from_control(c)
    return _correlation_value(h)


# ---------------------------------------------------------------------------
# Epigraph formulation: min t  s.t.  corr(h) <= t,  0<=h<=1,  sum(h)=N/2
# We use a smooth max-of-affine surrogate for the max constraint so a
# gradient method (SLSQP) can drive t down toward the true max.
# ---------------------------------------------------------------------------
def _epigraph_solve(h0, lam, maxiter=400):
    h0 = np.array(h0, dtype=float)
    corr0 = _correlation_array(h0)
    t0 = float(np.max(corr0)) + 1e-4

    def epig(z):
        h = z[:N]
        t = z[N]
        corr = np.correlate(h, 1 - h, mode="full")
        m = np.max(corr)
        # smooth max-of-affine: soft_max(corr - t, lam)
        shifted = (corr - t)
        smax = np.log(np.sum(np.exp(lam * shifted)) / len(corr)) / lam
        # objective: t + penalty for the (smooth) constraint violation
        return t + 10.0 * np.log1p(np.exp(50.0 * smax))

    def epig_jac(z):
        h = z[:N]
        t = z[N]
        corr = np.correlate(h, 1 - h, mode="full")
        m = np.max(corr)
        shifted = corr - t
        w = np.exp(lam * shifted)
        w = w / np.sum(w)
        # d smax / d corr_i = w_i ; d smax/dt = -1
        dcorr = -2.0 * np.correlate(h, 1 - h, mode="full")  # placeholder (unused)
        # Build gradient numerically-ish: use finite difference for robustness
        grad = np.zeros(N + 1)
        # numerical gradient (cheap enough for a final polish)
        eps = 1e-5
        f0 = epig(z)
        for i in range(N + 1):
            zz = z.copy()
            zz[i] += eps
            grad[i] = (epig(zz) - f0) / eps
        return grad

    bounds = [(0.0, 1.0)] * N + [(0.0, float(np.max(corr0)) + 0.5)]
    cons = [{'type': 'eq', 'fun': lambda z: np.sum(z[:N]) - N / 2.0,
             'jac': lambda z: np.concatenate([np.ones(N), [0.0]])}]
    try:
        res = minimize(epig, np.concatenate([h0, [t0]]), method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': 1e-14})
        h_opt = _feasible_from_h(res.x[:N])
        return h_opt
    except Exception:
        return h0


# ---------------------------------------------------------------------------
# Local polish: pairwise swaps preserve sum exactly.
# ---------------------------------------------------------------------------
def _local_polish_h(h, max_iters=25, step_sizes=None):
    best_h = np.array(h, dtype=float)
    best_val = _correlation_value(best_h)
    if step_sizes is None:
        step_sizes = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    n = len(best_h)
    for _ in range(max_iters):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                for d in step_sizes:
                    for sign in (+1, -1):
                        ni = best_h[i] + sign * d
                        nj = best_h[j] - sign * d
                        if ni < 0.0 or ni > 1.0 or nj < 0.0 or nj > 1.0:
                            continue
                        new_h = best_h.copy()
                        new_h[i] = ni
                        new_h[j] = nj
                        val = _correlation_value(new_h)
                        if val < best_val - 1e-13:
                            best_val = val
                            best_h = new_h
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best_h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()

    def _remaining():
        return DEADLINE - (time.time() - start_time)

    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)  # seed value

    # ---------------- Phase 1: differential evolution over control points ----
    if _remaining() > 40.0:
        de_deadline = start_time + min(40.0, _remaining() - 5.0)
        try:
            c_lo = np.full(K, 0.0)
            c_hi = np.full(K, 1.0)

            def de_obj(c):
                if time.time() > de_deadline:
                    raise TimeoutError()
                return _objective_soft(c, lam=25.0)

            res_de = differential_evolution(
                de_obj,
                bounds=list(zip(c_lo, c_hi)),
                maxiter=60, popsize=12, tol=1e-7,
                mutation=(0.4, 1.0), recombination=0.7,
                seed=7, polish=False, init='sobol')
            if res_de.x is not None:
                h_de = _feasible_from_control(res_de.x)
                v_de = _correlation_value(h_de)
                if v_de < best_val:
                    best_val = v_de
                    best_h = h_de.copy()
        except TimeoutError:
            pass
        except Exception:
            pass

    if _remaining() < 20.0:
        return _feasible_from_h(best_h)

    # ---------------- Build diverse control-point starts ---------------------
    rng = np.random.default_rng(2024)
    starts_c = []

    # bin averages of current best (smooth init)
    bin_avg = np.array([best_h[i * (N // K):(i + 1) * (N // K)].mean()
                        for i in range(K)])
    starts_c.append(bin_avg.copy())

    for _ in range(10):
        starts_c.append(rng.uniform(0.0, 1.0, K))

    for phase in (0, 1):
        starts_c.append(np.array([0.8 if (i + phase) % 2 == 0 else 0.2
                                  for i in range(K)]))
    c_bi = np.linspace(0.8, 0.2, K)
    starts_c.append(c_bi.copy())
    starts_c.append(c_bi[::-1].copy())

    # sine / oscillatory control shapes
    for freq in (1, 2, 3, 4, 5, 7):
        amp = min(0.4, 0.5 / freq)
        starts_c.append(np.clip(0.5 + amp * np.sin(2 * np.pi * freq * np.arange(K) / K),
                                0.0, 1.0))
    # triangular / sawtooth
    for period in (4, 5, 10):
        tri = np.zeros(K)
        for i in range(K):
            t = (i % period) / period
            tri[i] = 2 * t if t < 0.5 else 2 - 2 * t
        starts_c.append(np.clip(tri, 0.0, 1.0))

    # ---------------- Phase 2: ramping-lambda L-BFGS-B over control points ----
    slsqp_deadline = start_time + max(0.0, min(110.0, _remaining() - 15.0))
    for idx, c_start in enumerate(starts_c):
        if time.time() > slsqp_deadline - 12.0:
            break
        c_start = np.clip(np.asarray(c_start, dtype=float), 0.0, 1.0)
        current_c = c_start.copy()
        for lam in (15.0, 60.0, 300.0):
            if time.time() > slsqp_deadline - 6.0:
                break

            def soft_obj(c, lam=lam):
                return _objective_soft(c, lam)

            try:
                res = minimize(soft_obj, current_c, method='L-BFGS-B',
                               bounds=[(0.0, 1.0)] * K,
                               options={'maxiter': 200, 'ftol': 1e-14})
                c_opt = np.clip(res.x, 0.0, 1.0)
                h_opt = _feasible_from_control(c_opt)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_c = c_opt
            except Exception:
                pass

        # ---------------- Phase 3: exact SLSQP on control points -------------
        if time.time() > slsqp_deadline - 4.0:
            break

        def exact_obj(c):
            return _objective_exact(c)

        try:
            res_exact = minimize(exact_obj, current_c, method='SLSQP',
                                 bounds=[(0.0, 1.0)] * K,
                                 options={'maxiter': 300, 'ftol': 1e-15})
            h_exact = _feasible_from_control(np.clip(res_exact.x, 0.0, 1.0))
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass

    # ---------------- Phase 4: epigraph smooth solve from best ---------------
    if _remaining() > 25.0:
        epig_deadline = start_time + min(60.0, _remaining() - 10.0)
        for lam in (50.0, 200.0):
            if time.time() > epig_deadline:
                break
            try:
                h_ep = _epigraph_solve(best_h, lam, maxiter=300)
                v_ep = _correlation_value(h_ep)
                if v_ep < best_val:
                    best_val = v_ep
                    best_h = h_ep.copy()
            except Exception:
                pass

    # ---------------- Phase 5: pairwise local polish -------------------------
    if _remaining() > 12.0:
        try:
            best_h = _local_polish_h(best_h, max_iters=25)
            best_val = _correlation_value(best_h)
        except Exception:
            pass

    # Final projection + safety
    best_h = _feasible_from_h(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val

    # Guard: ensure feasibility (finite, in range, sum within tolerance)
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    best_h = np.clip(best_h, 0.0, 1.0)
    if abs(np.sum(best_h) - N / 2.0) > 1e-9 * N:
        best_h = _enforce_sum(best_h)

    return best_h
