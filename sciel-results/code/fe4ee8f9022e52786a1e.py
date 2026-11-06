import numpy as np
import time
import math


# EVOLVE-BLOCK-START
_DEADLINE_LIMIT = 135.0
_SAFETY = 10.0


def _make_policy(thresholds, n_steps, dt, T):
    class ExercisePolicy:
        def __init__(self, thresholds, n_steps, dt, T):
            self.thresholds = np.asarray(thresholds, dtype=np.float64)
            self.n_steps = n_steps
            self.dt = dt
            self.T = T

        def exercise(self, current_state, time):
            cs = np.asarray(current_state, dtype=np.float64)
            if time is None:
                t = 0.0
            else:
                try:
                    t = float(time)
                except Exception:
                    t = 0.0
            if self.dt > 0:
                j = int(round(t / self.dt))
            else:
                j = 1
            j = max(1, min(j, self.n_steps))
            threshold = self.thresholds[j - 1]
            return cs <= threshold

    return ExercisePolicy(thresholds, n_steps, dt, T)


def _simple_thresholds(K, r, sigma, T, n_steps):
    """Fallback: time-decaying thresholds."""
    dt = T / n_steps
    th = np.empty(n_steps)
    for j in range(n_steps):
        rem = (n_steps - j) * dt
        th[j] = K * (1.0 - math.exp(-2.5 * rem / max(dt, 1e-9)))
    th[-1] = K
    return th


def _continuation_gh(V_next, logS_grid, drift, vol_step, disc, zq, wq):
    """Exact risk-neutral continuation via Gauss-Hermite in log-space."""
    logS_next = logS_grid[:, None] + drift + vol_step * zq[None, :]
    logS_next = np.clip(logS_next, logS_grid[0], logS_grid[-1])
    V_interp = np.interp(logS_next.ravel(), logS_grid, V_next).reshape(logS_next.shape)
    cont = disc * np.sum(wq[None, :] * V_interp, axis=1)
    return cont


def _threshold_bisect(immediate, cont, S_grid, lo_idx, hi_idx):
    """Bisect for threshold where immediate(s) = cont(s)."""
    a = S_grid[lo_idx]
    b = S_grid[hi_idx]
    fa = immediate[lo_idx] - cont[lo_idx]
    fb = immediate[hi_idx] - cont[hi_idx]
    if fa < 0:
        return a
    if fb > 0:
        return b
    for _ in range(60):
        mid = 0.5 * (a + b)
        im = np.interp(mid, S_grid, immediate)
        co = np.interp(mid, S_grid, cont)
        fm = im - co
        if fm > 0:
            a = mid
        else:
            b = mid
        if b - a < 1e-7:
            break
    return 0.5 * (a + b)


def fit_exercise_policy(training_paths, contract):
    t_start = time.time()

    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12

    def _get(obj, *names, default=None):
        for nm in names:
            if isinstance(obj, dict):
                if nm in obj:
                    return obj[nm]
            elif hasattr(obj, nm):
                return getattr(obj, nm)
        return default

    try:
        K = float(_get(contract, 'K', 'strike', 'Strike', default=K))
        r = float(_get(contract, 'r', 'rate', 'Rate', 'risk_free', default=r))
        sigma = float(_get(contract, 'sigma', 'vol', 'volatility', default=sigma))
        T = float(_get(contract, 'T', 'maturity', 'Maturity', default=T))
        n_steps = int(_get(contract, 'n_dates', 'num_dates', 'exercise_dates', default=n_steps))
    except Exception:
        pass

    if n_steps < 1:
        n_steps = 1
    dt = T / n_steps if n_steps > 0 else T
    drift = (r - 0.5 * sigma * sigma) * dt
    vol_step = sigma * math.sqrt(dt)
    disc = math.exp(-r * dt)

    # Grid: log-spaced, wide range
    S_min = max(0.005 * K, 1e-4)
    S_max = 2.5 * K
    n_grid = 800
    S_grid = np.geomspace(S_min, S_max, n_grid)
    logS_grid = np.log(S_grid)
    immediate = np.maximum(K - S_grid, 0.0)

    # Gauss-Hermite quadrature
    n_quad = 80
    zq, wq = np.polynomial.hermite.hermgauss(n_quad)
    zq = zq * math.sqrt(2.0)
    wq = wq / math.sqrt(math.pi)

    def elapsed_ok():
        return (time.time() - t_start) < (_DEADLINE_LIMIT - _SAFETY)

    # Backward induction to get initial thresholds
    V_next = immediate.copy()
    thresholds = np.zeros(n_steps)

    for t_idx in range(n_steps - 1, -1, -1):
        if not elapsed_ok():
            break
        cont = _continuation_gh(V_next, logS_grid, drift, vol_step, disc, zq, wq)
        V_cur = np.maximum(immediate, cont)

        diff = immediate - cont
        valid = S_grid < K
        if np.any(valid):
            idx_valid = np.where(valid)[0]
            diff_v = diff[idx_valid]
            pos_idx = np.where(diff_v >= 0)[0]
            if len(pos_idx) > 0:
                last = pos_idx[-1]
                if last + 1 < len(idx_valid):
                    lo_idx = idx_valid[last]
                    hi_idx = idx_valid[last + 1]
                    thr = _threshold_bisect(immediate, cont, S_grid, lo_idx, hi_idx)
                else:
                    thr = S_grid[idx_valid[last]]
                thr = min(max(thr, 0.0), K)
            else:
                thr = 0.0
        else:
            thr = 0.0
        thresholds[t_idx] = thr
        V_next = V_cur

    # Monotonicity: earlier dates should have higher (or equal) thresholds
    for j in range(1, n_steps):
        if thresholds[j] > thresholds[j - 1]:
            thresholds[j] = thresholds[j - 1]

    # === STRUCTURAL CHANGE: Monte Carlo policy evaluation + coordinate ascent ===
    # Use the training paths to evaluate the actual expected discounted payoff
    # of a given threshold policy, then do coordinate ascent to improve.
    # This directly optimizes the objective rather than relying on grid approximation.

    if elapsed_ok() and training_paths is not None:
        try:
            tp = np.asarray(training_paths, dtype=np.float64)
        except Exception:
            tp = None

        if tp is not None and tp.ndim == 2:
            n_paths, n_times = tp.shape
            # n_times should be n_steps + 1 (time 0 through maturity) or n_steps
            # Exercise dates are at t=dt, 2dt, ..., n_steps*dt (no time-0)
            # So we need states at times dt, 2dt, ..., T
            # If tp has shape (n_paths, n_steps+1), columns 1..n_steps are exercise dates
            # If tp has shape (n_paths, n_steps), columns 0..n_steps-1 are exercise dates

            if n_times >= n_steps + 1:
                # Use columns 1..n_steps (skip time 0)
                exercise_states = tp[:, 1:n_steps + 1]  # shape (n_paths, n_steps)
            elif n_times >= n_steps:
                exercise_states = tp[:, :n_steps]
            else:
                exercise_states = None
        else:
            exercise_states = None

        if exercise_states is not None and elapsed_ok():
            n_p, n_d = exercise_states.shape
            # n_d should be n_steps (12 exercise dates)

            def _evaluate_policy(thr):
                """Compute expected discounted payoff for given thresholds.
                A path exercises at the first date j where S_j <= thr[j].
                Payoff = K - S_j, discounted to time 0.
                If never exercised, payoff = 0 (European put at maturity is not exercised
                unless in the money, but Bermudan gives option to exercise at each date).
                Actually for a put, if never exercised before maturity, we get the
                maturity payoff if ITM. But the last date IS maturity, so if S_12 <= thr[11],
                we exercise. If thr[11] >= K, we always exercise at maturity when ITM.
                """
                payoff = np.zeros(n_p)
                for j in range(n_d):
                    mask = exercise_states[:, j] <= thr[j]
                    # Only paths that haven't been exercised yet
                    mask = mask & (payoff == 0)
                    if np.any(mask):
                        disc_j = math.exp(-r * (j + 1) * dt)
                        payoff[mask] = disc_j * (K - exercise_states[mask, j])
                return np.mean(payoff)

            # Coordinate ascent
            best_val = _evaluate_policy(thresholds)
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]

            for outer in range(3):
                if not elapsed_ok():
                    break
                for j in range(n_steps):
                    if not elapsed_ok():
                        break
                    for ss in step_sizes:
                        if not elapsed_ok():
                            break
                        # Try nudging threshold up
                        new_th = thresholds.copy()
                        new_th[j] = min(thresholds[j] + ss, K)
                        val_up = _evaluate_policy(new_th)
                        if val_up > best_val + 1e-12:
                            best_val = val_up
                            thresholds[j] = new_th[j]

                        # Try nudging threshold down
                        new_th = thresholds.copy()
                        new_th[j] = max(thresholds[j] - ss, 0.0)
                        val_dn = _evaluate_policy(new_th)
                        if val_dn > best_val + 1e-12:
                            best_val = val_dn
                            thresholds[j] = new_th[j]

            # Re-enforce monotonicity after MC optimization
            for j in range(1, n_steps):
                if thresholds[j] > thresholds[j - 1]:
                    thresholds[j] = thresholds[j - 1]

    # Final safety
    thresholds = np.clip(thresholds, 0.0, K)
    thresholds[-1] = max(thresholds[-1], K * 0.99)

    return _make_policy(thresholds, n_steps, dt, T)
# EVOLVE-BLOCK-END
