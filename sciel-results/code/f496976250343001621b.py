"""Bermudan put exercise policy via analytic GBM backward induction (direct)."""
import time
import math
import numpy as np

_DEADLINE_LIMIT = 140.0  # wall-clock budget (evaluator kills at 150s)


def _make_policy(thresholds):
    class ExercisePolicy:
        def __init__(self, thresholds):
            self.thresholds = thresholds
            self._n = len(thresholds)

        def exercise(self, current_state, time_idx):
            try:
                ti = int(time_idx)
            except Exception:
                return np.zeros(np.asarray(current_state).shape, dtype=bool)
            if ti < 0 or ti >= self._n:
                return np.zeros(np.asarray(current_state).shape, dtype=bool)
            th = float(self.thresholds[ti])
            return np.asarray(current_state, dtype=np.float64) < th
    return ExercisePolicy(thresholds)


def _simple_thresholds(K, n_dates, T, sigma):
    # Fallback heuristic thresholds: increase toward K as maturity approaches.
    # At last date (i=n-1): threshold ~ K. At first date: lower.
    dt = T / n_dates
    thr = np.zeros(n_dates)
    for i in range(n_dates):
        rem = (n_dates - i) * dt  # time from this date to maturity
        # threshold = K * (1 - exp(-2.5 * rem / max(dt, 1e-9)))
        thr[i] = K * (1.0 - math.exp(-2.5 * rem / max(dt, 1e-9)))
    thr[-1] = K  # at maturity always exercise in-the-money
    return thr


def fit_exercise_policy(training_paths, contract):
    start = time.time()

    K = float(contract.get('K', contract.get('strike', 100.0)))
    r = float(contract.get('r', contract.get('rate', 0.05)))
    sigma = float(contract.get('sigma', contract.get('vol', 0.2)))
    T = float(contract.get('T', contract.get('maturity', 1.0)))
    n_dates = int(contract.get('n_dates', contract.get('n_exercise', 12)))
    if n_dates <= 0:
        n_dates = 12

    dt = T / n_dates
    drift = (r - 0.5 * sigma * sigma) * dt
    vol_step = sigma * math.sqrt(dt)
    disc = math.exp(-r * dt)

    # Grid
    n_grid = 800
    S_min = max(0.005 * K, 1e-6)
    S_max = 2.5 * K
    S_grid = np.linspace(S_min, S_max, n_grid)

    # Gauss-Hermite quadrature (standard normal)
    n_quad = 96
    zq, wq = np.polynomial.hermite.hermgauss(n_quad)
    zq = zq * math.sqrt(2.0)
    wq = wq / math.sqrt(math.pi)

    # log(S) on grid for fast evaluation
    logS_grid = np.log(S_grid)

    # V_next as function of log-S; represent on S_grid initially as terminal payoff
    # We'll store V on S_grid and interpolate in log-space.
    # log-space grid for interpolation
    logS = logS_grid

    # Terminal: V_n(S) = max(K - S, 0)
    V_next = np.maximum(K - S_grid, 0.0)

    thresholds = np.zeros(n_dates)

    def cont_value(V_next, S_grid, logS_grid):
        """Compute e^{-r dt} * E[V_next(S * exp(drift + vol_step*Z)) | S]
        for each S in S_grid, using Gauss-Hermite quadrature."""
        # For each S_i, log(S_{i+1}) = log(S_i) + drift + vol_step * Z
        # log_S_next shape (n_grid, n_quad)
        log_S_next = logS_grid[:, None] + drift + vol_step * zq[None, :]
        # Clip to grid log range for interpolation
        log_S_next = np.clip(log_S_next, logS_grid[0], logS_grid[-1])
        V_interp = np.interp(log_S_next, logS_grid, V_next)  # (n_grid, n_quad)
        cont = disc * np.sum(wq[None, :] * V_interp, axis=1)
        return cont

    for t_idx in range(n_dates - 1, -1, -1):
        cont = cont_value(V_next, S_grid, logS_grid)
        immediate = np.maximum(K - S_grid, 0.0)
        V_cur = np.maximum(immediate, cont)

        # Find threshold: largest S where immediate >= cont
        diff = immediate - cont
        # Only consider S < K (for S >= K, immediate = 0, never exercise unless cont < 0 which can't happen)
        valid = S_grid < K
        if np.any(valid):
            diff_v = diff[valid]
            S_v = S_grid[valid]
            pos_idx = np.where(diff_v >= 0)[0]
            if len(pos_idx) > 0:
                last = pos_idx[-1]
                if last + 1 < len(S_v):
                    d1 = diff_v[last]
                    d2 = diff_v[last + 1]
                    if d2 < 0 and d1 >= 0:
                        # Quadratic interpolation for finer threshold
                        if last >= 1:
                            d0 = diff_v[last - 1]
                            s0 = S_v[last - 1]
                            s1 = S_v[last]
                            s2 = S_v[last + 1]
                            # Lagrange interpolation of diff=0
                            # Solve quadratic: a*s^2 + b*s + c = 0
                            a = (d0 / ((s0 - s1) * (s0 - s2)) +
                                 d1 / ((s1 - s0) * (s1 - s2)) +
                                 d2 / ((s2 - s0) * (s2 - s1)))
                            b = (-d0 * (s0 + s1 + s2) / ((s0 - s1) * (s0 - s2)) +
                                 -d1 * (s0 + s1 + s2) / ((s1 - s0) * (s1 - s2)) +
                                 -d2 * (s0 + s1 + s2) / ((s2 - s0) * (s2 - s1)))
                            c = (d0 * s1 * s2 / ((s0 - s1) * (s0 - s2)) +
                                 d1 * s0 * s2 / ((s1 - s0) * (s1 - s2)) +
                                 d2 * s0 * s1 / ((s2 - s0) * (s2 - s1)))
                            if abs(a) > 1e-15:
                                disc_q = b * b - 4 * a * c
                                if disc_q >= 0:
                                    sqrt_disc = math.sqrt(disc_q)
                                    root1 = (-b + sqrt_disc) / (2 * a)
                                    root2 = (-b - sqrt_disc) / (2 * a)
                                    # Pick the root in [s1, s2]
                                    if s1 <= root1 <= s2:
                                        thr = root1
                                    elif s1 <= root2 <= s2:
                                        thr = root2
                                    else:
                                        frac = d1 / (d1 - d2)
                                        thr = s1 + frac * (s2 - s1)
                                else:
                                    frac = d1 / (d1 - d2)
                                    thr = s1 + frac * (s2 - s1)
                            else:
                                frac = d1 / (d1 - d2)
                                thr = s1 + frac * (s2 - s1)
                        else:
                            frac = d1 / (d1 - d2)
                            thr = S_v[last] + frac * (S_v[last + 1] - S_v[last])
                    else:
                        thr = S_v[last]
                else:
                    thr = S_v[last]
                # Cap at K
                if thr > K:
                    thr = K
                if thr < 0:
                    thr = 0.0
            else:
                thr = 0.0
        else:
            thr = 0.0
        thresholds[t_idx] = thr
        V_next = V_cur

        # Deadline guard
        if time.time() - start > _DEADLINE_LIMIT - 8.0:
            # Fill remaining thresholds with simple heuristic
            for j in range(t_idx - 1, -1, -1):
                rem = (n_dates - j) * dt
                thresholds[j] = K * (1.0 - math.exp(-2.5 * rem / max(dt, 1e-9)))
            break

    return _make_policy(thresholds)
