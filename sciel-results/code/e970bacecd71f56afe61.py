"""Bootstrap portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-strategy bootstrap portfolio:
    1) Latin Hypercube + random warm-start to find a good basin quickly.
    2) (1+1)-ES with adaptive sigma (1/5 success rule).
    3) Periodic restarts from the best point with perturbation.
    4) Deadline guard to stay within wall-clock limits.
    5) Nelder-Mead polish on the final best point for fine-grained convergence.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._start_time = None
        self._deadline_margin = 5.0  # seconds to spare before hard limit

    def __call__(self, func):
        self._start_time = time.time()
        total_budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        def check_deadline():
            return time.time() - self._start_time > 320.0

        # --- Phase 1: Diversified warm-start to find a good initial point ---
        # Latin Hypercube Sampling for broad initial exploration
        n_lhs = min(20, total_budget // 10)
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = -5.0 + 10.0 * (perm + rng.random(n_lhs)) / n_lhs

        candidates = list(lhs_points)
        # Pure random
        for _ in range(5):
            candidates.append(rng.uniform(-5, 5, dim))
        # Center and a few offsets
        candidates.append(np.zeros(dim))
        candidates.append(rng.uniform(-2, 2, dim))

        # Evaluate all candidates (cost: len(candidates) calls)
        best_x = None
        best_f = np.inf
        calls_made = 0
        for c in candidates:
            if check_deadline() or calls_made >= total_budget:
                break
            c = np.clip(c, -5, 5)
            fc = func(c)
            calls_made += 1
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        if best_x is None or calls_made >= total_budget:
            return best_x, best_f

        # --- Phase 2: (1+1)-ES with adaptive sigma and restart logic ---
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        remaining = total_budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        # 1/5 success rule parameters
        sigma_up = 1.34
        sigma_down = 0.8
        success_window = 5
        successes = 0
        window = 0

        # Restart counter
        restart_threshold = 200  # steps without improvement -> restart
        no_improve = 0

        for step in range(remaining):
            # Deadline guard
            if check_deadline():
                break

            # Generate trial
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                successes += 1
                window += 1
                # Adaptive sigma: increase on success
                if window >= success_window:
                    if successes / window > 0.3:
                        sigma *= sigma_up
                    elif successes / window < 0.1:
                        sigma *= sigma_down
                    successes = 0
                    window = 0
            else:
                no_improve += 1
                window += 1
                if window >= success_window:
                    if successes / window < 0.1:
                        sigma *= sigma_down
                    successes = 0
                    window = 0

            # Restart if stuck
            if no_improve >= restart_threshold:
                # Perturb from best known
                x = best_x + 0.5 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                sigma = max(0.1, sigma * 0.5)
                no_improve = 0
                restart_threshold = max(100, restart_threshold // 2)

            # Track global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 3: Nelder-Mead polish around the best point ---
        if best_x is not None and not check_deadline() and calls_made < total_budget:
            nm_budget = min(total_budget - calls_made, 50)
            if nm_budget > dim + 1:
                x_nm, f_nm, calls_nm = self._nm_polish(func, best_x, dim, nm_budget, rng)
                calls_made += calls_nm
                if f_nm < best_f:
                    best_f = f_nm
                    best_x = x_nm

        return best_x, best_f

    def _nm_polish(self, func, x0, dim, budget, rng, lo=-5.0, hi=5.0):
        """Nelder-Mead with box projection; returns (best_x, best_f, calls_used)."""
        x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
        f0 = func(x0)
        calls = 1
        # Initial simplex
        xk = [x0.copy()]
        for j in range(dim):
            xi = x0.copy()
            step = 0.25 if x0[j] < 0 else -0.25
            xi[j] += step
            xi = np.clip(xi, lo, hi)
            xk.append(xi)
        fx = [f0] + [func(xi) for xi in xk[1:]]
        calls += dim

        for _ in range(budget - calls):
            if calls >= budget:
                break
            order = np.argsort(fx)
            xk = [xk[i] for i in order]
            fx = [fx[i] for i in order]
            if calls >= budget:
                break

            # centroid of all but worst
            centroid = np.mean(xk[:-1], axis=0)
            worst = xk[-1]
            # reflection
            xr = np.clip(centroid + 1.0 * (centroid - worst), lo, hi)
            fr = func(xr)
            calls += 1
            if fr < fx[-2]:
                # expansion
                xe = np.clip(centroid + 2.0 * (xr - centroid), lo, hi)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    xk[-1], fx[-1] = xe, fe
                else:
                    xk[-1], fx[-1] = xr, fr
            elif fr <= fx[0]:
                xk[-1], fx[-1] = xr, fr
            else:
                # contraction
                xc = np.clip(centroid + 0.5 * (worst - centroid), lo, hi)
                fc = func(xc)
                calls += 1
                if fc < fx[-1]:
                    xk[-1], fx[-1] = xc, fc
                else:
                    # shrink toward best
                    for i in range(1, dim + 1):
                        xk[i] = np.clip(xk[0] + 0.5 * (xk[i] - xk[0]), lo, hi)
                        fx[i] = func(xk[i])
                        calls += 1
                        if calls >= budget:
                            break
            if calls >= budget:
                break

        order = np.argsort(fx)
        best_i = order[0]
        return xk[best_i], fx[best_i], calls
# EVOLVE-BLOCK-END
