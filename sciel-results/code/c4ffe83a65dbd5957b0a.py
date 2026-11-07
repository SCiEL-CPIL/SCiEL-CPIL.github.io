"""Rewritten optimizer: multi-start Nelder-Mead with adaptive restarts."""
import time
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead (derivative-free) on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        deadline = time.time() + 300.0 - 30.0  # 30s safety margin

        # call accounting
        state = {"calls": 0, "best_x": None, "best_f": np.inf}

        def f(x):
            if state["calls"] >= total_budget:
                return state["best_f"]
            try:
                v = float(func(np.asarray(x, dtype=float)))
            except Exception:
                return state["best_f"]
            state["calls"] += 1
            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = np.array(x, dtype=float)
            return v

        def nelder_mead(x0, fx, max_iter, sigma):
            # build simplex
            s = [np.array(x0, dtype=float)]
            vals = [fx]
            for i in range(n):
                p = np.array(x0, dtype=float)
                p[i] = p[i] + sigma
                p = np.clip(p, -5.0, 5.0)
                fp = f(p)
                s.append(p)
                vals.append(fp)
            # if any initial point coincides with x0, nudge
            for i in range(1, n + 1):
                if np.allclose(s[i], s[0]):
                    s[i] = np.clip(s[i] + sigma * 0.1 * (i + 1), -5.0, 5.0)
                    vals[i] = f(s[i])

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if state["calls"] >= total_budget:
                    break
                # sort
                order = np.argsort(vals)
                s = [s[i] for i in order]
                vals = [vals[i] for i in order]

                centroid = np.mean(s[:-1], axis=0)
                worst = s[-1]
                fw = vals[-1]

                # reflection
                xr = np.clip(centroid + alpha * (centroid - worst), -5.0, 5.0)
                fr = f(xr)

                if fr < vals[0]:
                    # expansion
                    xe = np.clip(centroid + gamma * (centroid - worst), -5.0, 5.0)
                    fe = f(xe)
                    if fe < fr:
                        s[-1], vals[-1] = xe, fe
                    else:
                        s[-1], vals[-1] = xr, fr
                elif fr < vals[-2]:
                    s[-1], vals[-1] = xr, fr
                else:
                    # contraction
                    if fr < fw:
                        xc = np.clip(centroid + rho * (worst - centroid), -5.0, 5.0)
                    else:
                        xc = np.clip(centroid - rho * (centroid - worst), -5.0, 5.0)
                    fc = f(xc)
                    if fc < min(fr, fw):
                        s[-1], vals[-1] = xc, fc
                    else:
                        # shrink
                        base = s[0]
                        for i in range(1, n + 1):
                            s[i] = np.clip(base + sigma_shrink * (s[i] - base), -5.0, 5.0)
                            vals[i] = f(s[i])

                # convergence check
                spread = max(vals) - min(vals)
                if spread < 1e-10:
                    break

            # return best of this run
            best_idx = int(np.argmin(vals))
            return s[best_idx], vals[best_idx]

        rng = np.random.default_rng(12345)

        # multi-start: first a coarse global probe, then local NM runs
        # We'll budget a portion for global probing and the rest for NM.
        # Strategy: run a sequence of NM starts; each start gets a cap on
        # evaluations. Restart if a start yields little improvement.

        # Initial: spread a few probe points to get a decent start
        # Use a small random sample to pick the best starting region.
        probe_count = min(20, total_budget // 10)
        probe_pts = rng.uniform(-4.0, 4.0, size=(probe_count, n))
        probe_vals = [f(p) for p in probe_pts]
        # pick best probe as first NM start
        best_probe = int(np.argmin(probe_vals))

        # NM start with reasonable initial simplex size
        sigma_init = 1.5
        iters_per_run = 60  # each NM run uses ~ (n+1) + iters*(1..3) evals

        # We'll do up to K restarts; stop when budget or deadline runs out.
        max_restarts = 8
        last_best = state["best_f"]

        for r in range(max_restarts):
            if state["calls"] >= total_budget:
                break
            if time.time() > deadline:
                break

            if r == 0:
                x0 = np.array(probe_pts[best_probe], dtype=float)
                fx0 = probe_vals[best_probe]
            else:
                # new random start, biased slightly toward current best region
                x0 = rng.uniform(-4.5, 4.5, size=n)
                # occasionally perturb the best found
                if state["best_x"] is not None and rng.random() < 0.3:
                    x0 = np.clip(state["best_x"] + rng.standard_normal(n) * 1.0, -5.0, 5.0)
                fx0 = f(x0)

            x_best_run, f_best_run = nelder_mead(x0, fx0, iters_per_run, sigma_init)

            # if no improvement over last_best, we could reduce sigma next time
            # (adaptive)
            if f_best_run >= last_best:
                sigma_init *= 0.7
            last_best = state["best_f"]

            # early stop if we're very close to a plateau
            if state["best_f"] < -1e3 or (state["best_f"] < last_best and (last_best - state["best_f"]) < 1e-6):
                # keep going a bit; not a hard stop
                pass

        return state["best_x"], state["best_f"]
