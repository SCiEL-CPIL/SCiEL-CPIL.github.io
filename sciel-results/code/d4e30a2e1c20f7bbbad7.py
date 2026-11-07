"""Optimizer for BBOB-style f11 (Discus): multi-start CMA-ES ensemble + coordinate-descent polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        d = self.dim
        budget = self.budget
        calls = [0]
        rng = np.random.default_rng(42)

        def wrapped(x):
            calls[0] += 1
            return float(np.asarray(func(x), dtype=float))

        def alive():
            return calls[0] < budget and (time.time() < deadline)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float).copy()

        # ---- Phase 0: diverse initial sampling ----
        candidates = [np.zeros(d)]
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, d))
        for axis in range(d):
            for val in (-3.0, 3.0):
                pt = np.zeros(d)
                pt[axis] = val
                candidates.append(pt)
        for c in candidates:
            if not alive():
                break
            c = np.clip(c, -5, 5)
            update_best(c, wrapped(c))

        # ---- CMA-ES (correct, full implementation) ----
        def cma_es(x0, sigma0):
            nonlocal best_x, best_f
            n = d
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = max(1, lam // 2)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = (2.0 / ((n + 1.8) ** 2 + mueff)) * (1.0 - cc)
            cmu = min(1.0 - c1, (2.0 * (mueff - 2.0 + 1.0 / mueff)) / ((n + 2.0) ** 2 + mueff))

            xmean = np.array(x0, dtype=float).copy()
            sigma = float(sigma0)
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
            gen = 0
            stall = 0

            while alive():
                ys = rng.standard_normal((lam, n))
                xs = np.empty((lam, n))
                for i in range(lam):
                    xs[i] = xmean + sigma * (B @ (D * ys[i]))
                xs = np.clip(xs, -5, 5)
                fs = np.array([wrapped(xs[i]) for i in range(lam)])

                idx = np.argsort(fs)
                xs = xs[idx]
                fs = fs[idx]
                ys = ys[idx]
                update_best(xs[0], fs[0])

                gen += 1
                # convergence / stall detection
                if fs[0] - fs[-1] < 1e-14 * max(1.0, abs(fs[0])):
                    break
                if fs[0] < best_f + 1e-10:
                    stall += 1
                else:
                    stall = 0
                if stall >= 6:
                    break

                # weighted recombination
                y_rec = np.sum(w[:, None] * ys[:mu], axis=0)
                y_rec = y_rec / np.linalg.norm(y_rec)

                # step-size path
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * y_rec
                hsig = (1.4 + 1.0 / n) * chi_n
                denom = np.sqrt(1.0 - (1.0 - cs) ** (2 * gen))
                if np.linalg.norm(ps) / max(denom, 1e-12) < hsig:
                    pc = (1.0 - cc) * pc + cc * y_rec
                    rank_pc = np.outer(pc, pc)
                else:
                    rank_pc = (1.0 + 2.0 * (cs / (1.0 - cs) - 1.0)) * np.outer(pc, pc)

                # covariance update
                rank_mu = np.sum(w[:, None] * np.einsum('ij,ik->ijk', ys[:mu], ys[:mu]), axis=0)
                C = (1.0 + c1 * (cc * (1.0 - cc) / (1.0 - cc)) + c1 - c1) * C + c1 * (rank_pc - C) + cmu * rank_mu
                C = 0.5 * (C + C.T)

                # eigen-decomposition for stability
                try:
                    eigval, eigvec = np.linalg.eigh(C)
                    eigval = np.clip(eigval, 1e-20, None)
                    B = eigvec
                    D = np.sqrt(eigval)
                except np.linalg.LinAlgError:
                    break

                # new mean
                xmean_new = xmean + sigma * (B @ (D * y_rec))
                # sigma update
                sigma *= np.exp(cs / (n + 2.0) * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-12, 10.0)
                xmean = np.clip(xmean_new, -5, 5)

            return xmean

        # ---- Coordinate descent with golden-section line search (polish) ----
        phi = (np.sqrt(5.0) - 1.0) / 2.0

        def line_search(axis, x0, fx0, init_step):
            e = np.zeros(d)
            e[axis] = 1.0

            def g(t):
                pt = np.clip(x0 + t * e, -5, 5)
                return wrapped(pt)

            best_t = 0.0
            best_g = fx0
            grid = np.linspace(-init_step, init_step, 11)
            for t in grid:
                if not alive():
                    break
                gt = g(t)
                if gt < best_g:
                    best_g = gt
                    best_t = t
            if best_t == 0.0:
                return x0, fx0, 0.0
            if best_t > 0:
                a = max(0.0, best_t - init_step * 0.5)
                b = min(init_step, best_t + init_step * 0.5)
            else:
                a = max(-init_step, best_t - init_step * 0.5)
                b = min(0.0, best_t + init_step * 0.5)
            if a >= b:
                a, b = -init_step, init_step
            n_iters = 18 if init_step < 0.1 else 14
            for _ in range(n_iters):
                if not alive():
                    break
                m1 = a + phi * (b - a)
                m2 = b - phi * (b - a)
                f1 = g(m1)
                f2 = g(m2)
                if f1 < f2:
                    b = m2
                else:
                    a = m1
            t_mid = 0.5 * (a + b)
            fm = g(t_mid)
            if fm < best_g:
                best_g = fm
                best_t = t_mid
            pt = np.clip(x0 + best_t * e, -5, 5)
            return pt, best_g, best_t

        def coord_descent(x, fx):
            nonlocal best_x, best_f
            step_scales = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            for scale in step_scales:
                if not alive():
                    break
                improved = False
                for _ in range(2):
                    if not alive():
                        break
                    axis_improved = False
                    for axis in range(d):
                        if not alive():
                            break
                        x, fx, step = line_search(axis, x, fx, scale)
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        if abs(step) > scale * 1e-6:
                            axis_improved = True
                    if not axis_improved:
                        improved = True
                        break
                if not improved:
                    break
            return x, fx

        # ---- Phase 1: broad CMA-ES to find the valley ----
        x0 = best_x.copy() if best_x is not None else np.zeros(d)
        cma_es(x0, 2.5)

        # ---- Phase 2: precise coordinate-descent polish ----
        if alive() and best_x is not None:
            x, fx = coord_descent(best_x.copy(), best_f)
            update_best(x, fx)

        # ---- Phase 3: final small CMA-ES to re-converge on the precise minimum ----
        if alive() and best_x is not None:
            cma_es(best_x.copy(), 0.05)

        return best_x, best_f
# EVOLVE-BLOCK-END
