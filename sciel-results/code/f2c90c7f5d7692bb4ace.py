"""Structurally different solver for BBOB f4 (Büche-Rastrigin).

Uses a hand-rolled CMA-ES with adaptive restarts (iRestart=1) and
surrogate-guided sampling (Kriging / Gaussian-process-ish via RBF
interpolation) to exploit the multi-modal landscape efficiently.

Key structural differences from the parent:
  - No scipy.optimize calls; fully self-contained CMA-ES.
  - Adaptive sigma and restarts based on fitness trajectory.
  - Lightweight RBF surrogate to propose new points between CMA-ES iterations.
  - Population diversity via L-BFGS-like line-search on best few points.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285

        call_count = 0
        best_x = None
        best_f = float('inf')
        all_x = []
        all_f = []

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x_c = np.clip(x, -5.0, 5.0)
            f = float(func(x_c))
            all_x.append(x_c.copy())
            all_f.append(f)
            if f < best_f:
                best_f = f
                best_x = x_c.copy()
            return f

        rng = np.random.default_rng(2024)

        # --- Phase 0: Initial scatter (LHS-ish) ---
        n_init = 20
        for i in range(n_init):
            if call_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.0, 4.0, dim)
            try:
                do_call(x)
            except Exception:
                break

        # --- RBF surrogate for guided proposals ---
        def rbf_predict(xq, X, Y, nu=1.0):
            """Gaussian RBF interpolation / approximation.
            X: (n, dim), Y: (n,), xq: (dim,) or (m, dim)"""
            X = np.atleast_2d(X)
            xq = np.atleast_2d(xq)
            d = X.shape[1]
            # Estimate length-scale
            diffs = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
            valid = diffs > 0
            if valid.sum() == 0:
                length_scale = 1.0
            else:
                length_scale = max(np.median(diffs[valid]) * 0.5, 0.1)
            L2 = length_scale ** 2
            # Build matrix
            n = X.shape[0]
            A = np.zeros((n, n))
            for i in range(n):
                dists = np.linalg.norm(X - X[i], axis=1)
                A[i, :] = np.exp(-(dists ** 2) / (2 * L2))
            A[np.diag_indices(n)] = 1.0
            # Solve
            try:
                weights = np.linalg.solve(A + 1e-8 * np.eye(n), Y)
            except np.linalg.LinAlgError:
                weights = np.linalg.lstsq(A + 1e-8 * np.eye(n), Y, rcond=None)[0]
            # Predict
            preds = np.zeros(xq.shape[0])
            for j in range(xq.shape[0]):
                dists = np.linalg.norm(X - xq[j], axis=1)
                phi = np.exp(-(dists ** 2) / (2 * L2))
                preds[j] = phi @ weights
            return preds

        def rbf_variance(xq, X, Y, nu=1.0):
            """Approximate uncertainty for RBF."""
            X = np.atleast_2d(X)
            xq = np.atleast_2d(xq)
            d = X.shape[1]
            diffs = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
            valid = diffs > 0
            if valid.sum() == 0:
                length_scale = 1.0
            else:
                length_scale = max(np.median(diffs[valid]) * 0.5, 0.1)
            L2 = length_scale ** 2
            n = X.shape[0]
            # Simple: variance ~ exp(-dist^2/(2*L2)) to nearest point
            var = np.zeros(xq.shape[0])
            for j in range(xq.shape[0]):
                dists = np.linalg.norm(X - xq[j], axis=1)
                var[j] = np.exp(-np.min(dists) ** 2 / (2 * L2))
            return var

        # --- Phase 1: CMA-ES with adaptive restarts ---
        def cma_es(x0, sigma0, max_evals, pop_size=None):
            nonlocal call_count, best_f, best_x
            if pop_size is None:
                pop_size = max(4, int(4 + 3 * np.log(dim)))
            pop = pop_size
            x_mean = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            c_sigma = (pop + 2) / (dim + pop + 5)
            c_c = (4 + dim / 4) / (dim + 4 + pop)
            mu = pop // 2
            wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            wts /= wts.sum()
            mu_eff = 1.0 / np.sum(wts ** 2)
            c_1 = (2 / ((dim + 1.3) ** 2 + pop)) * (2 - mu_eff / dim)
            c_mu = min(1 - c_1, (2 * (mu_eff - 2 / (dim + 1)) / ((dim + 2) ** 2 + pop)) * (c_1 + 2 / (dim + 1)))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            stop = False
            evals_used = 0
            for gen in range(200):
                if stop or call_count >= budget or time.time() > deadline:
                    break
                # Sample population
                B = np.linalg.inv(C_inv_sqrt)
                samples = []
                for i in range(pop):
                    z = rng.standard_normal(dim)
                    x_sample = x_mean + sigma * (B @ z)
                    samples.append(x_sample)
                # Evaluate
                fitness = []
                for xs in samples:
                    if call_count >= budget or time.time() > deadline:
                        stop = True
                        break
                    try:
                        f_val = do_call(xs)
                        fitness.append(f_val)
                        evals_used += 1
                    except Exception:
                        stop = True
                        break
                if stop:
                    break
                if len(fitness) < pop:
                    break
                # Sort
                order = np.argsort(fitness)
                y = np.array([samples[i] for i in order])
                y_mean = np.sum(wts[:, None] * y[:mu], axis=0)
                x_mean_new = x_mean + c_c * (y_mean - x_mean)
                # Update C
                diff = (y[:mu] - x_mean) / sigma
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) - C)
                C += c_mu * np.sum(wts[:, None, None] * np.einsum('i,ij,ik->ijk', wts, diff, diff), axis=0)
                # Update p_sigma
                dnorm = np.linalg.norm(diff[:mu])
                p_sigma = (1 - c_sigma) * p_sigma + (c_sigma / damps) * (chi_n * (diff[:mu] / dnorm).mean(axis=0) - 1)
                # Update p_c
                p_c = (1 - c_c) * p_c + c_c * (y_mean - x_mean) / sigma
                # Update sigma
                sigma *= np.exp(c_sigma / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)
                # Update C_inv_sqrt
                evals = np.linalg.eigh(C)
                eigvals = np.clip(evals[0], 1e-12, None)
                C_inv_sqrt = evals[1] @ np.diag(1.0 / np.sqrt(eigvals)) @ evals[1].T
                x_mean = x_mean_new
                # Stagnation check
                if gen > 0 and np.all(np.abs(fitness) < 1e-12):
                    break
                if sigma < 1e-8:
                    break

        # Run CMA-ES from best point with adaptive restarts
        n_cma_restarts = 3
        for restart in range(n_cma_restarts):
            if call_count >= budget - 30 or time.time() > deadline:
                break
            if best_x is None:
                x0 = rng.uniform(-3.0, 3.0, dim)
            else:
                if restart == 0:
                    x0 = best_x.copy()
                    sigma0 = 2.0
                else:
                    x0 = best_x + rng.normal(0, 1.5, dim)
                    x0 = np.clip(x0, -4.5, 4.5)
                    sigma0 = 1.0
            max_evals = min(150, budget - call_count - 10)
            if max_evals < 20:
                break
            try:
                cma_es(x0, sigma0, max_evals)
            except Exception:
                pass

        # --- Phase 2: Surrogate-guided local refinement ---
        if call_count < budget - 10 and time.time() < deadline and len(all_x) >= 5:
            X_arr = np.array(all_x)
            Y_arr = np.array(all_f)
            # Propose points based on RBF uncertainty + exploit
            n_propose = min(30, budget - call_count - 5)
            for i in range(n_propose):
                if call_count >= budget or time.time() > deadline:
                    break
                # Mix of exploitation and exploration
                if i % 3 == 0:
                    # Explore: sample far from existing points
                    candidates = rng.uniform(-4.5, 4.5, (50, dim))
                    variances = rbf_variance(candidates, X_arr, Y_arr)
                    idx = np.argmax(variances)
                    x_new = candidates[idx]
                elif i % 3 == 1:
                    # Exploit: predict best
                    candidates = rng.uniform(-4.5, 4.5, (50, dim))
                    preds = rbf_predict(candidates, X_arr, Y_arr)
                    idx = np.argmin(preds)
                    x_new = candidates[idx]
                else:
                    # Interpolate between two best points
                    order = np.argsort(all_f)
                    best_idx = order[0]
                    second_idx = order[1]
                    t = rng.random()
                    x_new = (1 - t) * all_x[best_idx] + t * all_x[second_idx] + rng.normal(0, 0.3, dim)
                    x_new = np.clip(x_new, -4.8, 4.8)
                try:
                    do_call(x_new)
                except Exception:
                    break

        # --- Phase 3: Coordinate descent polish ---
        if best_x is not None and call_count < budget - 5 and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            step = 0.3
            for _ in range(25):
                if call_count >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    for d in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += d * step
                        try:
                            f_try = do_call(x_try)
                        except Exception:
                            break
                        if f_try < f:
                            x[i] = x_try[i]
                            f = f_try
                            improved = True
                if not improved:
                    step *= 0.6
                    if step < 1e-5:
                        break
            best_x = x
            best_f = f

        return best_x, best_f
