"""Rewritten optimizer: Nelder-Mead from diverse grid + CMA-ES polish."""
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: Nelder-Mead from diverse starts + CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        calls = 0
        best_x = None
        best_f = np.inf

        def check_deadline():
            return time.time() > deadline

        def eval_point(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            f = float(func(np.asarray(x, dtype=float)))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=float).copy()
            return f

        # Phase 1: Nelder-Mead from 25 diverse starting points, 20 calls each
        # Generate diverse starting points using a grid + jitter
        n_starts = 25
        budget_per_start = 20

        # Create starting points: mix of center, corners, and random
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Corners of [-4, 4]^dim (since minimizer is in [-4,4]^5)
        for i in range(2 ** min(dim, 4)):
            corner = np.array([(-4 if (i >> d) & 1 else 4) for d in range(dim)])
            starts.append(corner)
        # Midpoints of edges
        for d in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[d] = sign * 4.0
                starts.append(pt)
        # Random points to fill up to n_starts
        while len(starts) < n_starts:
            starts.append(rng.uniform(-4.5, 4.5, dim))

        starts = np.array(starts[:n_starts])

        # Run Nelder-Mead from each start
        for s_idx in range(n_starts):
            if calls >= budget or check_deadline():
                break

            x0 = starts[s_idx].copy()
            # Initial simplex: x0 plus small perturbations
            simplex = [x0.copy()]
            f0 = eval_point(x0)
            if calls >= budget or check_deadline():
                break

            step = 0.5
            for d in range(dim):
                pt = x0.copy()
                pt[d] += step
                pt = np.clip(pt, -5, 5)
                simplex.append(pt)
                eval_point(pt)
                if calls >= budget or check_deadline():
                    break

            if check_deadline() or len(simplex) < dim + 1:
                continue

            # Nelder-Mead iterations
            nm_calls = 1 + dim  # already used
            max_nm_calls = budget_per_start
            while nm_calls < max_nm_calls and calls < budget and not check_deadline():
                # Sort simplex by function value
                simplex_sorted = sorted(simplex, key=lambda x: eval_point(x) if False else 0)
                # Actually we need to track function values
                # Let me restructure: keep (x, f) pairs

                # Re-evaluate to get current values (but this wastes calls)
                # Better: maintain a list of (x, f)
                break

            # The above is getting messy. Let me implement NM properly with (x, f) tracking.

        # Let me restart with a cleaner implementation

        # Reset state
        calls = 0
        best_x = None
        best_f = np.inf

        def eval_point2(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            f = float(func(np.asarray(x, dtype=float)))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=float).copy()
            return f

        # Phase 1: Nelder-Mead from diverse starts
        n_starts = 25
        budget_per_start = 20

        # Generate diverse starting points
        starts = []
        starts.append(np.zeros(dim))
        for i in range(2 ** min(dim, 4)):
            corner = np.array([(-4 if (i >> d) & 1 else 4) for d in range(dim)])
            starts.append(corner)
        for d in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[d] = sign * 4.0
                starts.append(pt)
        while len(starts) < n_starts:
            starts.append(rng.uniform(-4.5, 4.5, dim))
        starts = np.array(starts[:n_starts])

        # Run Nelder-Mead from each start
        for s_idx in range(n_starts):
            if calls >= budget or check_deadline():
                break

            x0 = starts[s_idx].copy()
            # Build initial simplex
            simplex = [(x0.copy(), eval_point2(x0))]
            if calls >= budget or check_deadline():
                break

            step = 0.5
            for d in range(dim):
                pt = x0.copy()
                pt[d] += step
                pt = np.clip(pt, -5, 5)
                simplex.append((pt.copy(), eval_point2(pt)))
                if calls >= budget or check_deadline():
                    break

            if len(simplex) < dim + 1:
                continue

            nm_calls_used = 1 + dim
            max_nm = budget_per_start

            while nm_calls_used < max_nm and calls < budget and not check_deadline():
                # Sort by function value
                simplex.sort(key=lambda t: t[1])

                # Worst, second worst, centroid of rest
                worst = simplex[-1]
                second_worst = simplex[-2]
                centroid = np.mean([x for x, f in simplex[:-1]], axis=0)

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - worst[0])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = eval_point2(x_ref)
                nm_calls_used += 1

                if f_ref < simplex[0][1]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - worst[0])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = eval_point2(x_exp)
                    nm_calls_used += 1
                    if f_exp < f_ref:
                        simplex[-1] = (x_exp, f_exp)
                    else:
                        simplex[-1] = (x_ref, f_ref)
                elif f_ref < second_worst[1]:
                    simplex[-1] = (x_ref, f_ref)
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (worst[0] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = eval_point2(x_con)
                    nm_calls_used += 1
                    if f_con < worst[1]:
                        simplex[-1] = (x_con, f_con)
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = (simplex[0][0] + 0.5 * (simplex[i][0] - simplex[0][0]),
                                          eval_point2(simplex[0][0] + 0.5 * (simplex[i][0] - simplex[0][0])))
                        nm_calls_used += (len(simplex) - 1)
                        break

        # Phase 2: CMA-ES polish from the best point found
        if calls < budget and not check_deadline() and best_x is not None:
            # CMA-ES parameters
            n = dim
            lam = max(4, int(4 + 2 * np.log(n)))
            mu = lam // 2

            w_raw = np.log(np.arange(1, mu + 1))
            w_raw /= w_raw.sum()
            mu_w = (w_raw ** 2).sum()
            mu_eff = 1.0 / mu_w

            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 1) / (mu_eff + 2) ** 2 - 1))
            chi_n = np.sqrt(n * (1 - 1 / (4 * n) - 1 / (21 * n ** 2)))

            mean = best_x.copy()
            sigma = 0.5
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)

            # Eigendecomposition for C
            def eigendecomp(C):
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                B = eigvecs
                C_sqrt = (B * D) @ B.T
                C_sqrt_inv = (B / D) @ B.T
                return C_sqrt, C_sqrt_inv, B, D

            C_sqrt, C_sqrt_inv, B, D = eigendecomp(C)

            cma_calls = 0
            max_cma_calls = budget - calls

            while cma_calls < max_cma_calls and calls < budget and not check_deadline():
                # Generate samples
                Z = rng.standard_normal((lam, n))
                X = mean[None, :] + sigma * (Z @ C_sqrt)
                X = np.clip(X, -5, 5)

                F = np.empty(lam)
                for i in range(lam):
                    if calls >= budget or check_deadline():
                        break
                    F[i] = eval_point2(X[i])
                    cma_calls += 1

                if cma_calls == 0:
                    break

                # Pad with inf if not all evaluated
                if len(F) < lam or any(np.isnan(F)):
                    F = np.array([f if np.isfinite(f) else np.inf for f in F])
                    if len(F) < lam:
                        F = np.concatenate([F, np.full(lam - len(F), np.inf)])

                # Update best
                idx_min = int(np.argmin(F))
                if F[idx_min] < best_f:
                    best_f = F[idx_min]
                    best_x = X[idx_min].copy()

                # Sort
                order = np.argsort(F)
                X_sorted = X[order]

                # Save old mean
                mean_old = mean.copy()

                # Update mean
                mean = mean_old + sigma * np.sum(w_raw[:, None] * X_sorted[:mu], axis=0)

                # Update ps
                ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_w) / sigma * (C_sqrt_inv @ (mean - mean_old))

                # Update pc
                gen = cma_calls // lam + 1
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen))) < (1 + 2 / (n + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * c_c * (mean - mean_old) / sigma

                # Update C
                delta_sig = hsig * (1 - c_c) + c_c
                C = (1 - c_1 - c_mu) * C + c_1 * (n + delta_sig - 1) * C + c_mu * np.outer((mean - mean_old) / sigma, (mean - mean_old) / sigma)

                # Ensure C is symmetric
                C = 0.5 * (C + C.T)

                # Update eigendecomposition
                C_sqrt, C_sqrt_inv, B, D = eigendecomp(C)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

        return best_x, best_f
