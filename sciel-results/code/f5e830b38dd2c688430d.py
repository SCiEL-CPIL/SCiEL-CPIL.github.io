"""CMA-ES with multi-start for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start restarts for rotated bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 330s kill

        # Global best tracking
        best_x = np.array([0.0] * dim)
        best_f = float('inf')
        calls_made = 0

        def eval_point(x):
            nonlocal calls_made
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_made += 1
            return x, f

        # Phase 1: Multi-start with CMA-ES
        # Use several restarts with different initial points
        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Initial sigma
        sigma0 = 1.5

        # Number of restarts - use budget to determine
        # Each CMA-ES run uses roughly lam * (1 + 1/c_c) * log(d) * 10 evaluations
        # For dim=5, that's roughly 4 * 1.5 * 1.6 * 10 ≈ 96 evals
        # So we can do about 5 restarts in 500 budget

        n_restarts = 5
        # Allocate budget: first restart gets more, later ones get less
        # Actually, let's do a simpler allocation
        # First 3 restarts: 100 evals each, remaining 2: 50 each
        # Total: 400, leaving 100 for final refinement

        # Better: use adaptive allocation based on progress
        # Let's do: 5 restarts, each with decreasing budget
        restart_budgets = [120, 100, 80, 70, 60]  # total = 430, leaving 70 for refinement

        # Initial points: center + several random points
        rng = np.random.default_rng(42)

        # Start from center (0,0,0,0,0) which is often near the minimum for Lunacek
        # and several random points
        init_points = [np.zeros(dim)]
        for i in range(n_restarts - 1):
            init_points.append(rng.uniform(-4, 4, dim))

        for restart_idx in range(n_restarts):
            if time.time() > deadline - 30:
                break
            if calls_made >= budget - 5:
                break

            x0 = init_points[restart_idx]
            x0, f0 = eval_point(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # CMA-ES parameters
            c1 = (4 + 3 / dim) / ((dim + 1) ** 2 + 4)
            cmu = min(1 - c1, 2 * (2 - (dim + 1) / (dim + 3)) ** 2 - 1) / ((dim + 1) ** 2 + 4)
            cc = (4 + dim / 4) / (dim + 4 + 4)
            cs = (cc + 1 + 2 * min(1, 1 / (cc * (lam - 2))) - c1) / (2 + cc + 2 * min(1, 1 / (cc * (lam - 2))))
            damps = dim + 2 + 2 * min(1, 1 / (cs * (lam - 2)))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # State
            m = x0.copy()  # mean
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            max_evals = restart_budgets[restart_idx]
            stagnation = 0
            best_in_run = f0
            best_x_in_run = x0.copy()

            for gen in range(max_evals):
                if time.time() > deadline - 20:
                    break
                if calls_made >= budget - 5:
                    break

                # Generate population
                ys = []
                fs = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    # Transform to search space
                    y = m + sigma * (B @ (D * z))
                    y, fy = eval_point(y)
                    ys.append(y)
                    fs.append(fy)
                    if fy < best_in_run:
                        best_in_run = fy
                        best_x_in_run = y.copy()

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = np.array([ys[i] for i in order])
                fs_sorted = [fs[i] for i in order]

                # Update best
                if fs_sorted[0] < best_f:
                    best_f = fs_sorted[0]
                    best_x = ys_sorted[0].copy()

                # Check stagnation
                if fs_sorted[0] >= best_in_run - 1e-10:
                    stagnation += 1
                else:
                    stagnation = 0

                if stagnation > 20:
                    break

                # Update mean
                weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
                weights /= weights.sum()
                mueff = int(weights.sum() ** 2 / (weights ** 2).sum())

                m_old = m.copy()
                for i in range(lam):
                    m += weights[i] * (ys_sorted[i] - m)

                # Update step size (CSA)
                ps = (1 - cs) * ps + cs * np.sqrt(cc * (2 - cc) / damps) * (B @ (D ** (-1)) @ ((m - m_old) / sigma))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) > (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))

                # Update covariance matrix
                delta = (m - m_old) / sigma
                pc = (1 - cc) * pc + cc * delta
                outer = np.outer(pc, pc)
                rank1 = c1 * (outer + (1 - (1 - cc) ** 2) * np.eye(dim))
                rankmu = cmu * np.sum([np.outer(ys_sorted[i] - m_old, ys_sorted[i] - m_old) for i in range(mueff)], axis=0) / sigma ** 2
                C = (1 + c1 + cmu - mu) * C + rank1 + rankmu if False else C  # placeholder

                # Actually let me redo this properly
                # I need to track C properly
                break  # This is getting complex, let me use scipy

        # Use scipy's CMA-ES instead for reliability
        from scipy.optimize import minimize
        # Actually, let me just implement a clean CMA-ES

        # Restart with clean implementation
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        def eval_point2(x):
            nonlocal calls_made
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_made += 1
            return x, f

        # Clean CMA-ES implementation
        def cma_es(x0, max_evals):
            nonlocal best_f, best_x, calls_made

            dim_local = dim
            lam_local = max(4, int(4 + 3 * np.log(dim_local)))
            c1 = (4 + 3 / dim_local) / ((dim_local + 1) ** 2 + 4)
            cmu = min(1 - c1, 2 * (2 - (dim_local + 1) / (dim_local + 3)) ** 2 - 1) / ((dim_local + 1) ** 2 + 4)
            cc = (4 + dim_local / 4) / (dim_local + 4 + 4)
            cs = (cc + 1 + 2 * min(1, 1 / (cc * (lam_local - 2))) - c1) / (2 + cc + 2 * min(1, 1 / (cc * (lam_local - 2))))
            damps = dim_local + 2 + 2 * min(1, 1 / (cs * (lam_local - 2)))
            chi_n = np.sqrt(dim_local) * (1 - 1 / (4 * dim_local) + 1 / (21 * dim_local ** 2))

            m = x0.copy()
            sigma = 1.5
            C = np.eye(dim_local)
            # Cholesky of C
            C_sqrt = np.linalg.cholesky(C)
            pc = np.zeros(dim_local)
            ps = np.zeros(dim_local)

            best_local_f = float('inf')
            best_local_x = x0.copy()

            # Initial evaluation
            x0_clipped, f0 = eval_point2(x0)
            m = x0_clipped.copy()
            best_local_f = f0
            best_local_x = x0_clipped.copy()
            if f0 < best_f:
                best_f = f0
                best_x = x0_clipped.copy()

            weights = np.log(lam_local / 2 + 0.5) - np.log(np.arange(1, lam_local + 1))
            weights /= weights.sum()
            mueff = int(weights.sum() ** 2 / (weights ** 2).sum())

            for gen in range(max_evals // lam_local + 1):
                if time.time() > deadline - 15:
                    break
                if calls_made >= budget - 5:
                    break

                # Generate population
                ys = []
                fs = []
                for i in range(lam_local):
                    z = rng.standard_normal(dim_local)
                    y = m + sigma * (C_sqrt @ z)
                    y, fy = eval_point2(y)
                    ys.append(y)
                    fs.append(fy)
                    if fy < best_local_f:
                        best_local_f = fy
                        best_local_x = y.copy()
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                # Sort
                order = np.argsort(fs)
                ys_sorted = np.array([ys[i] for i in order])
                fs_sorted = [fs[i] for i in order]

                # Update mean
                m_old = m.copy()
                for i in range(lam_local):
                    m += weights[i] * (ys_sorted[i] - m)

                # Update step size
                ps = (1 - cs) * ps + cs * np.sqrt(cc * (2 - cc) / damps) * np.linalg.solve(C_sqrt, (m - m_old) / sigma)
                norm_ps = np.linalg.norm(ps)
                if norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) > (1.4 + 2 / (dim_local + 1)) * chi_n:
                    sigma *= np.exp(cs / damps * (norm_ps / chi_n - 1))

                # Update covariance
                pc = (1 - cc) * pc + cc * (m - m_old) / sigma
                C = (1 + c1 + cmu) * C - c1 * np.outer(pc, pc)
                C += cmu * np.sum([np.outer(ys_sorted[i] - m_old, ys_sorted[i] - m_old) for i in range(mueff)], axis=0) / sigma ** 2

                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = (eigvecs * eigvals) @ eigvecs.T
                C = (C + C.T) / 2  # Symmetrize
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim_local)
                    C_sqrt = np.eye(dim_local)

                # Check convergence
                if sigma < 1e-10 or np.linalg.norm(m - m_old) < 1e-12:
                    break

            return best_local_f, best_local_x

        # Multi-start CMA-ES
        n_restarts = 4
        init_points = [np.zeros(dim)]
        for i in range(n_restarts - 1):
            init_points.append(rng.uniform(-4, 4, dim))

        # Budget allocation
        remaining = budget - calls_made
        # First restart gets 40%, rest split equally
        b0 = int(remaining * 0.4)
        b_rest = (remaining - b0) // (n_restarts - 1) if n_restarts > 1 else 0

        for i in range(n_restarts):
            if time.time() > deadline - 10:
                break
            if calls_made >= budget - 3:
                break

            b = b0 if i == 0 else b_rest
            cma_es(init_points[i], b)

        # Final local refinement with (1+1)-ES
        if calls_made < budget - 3 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            while calls_made < budget - 2 and time.time() < deadline - 3:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
