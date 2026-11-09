"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # State
        best_x = None
        best_f = np.inf
        call_count = 0

        def check_deadline():
            nonlocal deadline
            return (time.time() - start_time) > (deadline - 5.0)

        def do_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-restart CMA-ES with diverse initializations
        # We'll do several short CMA-ES runs from different starting points
        num_restarts = 8
        calls_per_restart = budget // num_restarts
        # Reserve some calls for final refinement
        calls_per_restart = max(30, calls_per_restart)

        for restart in range(num_restarts):
            if check_deadline() or call_count >= budget:
                break

            # Diverse initialization: mix of random, grid-like, and center
            if restart == 0:
                x0 = np.zeros(dim)
            elif restart == 1:
                x0 = rng.uniform(-3, 3, dim)
            elif restart == 2:
                x0 = rng.uniform(-4, 4, dim)
            elif restart == 3:
                x0 = rng.uniform(-2, 2, dim)
            else:
                # Random points with some structure
                x0 = rng.uniform(-5, 5, dim)
                # Add some bias toward center for later restarts
                x0 = x0 * (1.0 - 0.1 * restart)

            fx0 = do_call(x0)
            if fx0 is None:
                break

            # CMA-ES parameters
            sigma = 1.5  # initial step size
            lam = max(4, int(4 + 3 * np.log(dim)))  # population size
            C = np.eye(dim)  # covariance matrix
            ps = np.zeros(dim)  # evolution path for sigma
            pc = np.zeros(dim)  # evolution path for C
            cs = 1.0 / (lam + 3)
            ds = 1 + 2 * max(0, (np.sqrt((lam + 1 - 2) / (dim + 1)) - 1)) / (dim + 3)
            cmu = min(2 / ((dim + 1.3) ** 2 - 1), 0.5 * cs)
            c1 = (2 / ((dim + 1.7) ** 2 + 2))
            cmu = min(cmu, (1 - c1) * 0.5)
            chi = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))

            # Main CMA-ES loop for this restart
            for gen in range(calls_per_restart // lam + 1):
                if check_deadline() or call_count >= budget:
                    break

                # Generate population
                B = rng.standard_normal((lam, dim))
                # Whitening transformation
                D = np.diag(np.sqrt(np.diag(C)))
                B = B @ D @ np.linalg.inv(np.linalg.cholesky(C))
                # Actually, let's use a simpler approach: use eigendecomposition
                # For small dim, we can do it directly
                
                # Simpler: generate candidates using the current covariance
                # Use Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

                candidates = np.array([x0 + sigma * (B[i] @ L) for i in range(lam)])
                
                # Evaluate population
                fits = []
                for i in range(lam):
                    if check_deadline() or call_count >= budget:
                        break
                    fi = do_call(candidates[i])
                    if fi is None:
                        break
                    fits.append((fi, i))

                if len(fits) < 2:
                    break

                # Sort by fitness
                fits.sort()
                mu = lam // 2  # number of parents

                # Update mean
                w = np.log(np.arange(1, mu + 1))
                w = w / np.sum(w)
                mu_eff = 1.0 / np.sum(w ** 2)

                old_x0 = x0.copy()
                x0_new = np.sum(w[:, None] * candidates[[f[1] for f in fits[:mu]]], axis=0)
                x0 = x0_new

                # Update sigma
                if np.linalg.norm(x0_new - old_x0) > 1e-14:
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * \
                         (L.T @ ((x0_new - old_x0) / (sigma * np.linalg.norm(x0_new - old_x0))))
                    sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
                    sigma = np.clip(sigma, 1e-10, 10.0)

                # Update covariance
                v = x0_new - old_x0
                pc = (1 - c1) * pc + np.sqrt(c1 * (2 - c1) * mu_eff) * v / sigma
                hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < \
                          (1.4 + 2 / (dim + 1)) * chi else 0.0
                C = (1 + c1 * hsig - c1 - cmu) * C + \
                    c1 * np.outer(pc, pc) + \
                    cmu / sigma ** 2 * np.sum(w[:, None, None] * np.outer(v, v) * 0 + 
                    w[:, None] * np.outer(candidates[[f[1] for f in fits[:mu]]] - old_x0,
                                          candidates[[f[1] for f in fits[:mu]]] - old_x0), axis=0)
                # Ensure C is PSD
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Phase 2: Fine-tuning with small steps around the best point
        if best_x is not None and not check_deadline() and call_count < budget:
            sigma_final = 0.1
            for _ in range(min(50, budget - call_count)):
                if check_deadline() or call_count >= budget:
                    break
                y = best_x + sigma_final * rng.standard_normal(dim)
                fy = do_call(y)
                if fy is not None and fy < best_f:
                    sigma_final *= 1.2
                else:
                    sigma_final *= 0.85
                    sigma_final = max(sigma_final, 1e-4)

        return best_x, best_f
# EVOLVE-BLOCK-END
