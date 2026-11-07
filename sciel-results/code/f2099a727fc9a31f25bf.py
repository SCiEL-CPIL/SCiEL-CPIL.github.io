"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES optimizer for BBOB f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill

        # We will run 3 parallel CMA-ES instances (islands)
        # Budget per island: roughly total_budget / 3
        num_islands = 3
        budget_per_island = total_budget // num_islands
        remaining_budget = total_budget - (num_islands * budget_per_island)

        # Initial points: One at center, others random in [-4, 4]
        # The minimizer is in [-4, 4]^5
        start_points = [np.zeros(dim)]
        rng = np.random.default_rng(42)
        for _ in range(num_islands - 1):
            start_points.append(rng.uniform(-4, 4, dim))

        global_best_x = np.zeros(dim)
        global_best_f = np.inf

        # Helper to run CMA-ES from a specific start point with a specific budget
        def run_cmaes(start_x, budget):
            # CMA-ES Parameters
            lam = 8  # Population size for dim=5
            mu = lam // 2  # Number of parents
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = (2 * (mueff - 2 + 2 / mueff)) / ((dim + 2)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            
            # Initial state
            m = start_x.copy()
            sigma = 1.0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            best_x = m.copy()
            best_f = func(best_x)
            evals = 1
            
            chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
            
            while evals < budget:
                if time.time() - start_time > time_limit:
                    break
                
                # Sample
                ys = np.random.randn(lam, dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                # x = m + sigma * z, where z ~ N(0, C)
                # z = y @ L.T
                zs = ys @ L.T
                xs = m[None, :] + sigma * zs
                xs = np.clip(xs, -5, 5)
                
                # Evaluate
                fs = np.array([func(x) for x in xs])
                evals += lam
                
                # Update best
                idx_best = np.argmin(fs)
                if fs[idx_best] < best_f:
                    best_f = fs[idx_best]
                    best_x = xs[idx_best].copy()
                
                # Sort
                idx_sorted = np.argsort(fs)
                y_sorted = ys[idx_sorted]
                x_sorted = xs[idx_sorted]
                
                # Weighted mean
                new_m = np.dot(weights, x_sorted[:mu])
                
                # Update step size
                diff = (new_m - m) / sigma
                v = np.linalg.solve(L.T, diff)
                
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * v
                
                # Simple hsig condition
                hsig = 1.0 if np.linalg.norm(ps) < (1.4 + 2.0 / (dim + 1.0)) * chiN else 0.0
                
                sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1.0))
                
                # Update covariance
                pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * diff
                
                rank1_update = np.outer(pc, pc)
                v_mu = (x_sorted[:mu] - m) / sigma
                rankmu_update = np.dot((weights[:, None] * v_mu).T, v_mu)
                
                C = (1 + c1 * hsig * (2 - hsig) - c1 - cmu) * C + c1 * rank1_update + cmu * rankmu_update
                C = (C + C.T) / 2.0
                
                m = new_m
                
                # Check if we can do another generation
                if budget - evals < lam:
                    break

            return best_x, best_f

        # Run each island
        for i in range(num_islands):
            if time.time() - start_time > time_limit:
                break
            bx, bf = run_cmaes(start_points[i], budget_per_island)
            if bf < global_best_f:
                global_best_f = bf
                global_best_x = bx.copy()

        # Use remaining budget for local refinement
        if remaining_budget > 0 and time.time() - start_time < time_limit:
            # Simple random walk refinement
            local_sigma = 0.1
            for _ in range(remaining_budget):
                if time.time() - start_time > time_limit:
                    break
                # Try a few small steps
                step = local_sigma * np.random.randn(dim)
                new_x = np.clip(global_best_x + step, -5, 5)
                new_f = func(new_x)
                if new_f < global_best_f:
                    global_best_f = new_f
                    global_best_x = new_x
                else:
                    local_sigma *= 0.95

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
