"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hierarchical deterministic grid scan followed by localized CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse grid scan to identify the global basin.
        # Use a Halton sequence for deterministic space-filling sampling.
        # We'll use ~400 calls for this phase.
        
        def halton_sequence(n, base):
            """Generate n Halton sequence values in [0, 1) for given base."""
            result = []
            f = 1.0
            for i in range(1, n + 1):
                x = 0.0
                f = 1.0
                i_temp = i
                while i_temp > 0:
                    f /= base
                    x += f * (i_temp % base)
                    i_temp //= base
                result.append(x)
            return result

        # Generate Halton sequences for each dimension
        n_grid = 400
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
        
        # Generate grid points using Halton sequences
        grid_points = []
        for d in range(dim):
            halton_vals = halton_sequence(n_grid, bases[d])
            if d == 0:
                grid_points = [[v] for v in halton_vals]
            else:
                for i in range(n_grid):
                    grid_points[i].append(halton_vals[i])
        
        # Map Halton values from [0,1) to [-5, 5]
        for point in grid_points:
            for d in range(dim):
                point[d] = -5 + 10 * point[d]
        
        # Evaluate all grid points
        best_x = None
        best_f = np.inf
        calls_used = 0
        for point in grid_points:
            x = np.array(point, dtype=float)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Localized CMA-ES around the best grid point
        # Use remaining budget for CMA-ES
        remaining_budget = budget - calls_used
        
        # Initialize CMA-ES parameters
        x = best_x.copy()
        sigma = 1.0  # Start with a moderate step size
        C = np.eye(dim)  # Covariance matrix
        L = np.eye(dim)  # Cholesky factor
        ps = np.zeros(dim)  # Path for covariance adaptation
        pc = np.zeros(dim)  # Path for step-size adaptation
        cs = 1 / (dim + 4)  # Step-size adaptation time constant
        c1 = 2 / ((dim + 1.3) ** 2 + 10)  # Rank-one update weight
        cmu = min(1 - c1, 2 * ((dim - 2) + 2 / (dim + 1)) / ((dim + 2) ** 2 + 15))  # Rank-mu update weight
        cmu = min(cmu, 1 - c1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Number of offspring per iteration
        lambda_ = min(4 * dim, max(2, remaining_budget // 3))
        
        # CMA-ES loop
        gen = 0
        while calls_used < budget and sigma > 1e-8:
            # Generate lambda_ offspring
            offspring = []
            offspring_f = []
            for _ in range(lambda_):
                if calls_used >= budget:
                    break
                # Sample from N(x, sigma^2 * C) using Cholesky
                z = np.random.default_rng().standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append(y)
                offspring_f.append(fy)
                calls_used += 1
            
            if not offspring:
                break
            
            # Sort by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]
            
            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_offspring[0].copy()
            
            # Update mean
            mu_eff = int(np.floor(lambda_ / 2))
            weights = np.ones(mu_eff)
            weights[0] = 2
            weights /= weights.sum()
            
            new_mean = np.sum([weights[i] * sorted_offspring[i] for i in range(mu_eff)], axis=0)
            
            # Update paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (np.linalg.solve(L, (new_mean - x) / sigma))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < \
                   (1 + 2 / (dim + 1)) * chi_n
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * (new_mean - x) / sigma
            
            # Update covariance
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                c1 * np.outer(pc, pc) + \
                cmu * np.sum([weights[i] * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x) 
                              for i in range(mu_eff)], axis=0)
            
            # Update sigma
            sigma *= np.exp(cs / (dim + 4) * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            x = new_mean
            
            # Update Cholesky factor
            try:
                L = np.linalg.cholesky(C)
            except:
                # Regularize if not positive definite
                C = C + 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(dim)
            
            gen += 1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
