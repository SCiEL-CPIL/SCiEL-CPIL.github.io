"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid scan followed by localized CMA-ES for Büche-Rastrigin f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse grid scan with ~400 calls
        # Use a 4x4x4x4x4 grid = 1024 points, but we only have 400 calls
        # Use a smarter approach: stratified sampling with 4 points per dimension
        # Actually, let's use a Latin hypercube or stratified random sampling
        # For 400 calls in 5D, we can do a stratified grid: 4^5 = 1024 is too many
        # Let's use 3 points per dimension: 3^5 = 243, then add more around best regions
        # Or use a combination: first a coarse 3^5 grid, then refine

        # Strategy: Use stratified sampling with 400 points
        # Divide each dimension into 4 equal intervals, pick random point in each cell
        # But 4^5 = 1024 cells, we can only sample 400
        # Alternative: Use a grid with varying resolution
        # Let's use a 4x4x4x4x4 grid but subsample it, or use a different approach

        # Better approach: Use a hierarchical grid
        # Level 1: 3^5 = 243 points (coarse)
        # Level 2: Refine around best point with additional points

        # Phase 1a: Coarse 3^5 grid
        grid_vals = np.linspace(-5, 5, 3)  # [-5, 0, 5]
        grid_points = []
        for d0 in grid_vals:
            for d1 in grid_vals:
                for d2 in grid_vals:
                    for d3 in grid_vals:
                        for d4 in grid_vals:
                            grid_points.append([d0, d1, d2, d3, d4])
        grid_points = np.array(grid_points)

        # Evaluate all grid points (243 calls)
        for pt in grid_points:
            if calls_made >= total_budget or time.time() >= deadline:
                break
            evaluate(pt)

        # Phase 1b: Add midpoints between best grid point and its neighbors
        # Find the best grid point
        if best_x is not None:
            # Add points at midpoints between best and each axis neighbor
            for axis in range(dim):
                for delta in [-1, 1]:
                    neighbor = best_x.copy()
                    neighbor[axis] += delta * 5.0
                    if np.all(neighbor >= -5) and np.all(neighbor <= 5):
                        if calls_made < total_budget and time.time() < deadline:
                            mid = (best_x + neighbor) / 2
                            evaluate(mid)

            # Add points at quarter positions around best
            for axis in range(dim):
                for frac in [0.25, 0.75]:
                    pt = best_x.copy()
                    # Interpolate between -5 and 5 at this fraction for this axis
                    pt[axis] = -5 + frac * 10
                    if np.all(pt >= -5) and np.all(pt <= 5):
                        if calls_made < total_budget and time.time() < deadline:
                            evaluate(pt)

        # Phase 2: Localized CMA-ES around the best point found by grid
        if calls_made < total_budget and time.time() < deadline and best_x is not None:
            # CMA-ES Parameters
            lambda_pop = 9
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)

            cc = (4 + mu_eff / lambda_pop) / (lambda_pop + 4 + 2 * mu_eff / lambda_pop)
            cs = (mu_eff + 2) / (lambda_pop + mu_eff + 5)
            c1 = 2 / ((lambda_pop + 2) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((lambda_pop + 2) ** 2 + mu_eff))

            x_mean = best_x.copy()
            sigma = 1.0  # Start with a reasonable step size
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            local_best_x = best_x.copy()
            local_best_f = best_f

            no_improve_gen = 0
            gen = 0

            while calls_made < total_budget and time.time() < deadline:
                # Sample lambda points
                offspring = []
                for i in range(lambda_pop):
                    if calls_made >= total_budget or time.time() >= deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x_new = x_mean + sigma * y
                    f_new = evaluate(x_new)
                    offspring.append((x_new, f_new))

                if len(offspring) == 0:
                    break

                # Sort by fitness
                offspring.sort(key=lambda x: x[1])

                # Update mean
                x_new_mean = np.sum([weights[i] * offspring[i][0] for i in range(min(mu, len(offspring)))], axis=0)

                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (B.T @ (x_new_mean - x_mean))))

                hsig = (np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * np.sqrt(lambda_pop + 1 - mu_eff))

                if hsig:
                    sigma *= np.exp(cs / dim * (np.linalg.norm(ps) / np.sqrt(lambda_pop) - 1))

                # Update covariance
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new_mean - x_mean) / sigma

                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(offspring))):
                    y_i = (offspring[i][0] - x_mean) / sigma
                    rankmu += weights[i] * np.outer(y_i, y_i)

                C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2

                # Update mean
                x_mean = x_new_mean
                gen += 1

                # Update best
                if offspring[0][1] < local_best_f:
                    local_best_f = offspring[0][1]
                    local_best_x = offspring[0][0].copy()
                    no_improve_gen = 0
                else:
                    no_improve_gen += 1

                # Update basis
                try:
                    D2, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D2, 1e-10))
                except:
                    D2, B = np.linalg.eigh(C + 1e-10 * np.eye(dim))
                    D = np.sqrt(np.maximum(D2, 1e-10))

                # Check for stall
                if no_improve_gen >= 15:
                    break

                # Check convergence
                if np.linalg.norm(D) < 1e-10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
