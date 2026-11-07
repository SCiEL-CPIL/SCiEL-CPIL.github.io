"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES-like optimizer with global exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_step(self, x, sigma, func, rng, C=None, p_sigma=None, p_c=None,
                  alpha_mu=None, mu=None, weights=None, c_c=None, c_sigma=None):
        """One step of simplified CMA-ES. Returns new x, sigma, and updated state."""
        n = self.dim
        if C is None:
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            mu = max(1, n // 2)
            # Logarithmic weights
            w_raw = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights = w_raw / w_raw.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_sigma = (2 + mu_eff / (2 * n)) / (n + 5 + mu_eff / (2))
            alpha_mu = 2.0

        # Generate candidates
        # Whitened noise
        L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
        candidates = []
        for i in range(mu):
            z = rng.standard_normal(n)
            y = x + sigma * (L @ z)
            y = np.clip(y, -5, 5)
            candidates.append(y)
        
        # Evaluate
        fvals = [func(c) for c in candidates]
        
        # Sort by fitness
        indices = np.argsort(fvals)
        sorted_candidates = [candidates[i] for i in indices]
        sorted_fvals = [fvals[i] for i in indices]
        
        best_y = sorted_candidates[0]
        best_f = sorted_fvals[0]
        
        # Update mean
        x_new = x + sigma * sum(weights[i] * (sorted_candidates[i] - x) for i in range(mu))
        
        # Update C
        diff = np.array([sorted_candidates[i] - x for i in range(mu)])
        C_new = (1 - c_c * (2 - c_c)) * C
        for i in range(mu):
            C_new += weights[i] * np.outer(diff[i], diff[i])
        
        # Update step size
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (best_y - x) / sigma
        h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (n + 1)) * np.trace(C) / n else 0.0
        sigma *= np.exp(c_sigma / sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) - np.trace(C) / n))
        
        # Update C with rank-one correction for p_c
        p_c = (1 - c_c) * p_c + c_c * h_sigma * (best_y - x) / sigma
        C_new += (1 + (1 - h_sigma) * c_c * (2 - c_c)) * c_c * (1 - c_c) * np.outer(p_c, p_c) - c_c * (2 - c_c) * C
        
        # Ensure symmetry and positive semi-definite
        C_new = (C_new + C_new.T) / 2
        try:
            eigvals, eigvecs = np.linalg.eigh(C_new)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
        except:
            C_new = (C_new + C_new.T) / 2
            C_new += 1e-12 * np.eye(n)
        
        return best_y, best_f, x_new, sigma, C_new, p_sigma, p_c, alpha_mu, mu, weights, c_c, c_sigma

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        rng = np.random.default_rng(42)
        
        # Phase 1: Coarse grid search to find a good starting region
        # Use a small grid to quickly identify promising areas
        grid_size = 5
        grid_points = []
        for i in range(grid_size ** self.dim):
            coords = np.array([i // (grid_size ** (self.dim - 1 - d)) % grid_size for d in range(self.dim)])
            x = -5 + 2 * coords / (grid_size - 1) * 10
            grid_points.append(x)
        
        # Evaluate grid points (limited to avoid too many calls)
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Evaluate a subset of grid points first
        np.random.shuffle(grid_points)
        for x in grid_points:
            if time.time() > deadline:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if calls_made >= 100:
                break
        
        # Phase 2: Local refinement around the best grid point using CMA-ES
        if best_x is None:
            best_x = rng.uniform(-5, 5, self.dim)
            best_f = func(best_x)
            calls_made += 1
        
        x = best_x.copy()
        sigma = 1.0
        C = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        remaining_budget = self.budget - calls_made
        
        # Use CMA-ES for local optimization
        for step in range(remaining_budget):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break
            
            y = x + sigma * (np.linalg.cholesky(C + 1e-12 * np.eye(self.dim)) @ rng.standard_normal(self.dim))
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < best_f:
                # Update CMA-ES state
                diff = y - x
                C = (1 - 0.02) * C + 0.02 * np.outer(diff, diff) / sigma ** 2
                C = (C + C.T) / 2
                x = y
                best_f = fy
                sigma *= 1.1
            else:
                sigma *= 0.9
            
            # Periodically restart if stuck
            if step % 50 == 0 and step > 0:
                # Small perturbation to escape local minima
                x = x + 0.5 * sigma * rng.standard_normal(self.dim)
                x = np.clip(x, -5, 5)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
