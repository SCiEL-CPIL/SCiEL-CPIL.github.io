"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid optimizer: Lattice initialization + CMA-ES + Coordinate Descent."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')

        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline:
                return float('inf')
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        # 1. Deterministic Lattice Initialization
        # This provides a robust global overview and identifies the best basin quickly.
        # For dim=5, a 3^5=243 grid is feasible within the 500 budget if we are careful,
        # but we can be smarter. Let's use a stratified random or a smaller grid.
        # Given the "ill-conditioned" nature, a coarse grid might miss the narrow valley.
        # Let's use a combination: 20 random points + a small grid around the best random point?
        # Actually, the inspiration suggests a lattice. Let's try a 3x3x3x3x3 grid? 
        # 3^5 = 243 points. That leaves 257 for CMA-ES/CD. This is a good split.
        
        # Let's generate a 3-level grid in [-4.5, 4.5] to stay inside bounds comfortably
        # and cover the likely region [-4, 4].
        grid_vals = np.array([-3.0, 0.0, 3.0])
        
        # Generate all combinations
        # This creates 243 points
        grid_points = np.array(np.meshgrid(*([grid_vals]*n))).T.reshape(-1, n)
        
        # Shuffle to avoid bias in ordering (though deterministic is fine, shuffling helps if time cuts off early)
        # But we want reproducibility for the seed.
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(grid_points))
        
        for idx in indices:
            if n_calls >= budget or time.time() > deadline:
                break
            x = grid_points[idx]
            timed_call(x)
        
        if n_calls >= budget or time.time() > deadline:
            return best_x, best_f

        # 2. CMA-ES Run from the best point found
        # Use the best point from the lattice as the starting mean.
        # This allows CMA-ES to refine the solution in the identified basin.
        
        x_start = best_x.copy()
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n)) # ~10 for n=5
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        sigma = 1.0 # Start with a moderate step size
        m = x_start.copy()
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Precompute C_sqrt
        eigvals, eigvecs = np.linalg.eigh(C)
        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        gen = 0
        max_cma_gens = 20 # Limit CMA-ES generations to leave time for CD
        
        while gen < max_cma_gens and n_calls < budget and time.time() < deadline:
            # Check if we have enough budget for a full population
            if n_calls + lam > budget:
                break
                
            ys = []
            fs = []
            
            for i in range(lam):
                z = rng.standard_normal(n)
                y = m + sigma * C_sqrt @ z
                fy = timed_call(y)
                ys.append(y)
                fs.append(fy)
                
                if time.time() > deadline:
                    break
            
            if len(fs) < lam:
                break
                
            ys = np.array(ys)
            fs = np.array(fs)
            
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                
            # Weighted mean
            m_new = np.dot(weights, ys[:mu])
            
            # Step size adaptation
            sigma_old = sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(C_inv_sqrt @ (m_new - m)) / sigma_old - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Evolution paths
            hs = (np.linalg.norm(C_inv_sqrt @ (m_new - m)) / (1 - (1 + gen / (5 * (n + lam))) ** (-1)) < 1.4 + 2 / (n + 1))
            p_sigma = (1 - c_sigma) * p_sigma + hs * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (m_new - m) / sigma
            
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) * (2 - mu_eff)))
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) / sigma
            
            # Covariance matrix update
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = C_inv_sqrt @ (ys[i] - m) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
                
            C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C + c_c * (rank_one - (1 - hsig) * c_c * (2 - c_c) * C) + c_1 * (rank_one + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * rank_mu
            
            # Ensure positive definite
            eigvals_c = np.linalg.eigvalsh(C)
            if np.any(eigvals_c < 1e-12):
                C += 1e-12 * np.eye(n)
            
            m = m_new
            gen += 1
            
            # Update square roots periodically
            if gen % 5 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # 3. Coordinate Descent Polish
        # This exploits the axis-aligned structure of Schaffer's F7.
        # Use the remaining budget to refine the solution found by CMA-ES.
        
        if n_calls < budget and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            
            # Step sizes from coarse to fine
            steps = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            
            for step in steps:
                if n_calls >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and (n_calls < budget and time.time() < deadline):
                    improved = False
                    for i in range(n):
                        if n_calls >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            if n_calls >= budget or time.time() > deadline:
                                break
                            new_x = x.copy()
                            new_x[i] += direction * step
                            new_x = np.clip(new_x, -5.0, 5.0)
                            if np.allclose(new_x, x):
                                continue
                            f_new = timed_call(new_x)
                            if f_new < f:
                                x = new_x
                                f = f_new
                                improved = True
                                best_x = x.copy()
                                best_f = f

        return best_x, best_f
# EVOLVE-BLOCK-END
