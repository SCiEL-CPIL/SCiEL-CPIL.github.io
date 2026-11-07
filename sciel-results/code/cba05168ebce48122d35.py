"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        lam = 20  # Population size
        mu = lam // 2  # Number of parents
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff)))
        
        # Step size control
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Initial state
        mean = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        
        # Eigendecomposition of C for sampling and updates
        # We will update this each generation
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        
        # State variables for evolution path
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1
        
        gen = 0
        while calls < budget:
            # 1. Sample from N(mean, sigma^2 * C)
            # C = V D V^T, so C^{1/2} = V D^{1/2} V^T
            # x = mean + sigma * C^{1/2} @ z
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            z = rng.normal(size=(lam, n))
            # x = mean + sigma * (z @ C_sqrt.T)
            pop = (mean + sigma * (z @ C_sqrt.T)).T
            
            # 2. Evaluate
            fvals = np.array([func(x) for x in pop])
            calls += lam
            
            # 3. Update best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_f:
                best_f = fvals[idx_best]
                best_x = pop[idx_best].copy()
            
            # 4. Sort by fitness
            order = np.argsort(fvals)
            y = pop[order]
            z_sorted = z[order]
            
            # 5. Update mean
            mean_old = mean.copy()
            mean = np.sum(w[:, None] * y[:mu], axis=0)
            
            # 6. Update evolution paths
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (mean - mean_old) / sigma
            # Note: (mean - mean_old) is the step.
            # C_inv_sqrt is needed for the standard formula?
            # Standard formula: ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * C^{-1/2} * (mean - mean_old) / sigma
            # Let's use C_inv_sqrt
            D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
            C_inv_sqrt = eigvecs @ D_inv_sqrt @ eigvecs.T
            
            step = mean - mean_old
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ step / sigma)
            
            # hsig
            ps_norm = np.linalg.norm(ps)
            # Approximation for the denominator
            # 1 - (1-cs)^(2*(gen+1))
            denom = 1 - (1 - cs) ** (2 * (gen + 1))
            if denom < 1e-10:
                denom = 1e-10
            hsig = (ps_norm / (np.sqrt(denom) * chi_n)) < 1.4 + 2 / (n + 1)
            
            # pc = (1-cc)*pc + sqrt(cc*(2-cc)*hsig) * step / sigma
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * hsig) * (step / sigma)
            
            # 7. Update covariance matrix
            # C = (1 - c1 - cmu)*C + c1*(ps*ps.T + (1-hsig)*C) + cmu * sum(w_i * y_i * y_i.T)
            # y_i = (x_i - mean_old) / sigma
            y_norm = (y[:mu] - mean_old) / sigma
            # sum w_i * y_i * y_i.T
            # This is a weighted outer product sum
            # Can be computed as: (w[:, None] * y_norm) @ y_norm.T
            # Wait, y_norm is (mu, n). w is (mu,).
            # (w[:, None] * y_norm) is (mu, n).
            # Outer product sum: sum_i w_i * y_i y_i^T
            # This is equivalent to (w[:, None] * y_norm) @ y_norm.T
            # Let's verify:
            # Result[j, k] = sum_i (w_i * y_norm[i, j]) * y_norm[i, k] = sum_i w_i y_norm[i, j] y_norm[i, k]
            # Yes, correct.
            
            cov_update = (w[:, None] * y_norm) @ y_norm.T
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(ps, ps) + (1 - hsig) * C) + cmu * cov_update
            
            # 8. Update sigma
            sigma *= np.exp((cs / damps) * (ps_norm / chi_n - 1))
            
            # 9. Update eigendecomposition for next iteration
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            
            gen += 1
            
            # If we can't afford another full generation, break
            if calls + lam > budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
