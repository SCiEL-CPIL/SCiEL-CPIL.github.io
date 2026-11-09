"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with population size 4 and integrated restart rule for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # lambda = population size
        lam = 4
        mu = lam // 2  # number of best individuals
        # Weights
        w = np.log(mu + 1) - np.arange(1, mu + 1)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial step size
        sigma = 2.0
        # Initial mean
        m = np.zeros(dim)
        # Initial covariance matrix
        C = np.eye(dim)
        # Eigen decomposition cache
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)
        
        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 + 1 / (4 * dim) - 1 / (21 * dim ** 2))
        
        # Track best
        best_x = m.copy()
        best_f = func(best_x)
        calls_used = 1
        
        # Restart rule: if no improvement for a while or sigma becomes too small/large
        no_improve = 0
        max_no_improve = 100
        
        def do_generation():
            nonlocal m, C, sigma, pc, ps, eigvals, eigvecs, best_x, best_f, calls_used, no_improve
            nonlocal chi_n
            
            # Update eigen decomposition if needed (we do it every generation for simplicity with small dim)
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-20)
            
            # Sample
            ys = []
            for _ in range(lam):
                z = rng.standard_normal(dim)
                # Transform to correlated normal
                y = eigvecs @ (eigvals ** 0.5 * z)
                ys.append(y)
            ys = np.array(ys)
            
            # Evaluate
            xs = (m + sigma * ys).T  # shape (lam, dim)
            # Clip to box
            xs = np.clip(xs, -5, 5)
            
            fs = np.array([func(x) for x in xs])
            calls_used += lam
            
            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            ys_sorted = ys[order]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = xs_sorted[0].copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Update mean
            m_new = m + sigma * sum(w[i] * ys_sorted[i] for i in range(mu))
            
            # Update evolution path pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (m_new - m)
            
            # Update evolution path ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                 (eigvecs.T @ (eigvals ** -0.5) @ (m_new - m) / sigma)
            
            # Update covariance matrix
            delta_sigma = (np.linalg.norm(ps) / chi_n - 1) ** 2 - 1
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used - dim) / (dim + 1))) < (1 + 1.4 / dim) * chi_n
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                yi = (m_new - m) / sigma - sigma * (eigvecs @ (eigvals ** -0.5 * (eigvecs.T @ (ys_sorted[i] / sigma)))) 
                # Actually, simpler: the weighted sum of outer products of the selected y's
                # But we need the difference between new and old mean in the search space
                # The standard update is:
                # C = (1 - c1 - cmu) * C + c1 * pc*pc^T + cmu * sum(wi * (yi - m_y_diff) * (yi - m_y_diff)^T)
                # Where yi are the selected search steps
                pass
            
            # Let's use the standard formula more carefully
            # The search steps are ys_sorted[i]
            # The mean update uses the weighted sum of these steps
            # The covariance update uses the outer products of the steps, adjusted for the mean shift
            
            # Recompute rank-mu term
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = ys_sorted[i]
                rank_mu += w[i] * np.outer(y_i, y_i)
            
            # Correction for the mean shift in the rank-mu term?
            # In standard CMA-ES, the rank-mu term is sum(wi * (xi - m_new) * (xi - m_new)^T) / sigma^2
            # But since xi = m_old + sigma * yi, and m_new = m_old + sigma * sum(wi * yi),
            # then (xi - m_new)/sigma = yi - sum(wj * yj)
            # Let's compute this properly
            mean_y = sum(w[i] * ys_sorted[i] for i in range(mu))
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = ys_sorted[i] - mean_y
                rank_mu += w[i] * np.outer(diff, diff)
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * rank_mu
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals_new, eigvecs_new = np.linalg.eigh(C)
            if np.any(eigvals_new < 1e-10):
                C = C + 1e-10 * np.eye(dim)
            
            m = m_new
            # Clip m to box to keep it valid
            m = np.clip(m, -5, 5)
            
            return no_improve >= max_no_improve or sigma < 1e-8 or sigma > 1e8

        # Main loop
        while calls_used < budget:
            # Check if we should restart
            if no_improve >= max_no_improve or sigma < 1e-8 or sigma > 1e8:
                # Restart: reinitialize around best_x
                m = best_x.copy()
                C = np.eye(dim)
                sigma = 1.0
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                no_improve = 0
                continue
            
            # Do one generation
            should_restart = do_generation()
            
            if should_restart:
                # Restart in next iteration
                continue

        return best_x, best_f
# EVOLVE-BLOCK-END
