"""Bootstrap spawn: CMA-ES approach for BBOB f20 (Schwefel).

Strategy:
  1. Use Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
  2. CMA-ES is highly effective for multimodal, non-convex problems like Schwefel.
  3. It adapts the sampling distribution (mean and covariance) to the landscape.
  4. Strict wall-clock deadline guard with safety margin.
"""
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # Safety margin before 330s kill

        call_count = [0]

        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_check():
            return time.time() < deadline - 5.0

        def budget_check():
            return call_count[0] < budget - 5

        def cma_es(start_mean, initial_sigma, max_iter):
            """Run CMA-ES from start_mean."""
            nonlocal best_x, best_f
            
            # CMA-ES parameters
            n = dim
            lam = max(4, 4 + int(3 * np.log(n)))  # Population size, e.g., 4+3*ln(5) ~ 10
            mu = lam // 2  # Selection pressure
            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Learning rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            
            # Damping
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            
            # Initial state
            mean = start_mean.copy()
            sigma = initial_sigma
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            
            # Evaluate initial mean
            f_mean = make_call(mean)
            if f_mean is None: return
            update_best(mean, f_mean)
            
            for it in range(max_iter):
                if not budget_check() or not time_check():
                    break
                
                # Sample points
                points = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    # Transform z using B and D
                    y = B @ (D * z)
                    x = mean + sigma * y
                    x = np.clip(x, -5.0, 5.0)
                    points.append(x)
                
                # Evaluate points
                fvals = []
                for x in points:
                    f = make_call(x)
                    if f is None: break
                    fvals.append(f)
                    update_best(x, f)
                
                if len(fvals) < lam:
                    break
                
                # Sort points by fitness
                idx = np.argsort(fvals)
                points = [points[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Update mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * points[i]
                
                # Update sigma
                # Calculate p_sigma
                # We need the old mean and the old B, D
                old_mean = mean.copy()
                old_B = B.copy()
                old_D = D.copy()
                
                # y_plus = (new_mean - old_mean) / (sigma * ||new_mean - old_mean||) ? No, standard formula:
                # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) / sigma * (new_mean - old_mean)
                # Wait, the term is usually (new_mean - old_mean) / sigma
                diff = new_mean - old_mean
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (diff / sigma)
                
                # Update p_c
                # p_c = (1 - c_c) p_c + h_sigma * sqrt(c_c (2 - c_c) mu_eff) * B^{-1} D^{-1} (new_mean - old_mean)
                # h_sigma = (||p_sigma|| / chi_n) < c_1 + 1
                norm_p_sigma = np.linalg.norm(p_sigma)
                h_sigma = 1.0 if norm_p_sigma / chi_n < (1 + 2 / (n + 1)) * c_1 else 0.0
                
                # B^{-1} D^{-1} is the inverse of the transformation
                # y = B D z => z = D^{-1} B^{-1} y
                # So B^{-1} D^{-1} is not exactly right, it's (B D)^{-1} = D^{-1} B^{-1}
                # But we have B and D. B is orthogonal.
                # We need to transform the difference back to the z-space
                # diff_z = D^{-1} B^T diff
                diff_z = (diff / old_D) @ old_B.T  # This is D^{-1} B^T diff
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff_z
                
                # Update Covariance Matrix C
                # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_c (2 - c_c) C) + c_mu sum w_i y_i y_i^T
                # y_i = B D z_i, so y_i y_i^T = B D z_i z_i^T D B^T
                # We can update C directly in the B,D basis or in the original space.
                # Standard update:
                # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_c (2 - c_c) C)
                #    + c_mu sum w_i (B D z_i) (B D z_i)^T
                # To be efficient, we often update C in the eigenbasis or just do the matrix ops.
                # For n=5, direct matrix ops are fine.
                
                # First term: rank-1 update from p_c
                # Note: p_c is in the z-space (eigenbasis). The term p_c p_c^T needs to be transformed back?
                # No, the standard CMA-ES update formula for C is:
                # C <- (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_c (2 - c_c) C) + c_mu sum w_i y_i y_i^T
                # Here p_c is the cumulative step in the *transformed* space?
                # Actually, the standard formulation keeps p_c in the original space?
                # Let's check the standard pseudo-code.
                # p_c is updated with B^{-1} D^{-1} (mean - old_mean).
                # So p_c is in the z-space.
                # The term p_c p_c^T is in the z-space.
                # But C is in the original space.
                # So we must transform p_c p_c^T back: B D (p_c p_c^T) D B^T ?
                # Yes.
                
                # Let's compute the rank-1 term in original space
                # y_pc = B @ (D * p_c)
                y_pc = old_B @ (old_D * p_c)
                rank1_term = np.outer(y_pc, y_pc)
                
                # Sum of outer products of y_i
                sum_yy = np.zeros((n, n))
                for i in range(mu):
                    # y_i = B @ (D * z_i)
                    # We didn't store z_i, but we have x_i.
                    # x_i = old_mean + sigma * y_i => y_i = (x_i - old_mean) / sigma
                    y_i = (points[i] - old_mean) / sigma
                    sum_yy += weights[i] * np.outer(y_i, y_i)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (rank1_term + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum_yy
                
                # Ensure C is symmetric
                C = (C + C.T) / 2.0
                
                # Update eigen decomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    # Clip small eigenvalues
                    eigvals = np.clip(eigvals, 1e-12, None)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                except np.linalg.LinAlgError:
                    # If decomposition fails, reset C to identity
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update mean
                mean = new_mean
                
                # Convergence check
                if np.linalg.norm(mean - old_mean) < 1e-10 and sigma < 1e-10:
                    break

        # Multi-start configuration
        start_points = []
        
        # 1. Origin
        start_points.append(np.zeros(dim))
        
        # 2. Standard Schwefel minimum
        start_points.append(np.full(dim, 4.209687))
        
        # 3. Random points
        for _ in range(5):
            start_points.append(rng.uniform(-5, 5, dim))

        # Shuffle to mix
        rng.shuffle(start_points)
        
        n_restarts = len(start_points)
        for i in range(n_restarts):
            if not budget_check() or not time_check():
                break
            
            start_x = start_points[i].copy()
            
            # Adaptive max_iter based on remaining budget
            remaining_budget = budget - call_count[0]
            # CMA-ES costs ~ lambda calls per iteration
            lam = max(4, 4 + int(3 * np.log(dim)))
            max_iter = max(2, int(remaining_budget / (lam * 1.5)))
            
            cma_es(start_x, initial_sigma=2.0, max_iter=max_iter)
            
            # If we found a good point, do a local refinement
            if best_x is not None and call_count[0] < budget - 20 and time.time() < deadline - 10:
                # Refine around best point with smaller sigma
                cma_es(best_x.copy(), initial_sigma=0.5, max_iter=10)

        return best_x, best_f
