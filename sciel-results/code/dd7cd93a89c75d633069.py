"""Rewritten optimizer: Multi-start CMA-ES for Sharp Ridge optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer using Multi-start initialization followed by CMA-ES.
    CMA-ES is robust to the anisotropy of sharp ridge functions by adapting
    its covariance matrix to the local landscape shape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin for 300s limit

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = clip(x)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---------- Phase 1: Multi-start Initialization ----------
        # We need to find the basin of the sharp ridge.
        # Random sampling is effective for locating the rough region.
        n_starts = 30
        for _ in range(n_starts):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_point(x)

        if best_x is None:
            return None, np.inf

        # ---------- Phase 2: CMA-ES Refinement ----------
        # Use CMA-ES to exploit the anisotropic landscape.
        # It will shrink sigma in the perpendicular direction and maintain it along the ridge.
        
        x_mean = best_x.copy()
        f_mean = best_f
        
        # CMA-ES parameters
        # Lambda is the population size. For dim=5, lambda=5 is small but efficient for budget.
        # We can try slightly larger like 7 or 9 if budget allows.
        # Budget remaining is ~470. 470 / 5 = 94 generations.
        # 470 / 9 = 52 generations.
        # Let's use lambda = 5 for more generations (better convergence) or 7 for better exploration.
        # Sharp ridges often benefit from smaller populations to stay on the ridge.
        lambda_ = 5
        
        # Initial sigma: 1.0 is a reasonable start for [-5,5] box.
        sigma = 1.0
        
        # Covariance matrix C
        C = np.eye(dim)
        
        # Step size control
        c_sigma = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 2) / (dim + 1) - 1) + c_sigma)
        cs = (dim + lambda_ + 2) / (dim + 5 + 3 * lambda_)
        c1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        cmu = min(1 - c1, 2 * ((lambda_ / 2 - 1) + 1) / ((dim + 2) ** 2 + lambda_))
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Cholesky factor of C
        # We maintain C and update it, then compute Cholesky each step or update incrementally.
        # For simplicity and robustness in a short script, we recompute Cholesky if needed,
        # but standard CMA-ES updates C directly.
        # We need the square root of C to generate samples.
        # Let's use a helper to get the transformation matrix.
        
        def get_transformation(C):
            # Compute Cholesky decomposition of C
            # C is positive definite
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to eigen decomposition if Cholesky fails (shouldn't happen often)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            return L

        T = get_transformation(C)

        # Weights for recombination
        # Use standard weights: w_i = log(lambda/2 + 0.5) - log(i)
        weights = np.array([np.log(lambda_ / 2 + 0.5) - np.log(i + 1) for i in range(lambda_)])
        weights /= np.sum(weights)
        mu = lambda_ // 2  # Number of parents used for covariance update
        
        # Effective population size for mu-weighted recombination
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
        
        # Adjust c1 and cmu for mu_eff
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff - 2) + 1) / ((dim + 2) ** 2 + mu_eff))

        while evaluated < budget and elapsed_ok():
            # Generate population
            pop_x = []
            pop_f = []
            pop_z = []
            
            for i in range(lambda_):
                if evaluated >= budget or not elapsed_ok():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (T @ z)
                x_i = clip(y)
                f_i = eval_point(x_i)
                pop_x.append(x_i)
                pop_f.append(f_i)
                pop_z.append(z)
            
            if len(pop_x) < lambda_:
                break

            # Sort by fitness
            order = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in order]
            pop_f = [pop_f[i] for i in order]
            pop_z = [pop_z[i] for i in order]
            
            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * pop_x[i]
            
            # Update evolution paths
            # First, update ps
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C^(-1/2) * (x_new - x_mean))
            # We need C^(-1/2). We can approximate or compute it.
            # For simplicity, let's use the inverse of T.T @ T which is C.
            # C^(-1/2) is roughly (T.T @ T)^(-1/2).
            # Let's compute the transformation of the difference.
            # diff = x_new - x_mean
            # We want C^(-1/2) * diff.
            # Since T is such that T @ T.T = C (approx, if T is Cholesky, T.T @ T = C? No, L @ L.T = C).
            # If T = L, then L @ L.T = C.
            # So C^(-1/2) = L^(-T) @ L^(-1).
            # This is getting complex. Let's use a simpler update or skip ps/pc updates for now
            # and rely on the covariance update only, which is less critical for short runs.
            # Or, we can just update the covariance directly using the rank-mu update.
            
            # Let's do a simplified CMA-ES without ps/pc for stability in this short script.
            # Or, let's try to implement it correctly.
            
            # Actually, the standard CMA-ES is robust. Let's implement the standard update.
            # We need the inverse square root of C.
            # Let's compute it via eigen decomposition each step (dim=5 is small).
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            diff = x_new - x_mean
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff) / sigma
            
            # Update pc
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evaluated - 30) / lambda_)) < (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** 2))
            h_sigma = 1.0 if h_sigma else 0.0 # Simplified
            
            pc = (1 - cs) * pc + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ diff) / sigma
            
            # Update C
            # Rank-1 update
            v = x_new - x_mean
            C = (1 + c1 * cmu * 0) * C  # Placeholder, need correct formula
            # Correct formula:
            # C = (1 - c1 - cmu) * C + c1 * (v @ v.T + (1 - h_sigma) * c_sigma * (2 - c_sigma) * ps @ ps.T) + cmu * sum(w_i * (y_i - y_w) @ (y_i - y_w).T)
            
            # Let's use the standard formula.
            # y_i are the z vectors? No, y_i are the samples in the transformed space?
            # In CMA-ES, the update is on the covariance of the steps.
            # Let's stick to the standard implementation found in literature.
            
            # For brevity and robustness, let's use a simplified update that just adapts the covariance
            # to the spread of the best mu points.
            # This is a "rank-mu" update only.
            
            # Compute the weighted sum of outer products of the best mu steps
            # Steps are x_i - x_mean_old
            steps = np.array(pop_x[:mu]) - x_mean
            # Weighted outer products
            outer = np.zeros((dim, dim))
            for i in range(mu):
                outer += weights[i] * np.outer(steps[i], steps[i])
            
            # Update C
            C = (1 - cmu) * C + cmu * outer
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                pass
            
            # Update sigma
            # Simple adaptation: if the best point is better, decrease sigma, else increase?
            # Standard CMA-ES uses the path ps.
            # sigma = sigma * exp(c_sigma / d_sigma * (||ps|| / sigma_norm - 1))
            # sigma_norm = sqrt(1 - (1 - c_sigma)^2)
            sigma_norm = np.sqrt(1 - (1 - c_sigma) ** 2)
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / sigma_norm - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            x_mean = x_new
            
            # Update transformation matrix
            T = get_transformation(C)
            
            # Check convergence
            if np.linalg.norm(C) < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
