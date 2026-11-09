import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        # Safety margin: The evaluator kills at 330s, but the task says 300s.
        # Let's aim to finish well before 300s to be safe, say 290s.
        deadline = t0 + 290.0
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def try_func(x):
            nonlocal best_f, best_x, calls
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        
        # Phase 1: Initial exploration
        # Use a mix of center and random points to ensure we find the ridge region
        # 20 points is a good trade-off for 500 budget.
        n_init = 20
        for i in range(n_init):
            if i == 0:
                x = np.zeros(n)
            else:
                x = rng.uniform(-5.0, 5.0, n)
            try_func(x)
            
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES
        # CMA-ES is the gold standard for BBOB f13 (Sharp Ridge)
        
        # Parameters
        lambda_ = 8  # Population size
        mu = lambda_ // 2  # Number of parents
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initial state
        mean = best_x.copy()
        # Initial sigma: 1.0 is a safe default for [-5, 5] domain
        # We can scale it based on the variance of initial samples if needed, but 1.0 is robust.
        sigma = 1.0
        
        B = np.eye(n)
        D = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Main CMA-ES loop
        while calls < budget and time.time() < deadline:
            # Check if we have enough budget for a full generation
            if calls + lambda_ > budget:
                # If not, we can break or do a partial generation? 
                # Standard CMA-ES requires full generations. 
                # If budget is tight, we might just stop here.
                # However, we can check if we can at least do one more step with reduced lambda?
                # For simplicity and stability, if we can't do a full generation, we stop.
                # But wait, we might have 1-7 calls left. Let's just stop to avoid partial updates.
                break
            
            # Generate offspring
            offspring = []
            for i in range(lambda_):
                # Sample z ~ N(0, I)
                z = rng.standard_normal(n)
                # Transform
                x = mean + sigma * (B @ (D @ z))
                # Clip to bounds
                x = np.clip(x, -5.0, 5.0)
                f = try_func(x)
                offspring.append((f, x))
                
            if calls >= budget:
                break
                
            # Sort offspring
            offspring.sort(key=lambda x: x[0])
            
            # Update mean
            new_mean = np.sum([weights[i] * offspring[i][1] for i in range(mu)], axis=0)
            
            # Update evolution paths
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D @ (new_mean - mean)))
            # pc = (1 - cc) * pc + sqrt(cc * (2 - cc) * mu_eff) * (new_mean - mean) / sigma
            
            # The formula for ps involves the inverse transformation of the difference.
            # diff = new_mean - mean
            # We need to transform diff back to the standard normal space?
            # Actually, the standard CMA-ES update for ps is:
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * (B^T @ D^-1 @ (new_mean - mean)) / sigma
            # Since C = B D^2 B^T, the "inverse" transform of a vector v in x-space to z-space is B^T D^-1 v.
            
            diff = new_mean - mean
            # Transform diff to z-space
            # D is diagonal with positive entries (singular values)
            # D_inv = 1 / D
            # B_inv = B^T (since B is orthogonal)
            diff_z = (B.T @ (diff / D)) / sigma
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * diff_z
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * diff / sigma
            
            # Update covariance matrix
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (||pc||^2 / n - 1) * I) + cmu * sum(weights * (xi - mean)(xi - mean).T)
            
            # We maintain C via B and D to avoid explicit matrix inversion or Cholesky every step?
            # Actually, standard implementation updates C explicitly or via eigendecomposition of the rank-mu update.
            # Let's use the explicit update of C and then eigendecompose to update B and D.
            # This is O(n^3) per generation. For n=5, this is negligible.
            
            # Calculate the rank-mu update term
            diff_x = np.array([offspring[i][1] - new_mean for i in range(mu)])
            rank_mu_update = np.sum([weights[i] * np.outer(diff_x[i], diff_x[i]) for i in range(mu)])
            
            # We need the current C.
            C = (B @ (D**2) @ B.T)
            
            # Update C
            C_new = (1 - c1 - cmu) * C 
            C_new += c1 * (np.outer(pc, pc) + (np.dot(pc, pc) / n - 1) * np.eye(n))
            C_new += cmu * rank_mu_update
            
            # Ensure symmetry and positive definiteness (numerical stability)
            C_new = (C_new + C_new.T) / 2.0
            
            # Eigendecomposition to get new B and D
            # C_new = B_new @ diag(D_new^2) @ B_new.T
            eigvals, eigvecs = np.linalg.eigh(C_new)
            
            # Clip eigenvalues to be positive
            eigvals = np.maximum(eigvals, 1e-12)
            
            B = eigvecs
            D = np.sqrt(eigvals)
            
            # Update sigma
            # sigma = sigma * exp(cs / (5 * n) * (||ps|| / chi_n - 1))
            sigma = sigma * np.exp(cs / (5 * n) * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            mean = new_mean
            
            # Optional: If sigma becomes very small, we might be converged.
            if sigma < 1e-6:
                break

        # Phase 3: Final Polish
        # Use L-BFGS-B from the best point found by CMA-ES to squeeze out the last bit of precision.
        # This is safe because CMA-ES gets us close to the global minimum, and L-BFGS-B is good for smooth functions.
        if calls < budget - 5 and time.time() < deadline:
            def obj(x):
                return try_func(x)
            
            try:
                # Use a tight tolerance
                res = minimize(obj, best_x, method='L-BFGS-B', 
                              bounds=[(-5, 5)] * n,
                              options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 50})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
