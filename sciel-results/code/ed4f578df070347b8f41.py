"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-stage optimizer: LHS bracketing + CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget or time.time() - start_time > time_limit:
                return np.inf
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            calls_used += 1
            try:
                val = func(x_clipped)
                return float(val)
            except Exception:
                return np.inf

        def record(x, f):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: 3-level Latin Hypercube for rapid bracketing
        # 3 levels per dim, but we only have ~20 evals so use a stratified approach
        # Use 20 points: center + axis directions + random stratified
        num_phase1 = min(20, budget // 5)
        
        # Generate stratified points using a simple LHS-like approach
        # Divide [-4, 4] into 3 intervals per dim
        levels = [-4.0, 0.0, 4.0]
        
        # Create all combinations but sample from them
        # For dim=5, 3^5 = 243 combos, pick 20 diverse ones
        # Use a space-filling approach: stratify by taking one from each "cell"
        
        phase1_points = []
        
        # Always include origin and axis points
        phase1_points.append(np.zeros(dim))
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 2.0
                phase1_points.append(p)
        
        # Add random stratified points
        while len(phase1_points) < num_phase1:
            # Random point but ensure it's not too close to existing
            p = rng.uniform(-4.0, 4.0, dim)
            min_dist = min(np.linalg.norm(p - q) for q in phase1_points) if phase1_points else np.inf
            if min_dist > 1.0 or len(phase1_points) < 10:
                phase1_points.append(p)
            else:
                # Just accept if we've tried enough
                phase1_points.append(p)
                if len(phase1_points) >= num_phase1:
                    break
        
        # Deduplicate
        unique_points = [phase1_points[0]]
        for p in phase1_points[1:]:
            if all(np.linalg.norm(p - q) > 0.1 for q in unique_points):
                unique_points.append(p)
        phase1_points = unique_points[:num_phase1]
        
        for p in phase1_points:
            if calls_used >= budget or time.time() - start_time > time_limit:
                break
            f = safe_call(p)
            if np.isfinite(f):
                record(p, f)
        
        if calls_used >= budget or time.time() - start_time > time_limit:
            return best_x, best_f

        # Phase 2: CMA-ES from best point with small sigma
        # Implement a lightweight CMA-ES
        x_mean = best_x.copy()
        sigma = 0.5  # small initial sigma for exploitation
        lambda_ = 4 * dim  # population size
        mu = lambda_ // 2  # number of parents
        
        # CMA-ES weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (sqrt2mu_eff - 1) / ((dim + 2)**2 + mu_eff)) if (sqrt2mu_eff := np.sqrt(mu_eff)) else 2 * (2 - 1) / ((dim + 2)**2 + mu_eff)
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)  # C^(-1/2)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Cholesky factor of C
        try:
            B = np.linalg.cholesky(C)
            D = np.sqrt(np.diag(B @ B.T))
            C_inv_sqrt = np.linalg.inv(B)
        except:
            C_inv_sqrt = np.eye(dim)
            D = np.ones(dim)
        
        max_iter = 50
        no_improve = 0
        threshold = 1e-10
        
        for iteration in range(max_iter):
            if calls_used >= budget or time.time() - start_time > time_limit:
                break
            
            # Generate lambda_ candidates
            candidates = []
            for _ in range(lambda_):
                z = rng.standard_normal(dim)
                # Apply C^(-1/2) to get proper scaling
                x_cand = x_mean + sigma * (C_inv_sqrt @ z)
                candidates.append(x_cand)
            
            # Evaluate all candidates
            evals = []
            for x_cand in candidates:
                f_cand = safe_call(x_cand)
                evals.append(f_cand)
            
            # Sort by fitness
            sorted_idx = np.argsort(evals)
            
            # Update best
            best_idx = sorted_idx[0]
            if evals[best_idx] < best_f:
                best_f = evals[best_idx]
                best_x = candidates[best_idx].copy()
            
            # Check for improvement
            if evals[best_idx] < x_mean_f - threshold if 'x_mean_f' in dir() else True:
                no_improve = 0
            else:
                no_improve += 1
            
            x_mean_f = evals[best_idx]
            
            # Check convergence
            if no_improve > 10:
                break
            
            # Update mean (weighted recombination)
            x_old = x_mean.copy()
            x_mean_new = np.zeros(dim)
            for i, idx in enumerate(sorted_idx[:mu]):
                x_mean_new += weights[i] * candidates[idx]
            
            # Update sigma
            x_mean = x_mean_new
            
            # Update evolution paths
            # Compute C^(-1/2) again for proper scaling
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(B @ B.T))
                C_inv_sqrt = np.linalg.inv(B)
            except:
                C_inv_sqrt = np.eye(dim)
                D = np.ones(dim)
            
            # p_sigma update
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ (x_mean - x_old)) / sigma
            
            # h_sigma check
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2) + 1 / (8 * dim**4))
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (iteration + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            # p_c update
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * (C_inv_sqrt @ (x_mean - x_old)) / sigma
            
            # Update C
            delta_h = (h_sigma - 1) * cc * (2 - cc)
            rank_1 = np.outer(p_c, p_c) + delta_h * np.eye(dim)
            rank_mu = np.zeros((dim, dim))
            for i, idx in enumerate(sorted_idx[:mu]):
                y_i = (C_inv_sqrt @ (candidates[idx] - x_old)) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c1 * delta_h + cmu * (c1 + cmu - 1)) * C + c1 * rank_1 + cmu * rank_mu
            
            # Ensure positive definiteness
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                pass
            
            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update C_inv_sqrt
            try:
                B = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(B)
            except:
                C_inv_sqrt = np.eye(dim)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
