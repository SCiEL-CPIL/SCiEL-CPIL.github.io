import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        
        # CMA-ES Parameters
        m = (lo + hi) / 2.0
        sigma = 0.5 * np.max(hi - lo)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Weights and effective population size
        lam = 10
        mu = lam // 2
        
        # Weights: log-linear
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Archive for non-dominated points
        archive_X = []
        archive_F = []
        
        evals = 0
        rng = np.random.default_rng(0)
        
        while evals < self.budget and time.time() < deadline:
            # Generate lambda samples
            B = rng.multivariate_normal(np.zeros(dim), C, size=lam)
            X = m + sigma * B
            X = np.clip(X, lo, hi)
            
            # Evaluate
            F = np.array([problem(x) for x in X])
            evals += lam
            
            # Update Archive
            for i in range(lam):
                archive_X.append(X[i])
                archive_F.append(F[i])
            
            # Find Non-Dominated Set in Archive
            # Since archive grows, we need to filter it.
            # For efficiency, we can just keep the current generation's ND set + previous ND set?
            # Or just compute ND on the whole archive? Archive size <= 200.
            # 200x200 dominance check is fast.
            
            archive_X = np.array(archive_X)
            archive_F = np.array(archive_F)
            
            # Compute Non-Dominated Indices
            nd_indices = self._get_non_dominated_indices(archive_F)
            
            # Select mu parents
            if len(nd_indices) <= mu:
                parent_indices = nd_indices
            else:
                # Crowding Distance
                cd = self._crowding_distance(archive_F[nd_indices])
                # Sort by crowding distance descending
                sorted_idx = np.argsort(-cd)
                parent_indices = nd_indices[sorted_idx[:mu]]
            
            # Extract parents
            X_parents = archive_X[parent_indices]
            
            # Update CMA-ES
            # Mean update
            m_new = np.sum(w[:, np.newaxis] * X_parents, axis=0)
            
            # Path updates
            # Need C^-1/2
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L).T
            except:
                C_inv_sqrt = np.eye(dim)
            
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (m_new - m) / sigma
            
            # h_sigma condition
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2) + 1 / (8 * dim**4))
            h_sigma = 1 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (evals/lam + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (m_new - m) / sigma
            
            # Covariance update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_c * (2 - c_c) + delta_h) * C - c_c * (2 - c_c) * np.outer(p_c, p_c)
            
            # Rank-mu update
            for i in range(mu):
                y_i = C_inv_sqrt @ (X_parents[i] - m) / sigma
                C += 2 * w[i] * c_c * (2 - c_c) * np.outer(y_i, y_i)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definiteness
            try:
                np.linalg.cholesky(C)
            except:
                # Reset C if it becomes invalid
                C = np.eye(dim)
            
            m = m_new
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Recompute Cholesky for next iteration
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L).T
            except:
                C_inv_sqrt = np.eye(dim)
                
            # Prune archive to prevent memory bloat (optional, but good practice)
            # Keep only non-dominated points in archive
            archive_X = archive_X[nd_indices]
            archive_F = archive_F[nd_indices]

        return archive_X

    def _get_non_dominated_indices(self, F):
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return np.where(~is_dominated)[0]

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 2:
            return np.ones(n) * np.inf
        cd = np.zeros(n)
        for m in range(F.shape[1]):
            idx = np.argsort(F[:, m])
            cd[idx[0]] = np.inf
            cd[idx[-1]] = np.inf
            for i in range(1, n - 1):
                cd[idx[i]] += (F[idx[i+1], m] - F[idx[i-1], m]) / (F[idx[-1], m] - F[idx[0], m] + 1e-9)
        return cd
