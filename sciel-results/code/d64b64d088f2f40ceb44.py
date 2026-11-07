"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using NSGA-II with CMA-ES adaptive step sizes."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Population size
        # NSGA-II typically works well with pop_size ~ 50-100
        # Budget 200. Let's use pop_size = 50.
        # Generations: 200 / 50 = 4 generations? That's very few.
        # Better: Smaller population, more generations.
        # Let's try pop_size = 20. Generations = 10.
        # Or pop_size = 10. Generations = 20.
        # For 5 dimensions, pop_size 20 is reasonable.
        N_POP = 20
        n_gen = self.budget // N_POP
        
        # Initialize population
        X = np.zeros((N_POP, self.dim))
        for i in range(N_POP):
            X[i] = lo + span * rng.random(self.dim)
        
        F = np.array([problem(x) for x in X])
        all_F = [F]
        used = N_POP
        
        # CMA-ES state
        # Mean
        mean = X.mean(axis=0)
        # Step size
        sigma = 0.5 * np.min(span)
        # Covariance matrix
        C = np.eye(self.dim)
        # Inverse covariance
        C_inv = np.eye(self.dim)
        # Eigen decomposition for C
        # We need to sample from N(mean, sigma^2 * C)
        # Using Cholesky
        L = np.linalg.cholesky(C)
        
        # Weights for CMA-ES update
        mu = N_POP // 2
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(N_POP / 2 + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # Evolution path for covariance
        p_c = np.zeros(self.dim)
        # Evolution path for sigma
        p_sigma = np.zeros(self.dim)
        # Norm of C
        norm_C = np.sqrt(np.trace(C))
        # Chi-square expectation
        chi_n = np.sqrt(self.dim * (1 - 1. / (4. * self.dim) - 1. / (21. * self.dim**2)))
        
        # Non-dominated sorting and crowding distance helpers
        def non_dominated_sort(F):
            """
            Returns a list of fronts, where each front is a list of indices.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # Dominates
            n_dominates = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if all(F[i] <= F[j]) and any(F[i] < F[j]):
                        S[i].append(j)
                    # Check if j dominates i
                    if all(F[j] <= F[i]) and any(F[j] < F[i]):
                        n_dominates[i] += 1
                        
            current_front = [i for i in range(n) if n_dominates[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of indices in F.
            """
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            n = len(indices)
            dists = np.zeros(n)
            F_sub = F[indices]
            
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = np.argsort(F_sub[:, m])
                # Boundary points get infinity
                dists[sorted_idx[0]] = np.inf
                dists[sorted_idx[-1]] = np.inf
                # Normalize
                range_val = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if range_val > 1e-10:
                    for i in range(1, n - 1):
                        dists[sorted_idx[i]] += (F_sub[sorted_idx[i+1], m] - F_sub[sorted_idx[i-1], m]) / range_val
            return dists

        for gen in range(n_gen):
            if time.time() > deadline:
                break
            
            # Create offspring
            X_off = np.zeros((N_POP, self.dim))
            for i in range(N_POP):
                # Sample from CMA-ES distribution
                z = rng.standard_normal(self.dim)
                x = mean + sigma * (L @ z)
                x = np.clip(x, lo, hi)
                X_off[i] = x
            
            F_off = np.array([problem(x) for x in X_off])
            all_F.append(F_off)
            used += N_POP
            
            # Combine parents and offspring
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            
            # Non-dominated sorting
            fronts = non_dominated_sort(F_all)
            
            # Select next population
            X_next = []
            F_next = []
            for front in fronts:
                if len(X_next) + len(front) <= N_POP:
                    # Take all
                    for idx in front:
                        X_next.append(X_all[idx])
                        F_next.append(F_all[idx])
                else:
                    # Need to truncate
                    needed = N_POP - len(X_next)
                    # Calculate crowding distance for this front
                    dists = crowding_distance(F_all, front)
                    # Sort by distance (descending)
                    sorted_front = sorted(front, key=lambda i: dists[front.index(i)], reverse=True)
                    for idx in sorted_front[:needed]:
                        X_next.append(X_all[idx])
                        F_next.append(F_all[idx])
                    break
            
            X = np.array(X_next)
            F = np.array(F_next)
            
            # Update CMA-ES state based on the best mu individuals in the new population
            # We pick the first mu non-dominated individuals (front 0) or just the first mu if front 0 is small
            # For simplicity and stability, let's use the first mu individuals from the sorted fronts (best first)
            selected_indices = []
            for front in fronts:
                for idx in front:
                    selected_indices.append(idx)
                    if len(selected_indices) == mu:
                        break
                if len(selected_indices) == mu:
                    break
            
            if len(selected_indices) < mu:
                # If we don't have enough, pad with random from population
                while len(selected_indices) < mu:
                    selected_indices.append(rng.integers(len(X_all)))
            
            # Calculate new mean
            y = np.array([X_all[i] for i in selected_indices])
            mean = np.sum(weights[:, np.newaxis] * y, axis=0)
            
            # Update evolution paths
            # We need the inverse of L (Cholesky of C)
            L_inv = np.linalg.inv(L)
            d_mean = (mean - X.mean(axis=0)) / sigma # Wait, mean was updated. Use old mean?
            # Standard CMA-ES uses the previous mean.
            # Let's store old_mean before updating
            # Actually, let's restructure:
            # 1. Generate offspring
            # 2. Select parents for CMA-ES update (best mu from combined pop)
            # 3. Update mean, paths, sigma, C
            
            # Re-do CMA-ES update logic properly
            # We need the old mean for the path update
            # Let's store old_mean before the loop iteration start? No, it's updated every gen.
            # Let's just use the current mean as the reference for the path update? 
            # Standard algorithm:
            # m_k is the new mean.
            # p_sigma = (1-c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma)mu_eff) * C^{-1/2} (m_k - m_{k-1}) / sigma
            # So we need m_{k-1}.
            
            # Let's store m_prev
            m_prev = mean # This is the mean BEFORE this generation's update? 
            # No, `mean` variable holds the mean used to generate offspring.
            # So `mean` is m_{k-1}.
            # `y` are the selected parents.
            # m_new = sum(w_i * y_i)
            
            m_new = np.sum(weights[:, np.newaxis] * y, axis=0)
            
            # Update p_sigma
            diff = m_new - mean
            if np.linalg.norm(diff) > 1e-10:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv @ (diff / sigma))
            
            # Check if sigma adaptation is needed
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (self.dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update p_c
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (self.dim + 1)) * chi_n else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (L_inv @ (diff / sigma))
            
            # Update C
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1-h_sigma) c_c (2-c_c) C) + c_mu sum(w_i (y_i - m_new)(y_i - m_new)^T)
            C = (1 - c_1 - c_mu) * C
            if h_sigma == 0:
                C += c_1 * c_c * (2 - c_c) * C
            C += c_1 * np.outer(p_c, p_c)
            
            for i in range(mu):
                diff_i = y[i] - m_new
                C += c_mu * weights[i] * np.outer(diff_i, diff_i)
            
            # Ensure C is positive definite and update L
            # Symmetrize
            C = (C + C.T) / 2
            # Eigen decomposition to handle numerical errors
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small eigenvalues
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(C)
            
            # Update mean
            mean = m_new

        return X

# EVOLVE-BLOCK-END
