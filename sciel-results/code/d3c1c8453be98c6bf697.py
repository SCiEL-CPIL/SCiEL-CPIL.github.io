"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MO-CMA-ES: Multi-objective CMA-ES with adaptive covariance matrix."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # CMA-ES parameters
        lambda_ = 4 * int(np.ceil(3 + np.log(dim)))  # population size
        mu = lambda_ // 2
        # Weighted recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        # Step size and covariance adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Initialize distribution
        center = (lo + hi) / 2
        sigma = 0.25 * (hi - lo).mean()
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial population: sample from N(center, sigma^2 * I)
        def sample(n):
            z = np.random.standard_normal((n, dim))
            return center + sigma * z @ C_sqrt.T
        
        # Clip to bounds
        def clip(x):
            return np.clip(x, lo, hi)
        
        # Evaluate initial population
        X = clip(sample(lambda_))
        F = np.array([problem(x) for x in X])
        used = lambda_
        
        # Pareto ranking function
        def pareto_rank(F):
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            remaining = set(range(n))
            rank = 0
            while remaining:
                # Find non-dominated in remaining
                front = []
                for i in remaining:
                    dominated = False
                    for j in remaining:
                        if i != j:
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                dominated = True
                                break
                    if not dominated:
                        front.append(i)
                for i in front:
                    ranks[i] = rank
                    remaining.remove(i)
                rank += 1
            return ranks
        
        # Crowding distance for diversity
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            cd = np.zeros(len(indices))
            for m in range(n_obj):
                vals = F[indices, m]
                mn, mx = vals.min(), vals.max()
                if mx - mn < 1e-12:
                    cd[:] = np.inf
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, len(order) - 1):
                    cd[order[k]] += (vals[order[k+1]] - vals[order[k-1]]) / (mx - mn)
            return cd
        
        # Selection: pick mu best based on rank + crowding
        def select_best(X, F, mu):
            n = len(F)
            ranks = pareto_rank(F)
            # Group by rank
            best_indices = []
            max_rank = ranks.max()
            for r in range(max_rank + 1):
                group = np.where(ranks == r)[0]
                if len(best_indices) + len(group) <= mu:
                    best_indices.extend(group.tolist())
                else:
                    # Need to select some from this group
                    need = mu - len(best_indices)
                    cd = crowding_distance(F, group)
                    order = np.argsort(-cd)
                    best_indices.extend(group[order[:need]].tolist())
                    break
            return np.array(best_indices[:mu])
        
        # Main loop
        while used < budget:
            if time.time() > deadline:
                break
                
            # Select best mu individuals
            best_idx = select_best(X, F, mu)
            X_best = X[best_idx]
            F_best = F[best_idx]
            
            # Weighted mean
            m_new = np.sum(weights[:, None] * X_best, axis=0)
            
            # Update evolution paths
            sigma_old = sigma
            d = (m_new - center) / (sigma * np.sqrt(mu_eff))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (lambda_ - mu))) / chi_n) < (1.4 + 2 / (dim + 1))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - center) / sigma
            center = m_new
            
            # Update covariance
            delta_sig = 1 + 2 * min(1, (mu_eff - 1) / (dim + 2)) * (np.linalg.norm(p_c) ** 2 / dim - 1) / c_c
            rank_mu = np.sum(weights[:, None] * (X_best - m_new) @ (X_best - m_new).T, axis=0)
            C = (1 + c_1 * (delta_sig - 1) + c_mu) * C - c_1 * np.outer(p_c, p_c) - c_mu * rank_mu / sigma_old ** 2
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 1e-10:
                C += 1e-8 * np.eye(dim)
            
            # Update sigma
            sigma *= np.exp((c_sigma / 5) * (np.linalg.norm(p_sigma) / chi_n - 1))
            # Keep sigma in reasonable range
            sigma = np.clip(sigma, 1e-6, 0.5 * (hi - lo).max())
            
            # Update matrix square roots (SVD for stability)
            U, s, Vt = np.linalg.svd(C)
            C_sqrt = (U * np.sqrt(np.clip(s, 0, None))) @ Vt
            C_inv = np.linalg.inv(C)
            C_inv_sqrt = (Vt.T * np.sqrt(np.clip(1.0 / s, 0, None))) @ U.T
            
            # Generate offspring
            X_off = clip(sample(lambda_))
            F_off = np.array([problem(x) for x in X_off])
            used += lambda_
            
            # Combine parent and offspring
            X = np.vstack([X, X_off])
            F = np.vstack([F, F_off])
            
            # Select best lambda_ for next generation
            keep_idx = select_best(X, F, lambda_)
            X = X[keep_idx]
            F = F[keep_idx]
        
        return X
# EVOLVE-BLOCK-END
