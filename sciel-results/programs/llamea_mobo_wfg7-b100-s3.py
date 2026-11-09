import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s. We aim to finish well before.
        deadline = start_time + 300.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([3.0, 5.0, 7.0])
        
        X_data = []
        F_data = []
        
        def evaluate(x):
            # Ensure x is within bounds
            x = np.clip(x, lo, hi)
            # problem(x) handles clipping again per contract, but we do it here for safety
            f = np.array(problem(x))
            X_data.append(x.copy())
            F_data.append(f.copy())
            return f

        # 1. Initialization
        # Use a larger initial population to cover the space well
        # Budget is 200. Let's use lambda=40 for CMA-ES.
        # This leaves 160 evaluations for ~4 generations.
        # Maybe smaller lambda for more generations?
        # WFG7 is deceptive. Let's try lambda=30.
        # 200 / 30 = 6.6 generations.
        # Let's try lambda=20 for ~10 generations.
        # Or lambda=50 for 4 generations?
        # CMA-ES usually works well with lambda ~ 4 + 3*ln(n). For n=12, lambda ~ 15-20.
        # Let's use lambda=30 to balance exploration and exploitation.
        
        lam = 30
        mu = lam // 2
        
        # Initial mean: center of the box
        mean = (lo + hi) / 2.0
        
        # Initial step size: 1/4 of the range, or smaller?
        # WFG7 bounds are typically [0, 1] or similar.
        # Let's use 0.25 * (hi - lo)
        step_size = 0.25 * (hi - lo)
        
        # Covariance matrix: Identity
        C = np.eye(self.dim)
        
        # Weights for selection (standard CMA-ES weights)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # Evolution path for covariance
        p_c = np.zeros(self.dim)
        # Evolution path for step size
        p_sigma = np.zeros(self.dim)
        
        # Normalization factor for step size adaptation
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        
        # Non-dominated sorting function
        def non_dominated_sort(F):
            """
            Returns list of lists, each containing indices of non-dominated points in that front.
            F: (N, M) array of objective values.
            """
            N = F.shape[0]
            M = F.shape[1]
            dominated = [0] * N
            domination_count = [0] * N
            fronts = [[]]
            
            for p in range(N):
                for q in range(N):
                    if p == q:
                        continue
                    # Check if p dominates q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        dominated[q] += 1
                        fronts[0].append(p)
                    # Check if q dominates p
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        dominated[p] += 1
            
            # Actually, standard NSGA-II sort is better.
            # Let's implement a simple version that just returns the first front (non-dominated set)
            # For CMA-ES selection, we often just pick the best 'mu' points based on hypervolume or crowding.
            # Simpler: Pick the 'mu' points with the best weighted sum or hypervolume contribution.
            # Given the tight budget, simple non-dominated selection of the best 'mu' points is sufficient.
            
            # Find indices of non-dominated points
            non_dom = []
            for i in range(N):
                is_dom = False
                for j in range(N):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    non_dom.append(i)
            
            # If we have more than mu non-dominated points, we need to select the best mu.
            # We can use hypervolume contribution or just random selection among them.
            # Better: Use a simple scalarization or distance to ideal point to pick 'mu' from non_dom.
            # Or just take the first 'mu' non-dominated points.
            # To be more robust, let's rank them by their distance to the current best known Pareto front approximation (or ideal point).
            
            if len(non_dom) > mu:
                # Calculate distance to ideal point (min of each objective in current set)
                ideal = np.min(F[non_dom], axis=0)
                dists = np.sum((F[non_dom] - ideal)**2, axis=1)
                # Sort by distance (lower is better)
                sorted_idx = np.argsort(dists)
                selected = [non_dom[i] for i in sorted_idx[:mu]]
            else:
                # If fewer than mu non-dominated, pad with dominated points with best scalarization
                # Or just repeat the best ones? No, CMA-ES needs distinct points.
                # If we don't have enough, we take all non-dominated and fill the rest with the best dominated points.
                selected = non_dom.copy()
                remaining_needed = mu - len(selected)
                if remaining_needed > 0:
                    # Find dominated points
                    dom_points = [i for i in range(N) if i not in selected]
                    # Sort by sum of objectives (simple scalarization)
                    scores = np.sum(F[dom_points], axis=1)
                    sorted_dom = np.argsort(scores)
                    for i in sorted_dom[:remaining_needed]:
                        selected.append(dom_points[i])
                        
            return selected

        # Main Loop
        eval_count = 0
        while eval_count < self.budget:
            if time.time() > deadline - 10:
                break
            
            # Generate offspring
            # Use Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except:
                L = np.eye(self.dim)
            
            Y = np.zeros((lam, self.dim))
            for i in range(lam):
                z = np.random.randn(self.dim)
                Y[i] = mean + step_size * (L @ z)
            
            # Evaluate offspring
            F_offspring = np.zeros((lam, self.n_obj))
            for i in range(lam):
                if time.time() > deadline - 5:
                    break
                f = evaluate(Y[i])
                F_offspring[i] = f
                eval_count += 1
                
                if eval_count >= self.budget:
                    break
            
            # If we ran out of budget mid-generation, stop
            if eval_count >= self.budget:
                break
            
            # Selection: Pick best 'mu' points
            selected_idx = non_dominated_sort(F_offspring)
            
            # Update Mean
            y_w = np.zeros((mu, self.dim))
            for i in range(mu):
                y_w[i] = Y[selected_idx[i]]
            
            # Transform y_w to search space (inverse Cholesky)
            # y_w = mean + step_size * L @ z
            # So z = inv(L) @ (y_w - mean) / step_size
            # We want to update mean in the original space.
            # mean_new = mean + step_size * L @ (w @ z)
            # This is equivalent to mean_new = sum(w_i * y_w_i)
            
            mean_old = mean.copy()
            mean = np.sum(w[:, np.newaxis] * y_w, axis=0)
            
            # Update Evolution Paths
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv(L) @ (mean - mean_old) / step_size
            delta_mean = (mean - mean_old) / step_size
            inv_L_delta = np.linalg.solve(L, delta_mean)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_L_delta
            
            # Check if p_sigma is too large
            norm_p_sigma = np.linalg.norm(p_sigma)
            if norm_p_sigma > (1.4 + c_sigma) * chi_n:
                # Skip covariance update for p_sigma effect? Or just clip?
                # Standard CMA-ES: if norm is too large, don't update C with p_sigma term
                pass
            else:
                # Update p_c
                # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * inv(L) @ (mean - mean_old) / step_size
                # Wait, p_c is in the same space as p_sigma?
                # Yes, p_c = (1-c_c)p_c + sqrt(...) * (mean - mean_old) / (step_size * L)
                # Actually, the formula is:
                # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (mean - mean_old) / (step_size * L)
                # Let's use the standard form:
                # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * inv_L_delta
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * inv_L_delta
                
                # Update Covariance C
                # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - (1 - c_c)^2) * I) + c_mu * sum(w_i * y_w_i y_w_i^T)
                # Where y_w_i are the normalized steps (z vectors)
                
                # Calculate the weighted sum of outer products of z vectors
                z_w = np.zeros((mu, self.dim))
                for i in range(mu):
                    z_w[i] = np.linalg.solve(L, (y_w[i] - mean_old) / step_size)
                
                # c_mu term
                H = np.zeros((self.dim, self.dim))
                for i in range(mu):
                    H += w[i] * np.outer(z_w[i], z_w[i])
                
                # c_1 term
                p_c_outer = np.outer(p_c, p_c)
                I = np.eye(self.dim)
                c1_term = c_1 * (p_c_outer + (1 - (1 - c_c)**2) * I)
                
                C = (1 - c_1 - c_mu) * C + c1_term + c_mu * H
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Add small regularization if needed
                C += 1e-10 * np.eye(self.dim)
            
            # Update Step Size
            # sigma_new = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
            sigma_factor = np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            step_size *= sigma_factor
            
            # Clip step size to reasonable bounds
            min_step = 1e-6
            max_step = 1.0 # Relative to bounds? No, step_size is absolute.
            # Let's clip to [1e-6, 1.0]
            step_size = np.clip(step_size, min_step, 1.0)
            
            # Re-cholesky for next iteration? No, C is updated, L will be recomputed at start of loop.

        return X_data[-1] if X_data else None
