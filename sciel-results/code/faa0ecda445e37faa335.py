"""Multi-objective optimizer for RE21 using surrogate-assisted multi-start local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer using Kriging surrogate and multi-start local search.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def get_pareto_front(F):
            """
            Returns indices of non-dominated points in F.
            """
            n = len(F)
            if n == 0:
                return []
            
            F_arr = np.array(F)
            is_pareto = np.ones(n, dtype=bool)
            
            for i in range(n):
                if not is_pareto[i]:
                    continue
                # Check if i is dominated by any j
                # F[j] <= F[i] and F[j] != F[i]
                # Vectorized check:
                # j dominates i if all(F[j] <= F[i]) and any(F[j] < F[i])
                # We can check if there exists j such that:
                # (F[j] <= F[i]).all(axis=1) & (F[j] < F[i]).any(axis=1)
                
                # Exclude i itself
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                
                # Conditions for j to dominate i
                le = np.all(F_arr[mask] <= F_arr[i], axis=1)
                lt = np.any(F_arr[mask] < F_arr[i], axis=1)
                
                if np.any(le & lt):
                    is_pareto[i] = False
                    
            return np.where(is_pareto)[0]

        def kriging_predict(X_train, y_train, X_new):
            """
            Simple Kriging (Gaussian Process) prediction.
            X_train: (n, d)
            y_train: (n,)
            X_new: (m, d)
            Returns: mean (m,), var (m,)
            """
            n, d = X_train.shape
            
            # Normalize inputs for stability
            X_mean = X_train.mean(axis=0)
            X_std = X_train.std(axis=0) + 1e-8
            Xn_train = (X_train - X_mean) / X_std
            Xn_new = (X_new - X_mean) / X_std
            
            # Squared Euclidean distance matrix
            # (n, 1, d) - (1, n, d) -> (n, n, d) -> sum -> (n, n)
            diff = Xn_train[:, np.newaxis, :] - Xn_train[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            
            # Correlation matrix K
            # Using exponential kernel
            # Avoid log(0) issues
            sq_dist = np.maximum(sq_dist, 1e-12)
            K = np.exp(-0.5 * sq_dist)
            np.fill_diagonal(K, 1.0)
            
            # Add nugget for numerical stability
            K += 1e-8 * np.eye(n)
            
            # Cross-correlation between train and new
            diff_new = Xn_train[:, np.newaxis, :] - Xn_new[np.newaxis, :, :]
            sq_dist_new = np.sum(diff_new ** 2, axis=2)
            sq_dist_new = np.maximum(sq_dist_new, 1e-12)
            K_star = np.exp(-0.5 * sq_dist_new) # (n, m)
            
            # Solve for weights
            # beta = K^-1 y
            try:
                beta = np.linalg.solve(K, y_train)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(K, y_train, rcond=None)[0]
                
            # Mean prediction
            mu = K_star.T @ beta
            
            # Variance prediction
            # Var = 1 - K_star.T K^-1 K_star
            # This can be unstable. We use a simpler heuristic for EI if needed, 
            # but let's try to compute it.
            # Actually, for EI we just need an estimate of uncertainty.
            # A common approximation is to use the distance to nearest neighbor.
            # Let's stick to a robust variance estimate:
            # var = 1 - K_star.T @ beta_cross? No.
            # Let's just use a simple distance-based uncertainty for EI, 
            # or assume a constant variance if K is ill-conditioned.
            
            # For robustness and speed, let's use a simplified EI based on distance.
            # Distance from X_new to X_train
            dists = np.sqrt(np.sum((Xn_new[:, np.newaxis, :] - Xn_train[np.newaxis, :, :]) ** 2, axis=2))
            min_dists = np.min(dists, axis=1)
            
            # Heuristic uncertainty: increases with distance
            # Normalize distance to [0, 1] roughly
            max_d = np.sqrt(d)
            uncertainty = np.clip(min_dists / max_d, 0, 1)
            # Scale uncertainty to be comparable to y range
            y_range = np.max(y_train) - np.min(y_train)
            if y_range == 0:
                y_range = 1.0
            var = (uncertainty ** 2) * (y_range ** 2) * 0.1 + 1e-6
            
            return mu, var

        # --- Main Algorithm ---

        # 1. Initial Sampling (Latin Hypercube or Random)
        n_init = min(20, budget // 2)
        
        # Simple random init for simplicity and speed
        for _ in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            evaluate(x)

        # 2. Iterative Surrogate-Assisted Local Search
        # We will perform a fixed number of "generations" of local search
        # Each generation:
        #   a. Build surrogate
        #   b. Select next start point using EI
        #   c. Run local search from that point
        
        # Budget allocation:
        # ~10 local searches, each with ~15-20 evals?
        # Total 200. Init 20. Remaining 180.
        # 12 local searches * 15 evals = 180.
        
        n_local_searches = 12
        budget_per_local = max(10, (budget - used) // n_local_searches)
        
        # Store best points found by local search to use as seeds for future EI?
        # Or just use all points.
        
        for ls_iter in range(n_local_searches):
            if used >= budget or time.time() > deadline:
                break
            
            # Build Surrogate
            if len(all_X) > 5:
                X_train = np.array(all_X)
                F_train = np.array(all_F)
                
                # Predict for a set of candidate points
                # We generate a pool of candidates to choose from
                n_candidates = 50
                X_cands = lo + (hi - lo) * np.random.random((n_candidates, dim))
                
                # For each objective, calculate EI
                # We want to find points that are likely to be non-dominated
                # A common approach is to use the minimum of the EIs or a combined metric.
                # Let's use the minimum of the normalized EIs.
                
                # Normalize objectives for EI calculation
                F_min = F_train.min(axis=0)
                F_max = F_train.max(axis=0)
                F_range = F_max - F_min
                F_range[F_range == 0] = 1.0
                
                F_norm = (F_train - F_min) / F_range
                
                best_ei = np.zeros(n_candidates)
                best_ei[:] = np.inf
                
                for k in range(n_obj):
                    y_k = F_train[:, k]
                    mu_k, var_k = kriging_predict(X_train, y_k, X_cands)
                    
                    # EI for minimization
                    # f_best is the current best value
                    f_best = np.min(y_k)
                    
                    # Z = (f_best - mu) / sqrt(var)
                    # EI = (f_best - mu) * Phi(Z) + sqrt(var) * phi(Z)
                    # Note: This is for maximization of (f_best - f).
                    # For minimization, we want f < f_best.
                    # Let's define Y = f_best - f. We want Y > 0.
                    # Mean of Y = f_best - mu
                    # Var of Y = var
                    # EI = E[Y | Y > 0] * P(Y > 0)
                    # = (f_best - mu) * Phi((f_best - mu) / sqrt(var)) + sqrt(var) * phi((f_best - mu) / sqrt(var))
                    
                    sqrt_var = np.sqrt(np.maximum(var_k, 1e-8))
                    z = (f_best - mu_k) / sqrt_var
                    ei_k = (f_best - mu_k) * norm.cdf(z) + sqrt_var * norm.pdf(z)
                    
                    # Normalize EI by range to combine objectives
                    ei_norm = ei_k / (F_range[k] + 1e-8)
                    
                    # We want high EI. 
                    # To find a point that is good in ALL objectives, we take the minimum EI across objectives.
                    # This is a conservative choice.
                    best_ei = np.minimum(best_ei, ei_norm)
                    
                # Select candidate with highest "minimum EI"
                # Add some randomness to avoid getting stuck
                # Softmax selection
                # Or just pick the top few and randomize
                idx_best = np.argmax(best_ei)
                x_start = X_cands[idx_best]
                
            else:
                # Not enough data for surrogate, use random
                x_start = lo + (hi - lo) * np.random.random(dim)
                
            # Run Local Search from x_start
            # Use Nelder-Mead on a scalarized objective (Tchebycheff or Weighted Sum)
            # To maintain diversity, we can rotate weights or use Tchebycheff.
            # Tchebycheff: max w_i * (f_i(x) - z_i^*)
            # Let's use a weighted sum with random weights for simplicity, 
            # but bias towards the current Pareto front gaps.
            
            # Simple strategy: Use random weights for each local search
            weights = np.random.dirichlet(np.ones(n_obj))
            
            # Local search function
            def obj_func(x):
                # Clip x
                x = np.clip(x, lo, hi)
                f = problem(x)
                # Weighted sum
                return np.dot(weights, f)
                
            # Initial guess
            x0 = x_start
            # We already have f for x_start? No, we just picked it.
            # We need to evaluate x_start first to get f0 for NM? 
            # NM in scipy doesn't require f0, it evaluates.
            # But we want to count evaluations.
            # SciPy's minimize does not allow us to hook into its evaluations easily 
            # to count towards our budget and record in all_X/all_F.
            
            # So we must implement a simple Nelder-Mead ourselves or use a fixed step local search.
            # Given the time constraint and the need to track budget strictly, 
            # a simple coordinate descent or fixed-step Nelder-Mead is safer.
            
            # Let's use a simple "Hill Climbing" with random perturbations.
            # It's less efficient than NM but easier to control.
            # Or, we can run NM but wrap the function to count.
            
            # Let's try a custom simple Nelder-Mead.
            # Actually, let's just do a local random search with shrinking step.
            # It's robust and easy to budget.
            
            x_cur = x_start.copy()
            f_cur = evaluate(x_cur)
            
            if used >= budget or time.time() > deadline:
                continue
                
            # Local search loop
            local_budget = min(budget_per_local, budget - used)
            step_size = 0.1
            
            for _ in range(local_budget):
                if used >= budget or time.time() > deadline:
                    break
                    
                # Propose a new point
                # Random direction
                direction = np.random.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-8)
                
                x_new = x_cur + step_size * (hi - lo) * direction
                x_new = np.clip(x_new, lo, hi)
                
                f_new = evaluate(x_new)
                
                # Check if improvement (in terms of weighted sum or domination)
                # Use weighted sum for decision
                score_cur = np.dot(weights, f_cur)
                score_new = np.dot(weights, f_new)
                
                if score_new < score_cur:
                    x_cur = x_new
                    f_cur = f_new
                    # Increase step slightly if improving? Or keep?
                    # step_size = min(step_size * 1.1, 0.2)
                else:
                    # Shrink step
                    step_size *= 0.7
                    if step_size < 1e-4:
                        break
                        
        return all_X
# EVOLVE-BLOCK-END
