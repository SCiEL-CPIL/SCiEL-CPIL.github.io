"""Island Model CMA-ES with Adaptive Population Size for Katsuura f23."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Optimizes Katsuura f23 using an Island Model of CMA-ES instances with adaptive population sizes.
    1. Initialize multiple CMA-ES islands with different starting points.
    2. Each island adapts its population size (lambda) to balance exploration and exploitation.
    3. Periodic migration teleports the worst island to the best global solution to escape deep local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        lower = -5.0
        upper = 5.0
        
        calls_made = 0
        x_global_best = None
        f_global_best = np.inf
        
        def check_deadline():
            return time.time() > deadline
            
        def evaluate(x):
            nonlocal calls_made, x_global_best, f_global_best
            x_clipped = np.clip(x, lower, upper)
            fx = func(x_clipped)
            calls_made += 1
            if fx < f_global_best:
                f_global_best = fx
                x_global_best = x_clipped.copy()
            return x_clipped, fx

        # --- Initialize Islands ---
        n_islands = 4
        islands = []
        
        for i in range(n_islands):
            if calls_made >= budget or check_deadline():
                break
                
            # Start each island at a different random point
            x0 = rng.uniform(lower, upper, size=dim)
            x0, f0 = evaluate(x0)
            
            # Initialize CMA-ES state for this island
            island = {
                'x_mean': x0.copy(),
                'sigma': 1.0,
                'C': np.eye(dim),
                'p_sigma': np.zeros(dim),
                'p_c': np.zeros(dim),
                'best_x': x0.copy(),
                'best_f': f0,
                'lambda': 4,       # Start with small population
                'no_improve': 0,   # Stagnation counter
                'last_best_f': f0  # To track improvement
            }
            islands.append(island)
        
        if len(islands) == 0:
            return x_global_best, f_global_best
            
        # Migration parameters
        migration_interval = 5  # Generations between migrations
        max_lambda = 16
        min_lambda = 4
        stagnation_threshold = 5
        
        total_generations = 0
        max_generations = 1000 # Safety cap on iterations, budget is the real limit
        
        while calls_made < budget and not check_deadline() and total_generations < max_generations:
            total_generations += 1
            any_progress_global = False
            
            # Run one generation on each island
            for idx, island in enumerate(islands):
                if calls_made >= budget or check_deadline():
                    break
                    
                lam = island['lambda']
                mu = max(2, lam // 2)
                
                # Recalculate CMA-ES parameters based on current lam/mu
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w / np.sum(w)
                mu_eff = 1.0 / np.sum(w**2)
                
                c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
                d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
                c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
                c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
                c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
                
                x_mean = island['x_mean']
                sigma = island['sigma']
                C = island['C']
                p_sigma = island['p_sigma']
                p_c = island['p_c']
                
                # Generate population
                ys = []
                fs = []
                for i in range(lam):
                    if calls_made >= budget or check_deadline():
                        break
                        
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)
                        L = np.eye(dim)
                    
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (L @ z)
                    y, fy = evaluate(y)
                    ys.append(y)
                    fs.append(fy)
                
                if len(ys) == 0:
                    break
                    
                # Sort by function value
                indices = np.argsort(fs)
                ys_sorted = [ys[i] for i in indices]
                fs_sorted = [fs[i] for i in indices]
                
                # Update island best and check for improvement
                if fs_sorted[0] < island['best_f']:
                    island['best_f'] = fs_sorted[0]
                    island['best_x'] = ys_sorted[0].copy()
                    island['no_improve'] = 0
                    any_progress_global = True
                else:
                    island['no_improve'] += 1
                    
                # Update mean
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += w[i] * ys_sorted[i]
                x_mean = y_w
                
                # Update sigma control
                # Note: L from the last Cholesky is used for the inverse transform in cumulation
                # Ideally we should use the L corresponding to the population generation, 
                # but since C changes slowly, using the last L is standard approximation in some implementations.
                # However, to be safer, we recompute L for the update step if C changed significantly? 
                # For dim=5, Cholesky is cheap. We already have L from the loop. 
                # The y_w is in the transformed space? No, y_w is in x-space.
                # We need to transform y_w back to the standard normal space for p_sigma/p_c updates.
                # y = x_mean_old + sigma * L @ z  =>  z = L^-1 @ (y - x_mean_old) / sigma
                # Here x_mean_old is the mean before update.
                
                # We need L corresponding to the C used for sampling.
                # The loop above used the current C.
                try:
                    L_update = np.linalg.cholesky(C)
                except:
                    L_update = np.eye(dim)

                y_sigma = (L_update.T @ ((x_mean - island['x_mean']) / sigma))
                p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                
                # Hessian potential
                h_sigma = 1.0 if (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * math.sqrt(mu_eff + 1 - 1)) else 0.0
                
                sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / math.sqrt(mu_eff) - 1))
                
                # Update covariance control
                y_c = (L_update.T @ ((x_mean - island['x_mean']) / sigma))
                p_c = (1 - c_c) * p_c + math.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 - c_1 - c_mu + delta_h) * C
                C += c_1 * (np.outer(p_c, p_c) + delta_h * np.eye(dim))
                for i in range(mu):
                    # Transform selected points back to z-space
                    y_i = (L_update.T @ ((ys_sorted[i] - island['x_mean']) / sigma))
                    C += c_mu * w[i] * np.outer(y_i, y_i)
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    if np.min(eigvals) < 1e-10:
                        C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                
                # Update island state
                island['x_mean'] = x_mean
                island['sigma'] = sigma
                island['C'] = C
                island['p_sigma'] = p_sigma
                island['p_c'] = p_c
                
                # --- Adaptive Population Size Logic ---
                if island['no_improve'] >= stagnation_threshold:
                    if island['lambda'] < max_lambda:
                        # Increase population size to explore more broadly
                        island['lambda'] = min(max_lambda, island['lambda'] * 2)
                    else:
                        # If already at max, maybe reset sigma or do nothing
                        # Resetting sigma to 1.0 can help escape if C is collapsed
                        island['sigma'] = max(island['sigma'], 0.5) 
                    island['no_improve'] = 0
                elif island['no_improve'] == 0 and island['lambda'] > min_lambda:
                    # Made progress, try to reduce population size to refine
                    # Only reduce if we've made progress consistently? 
                    # Simple approach: reduce by 1 step if at max, or keep.
                    # Let's be conservative: only reduce if lambda is high and we just improved.
                    if island['lambda'] > min_lambda:
                         island['lambda'] = max(min_lambda, island['lambda'] // 2)
            
            # --- Migration ---
            if total_generations % migration_interval == 0 and len(islands) > 1:
                # Sort islands by best fitness
                sorted_islands = sorted(range(len(islands)), key=lambda i: islands[i]['best_f'])
                
                best_idx = sorted_islands[0]
                worst_idx = sorted_islands[-1]
                
                # If the worst island is significantly worse, teleport it
                # Threshold: if worst is > 10% worse than best (relative to scale) or just always if distinct
                # Given Katsuura's scale, absolute difference is tricky.
                # Let's always migrate if the worst island has been stagnant for a while or is just the worst.
                # To avoid overwriting a potentially good island that is just slightly worse,
                # we check if the worst island's best_f is much larger than the global best.
                
                if islands[worst_idx]['best_f'] > islands[best_idx]['best_f'] + 1e-6:
                    # Teleport worst island to best island's best solution
                    x_migrate = islands[best_idx]['best_x'] + rng.normal(0, 0.2, size=dim)
                    x_migrate = np.clip(x_migrate, lower, upper)
                    
                    # Evaluate the migrant point to get a valid starting point for the new island
                    # This consumes 1 call.
                    if calls_made < budget and not check_deadline():
                        _, f_migrate = evaluate(x_migrate)
                        
                        # Reset the worst island's CMA-ES state to start from this new point
                        islands[worst_idx]['x_mean'] = x_migrate.copy()
                        islands[worst_idx]['sigma'] = 1.0
                        islands[worst_idx]['C'] = np.eye(dim)
                        islands[worst_idx]['p_sigma'] = np.zeros(dim)
                        islands[worst_idx]['p_c'] = np.zeros(dim)
                        islands[worst_idx]['best_x'] = x_migrate.copy()
                        islands[worst_idx]['best_f'] = f_migrate
                        islands[worst_idx]['no_improve'] = 0
                        islands[worst_idx]['lambda'] = min_lambda # Start fresh with small pop

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
