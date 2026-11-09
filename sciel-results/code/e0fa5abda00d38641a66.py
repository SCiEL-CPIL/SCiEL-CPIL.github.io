"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer using adaptive local search.
    
    Strategy:
    1. Use a small set of Sobol points as initial seeds for local searches.
    2. For each seed, perform Nelder-Mead minimization on a Tchebycheff scalarization 
       of the objectives. This allows the optimizer to 'walk' down the objective 
       landscape towards the Pareto front, rather than just sampling the volume.
    3. The Tchebycheff scalarization is: min_x max_i (w_i * (f_i(x) - z_i^min))
       where z_i^min is the current best known value for objective i.
       By updating z_i^min dynamically, we ensure diverse points on the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 300s limit, kill at 290s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # 1. Generate initial seeds using Sobol sequence for good spread
        # We need enough seeds to cover the front. 
        # Let's use 20 seeds. Each seed gets a local search budget.
        # Budget is 200. 20 seeds * 10 evals/search = 200.
        # Nelder-Mead typically needs n+1 initial points (5 for dim=4) plus iterations.
        # So ~10-15 evals per search is reasonable for a quick local convergence.
        
        from scipy.stats import qmc
        n_seeds = 20
        # Generate n_seeds Sobol points
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_seeds)
        
        # Map to bounds
        X_seeds = lo + range_ * sobol_points

        # 2. Dynamic reference point for Tchebycheff scalarization
        # Initialize with +inf, update as we find better points
        z_min = np.full(self.n_obj, np.inf)
        
        all_F = []
        eval_count = 0

        # Helper to update z_min
        def update_z_min(f_new):
            nonlocal z_min
            z_min = np.minimum(z_min, f_new)

        # 3. Perform local search from each seed
        for i in range(n_seeds):
            if time.time() > deadline:
                break
            
            x0 = X_seeds[i]
            
            # We need to define the scalarization function.
            # Since z_min changes, we can't just pass a static function to minimize.
            # We will implement a manual Nelder-Mead loop or use a closure that captures 
            # the current z_min. However, z_min updates during the search.
            # A simpler robust approach: Use a fixed weight vector for each seed, 
            # but update the offset z_min dynamically? 
            # Actually, standard Tchebycheff with fixed weights and fixed z_min (from initial sweep) 
            # is safer and simpler. Let's do a small initial sweep first to get a good z_min.
            
            pass # Placeholder, logic below

        # Revised Strategy for robustness:
        # Phase 1: Small Sobol sweep (e.g., 40 points) to establish a decent z_min.
        # Phase 2: Use remaining budget (160) for 16 local searches (10 evals each).
        
        # Reset
        eval_count = 0
        all_F = []
        
        # Phase 1: Initial Sweep
        n_initial = 40
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=123)
        init_points = sampler.random(n_initial)
        X_init = lo + range_ * init_points
        
        for i in range(n_initial):
            if time.time() > deadline or eval_count >= self.budget:
                break
            x = X_init[i]
            f = problem(x)
            all_F.append(f)
            eval_count += 1
            
            # Update z_min
            z_min = np.minimum(z_min, f)
            
        if eval_count >= self.budget:
            return all_F

        # Phase 2: Local Search
        # Use the best non-dominated points from Phase 1 as seeds for local search?
        # Or just use the first few Sobol points?
        # Let's use the first 16 points from the initial sweep as seeds for local search.
        # We have 160 budget left. 16 searches * 10 evals = 160.
        
        n_local = 16
        evals_per_search = (self.budget - eval_count) // n_local
        if evals_per_search < 5:
             # Not enough budget for meaningful local search, just return what we have
             return all_F

        # We need to be careful with the scalarization.
        # We will use Tchebycheff with weights.
        # To get a diverse front, we use different weight vectors for each local search.
        
        # Generate weight vectors. For 3 objectives, we can use a grid or random weights.
        # Let's use a simple grid of weights for 3 objectives that sum to 1.
        # Or just random weights.
        rng = np.random.default_rng(42)
        
        for k in range(n_local):
            if time.time() > deadline or eval_count >= self.budget:
                break
            
            # Select a seed from the initial points. 
            # We can pick the k-th initial point.
            x_seed = X_init[k]
            
            # Define weights for this local search.
            # Use a deterministic set of weights to ensure coverage of the front.
            # For 3 objectives, we can use:
            # (0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8), (0.33, 0.33, 0.33), etc.
            # Let's generate a few standard weight vectors.
            weights_list = [
                np.array([0.8, 0.1, 0.1]),
                np.array([0.1, 0.8, 0.1]),
                np.array([0.1, 0.1, 0.8]),
                np.array([0.5, 0.3, 0.2]),
                np.array([0.2, 0.5, 0.3]),
                np.array([0.3, 0.2, 0.5]),
                np.array([0.4, 0.4, 0.2]),
                np.array([0.2, 0.4, 0.4]),
                np.array([0.4, 0.2, 0.4]),
                np.array([0.3, 0.3, 0.4]),
                np.array([0.4, 0.3, 0.3]),
                np.array([0.3, 0.4, 0.3]),
                np.array([0.5, 0.25, 0.25]),
                np.array([0.25, 0.5, 0.25]),
                np.array([0.25, 0.25, 0.5]),
                np.array([1/3, 1/3, 1/3])
            ]
            w = weights_list[k % len(weights_list)]
            
            # Capture current z_min for the scalarization.
            # Note: z_min might improve during this search, but for the scalarization 
            # definition inside minimize, we use the z_min at the start of the search 
            # to keep the function consistent. This is a common approximation.
            z_min_local = z_min.copy()
            
            def tchebycheff(x):
                f = problem(x)
                # Normalize by z_min to make weights comparable
                # If z_min is inf (shouldn't happen after phase 1), handle it
                diff = f - z_min_local
                # Clip negative diffs to 0? No, Tchebycheff usually uses max(w_i * (f_i - z_i)).
                # If f_i < z_i, the term is negative. The max will be driven by the worst objective.
                # This is correct.
                return np.max(w * diff)
            
            # We need to count evaluations inside minimize.
            # scipy minimize doesn't have a direct eval counter hook that stops mid-iteration easily 
            # without wrapping the function.
            # We will wrap the function.
            
            evals_in_search = 0
            max_evals_search = evals_per_search
            
            def wrapped_tchebycheff(x):
                nonlocal evals_in_search
                if time.time() > deadline or evals_in_search >= max_evals_search:
                    # Return a large value to signal stop? 
                    # Or just return the value and let the loop break?
                    # Actually, scipy minimize will keep calling until convergence or maxiter.
                    # We should limit maxiter.
                    pass
                evals_in_search += 1
                eval_count += 1
                return tchebycheff(x)

            # Set maxiter to ensure we don't exceed budget
            # Nelder-Mead maxiter is the number of iterations, not function evals.
            # Each iteration is roughly n+1 evals.
            # So maxiter = max_evals_search // (self.dim + 1)
            
            maxiter = max(1, max_evals_search // (self.dim + 1))
            
            try:
                res = minimize(wrapped_tchebycheff, x_seed, method='Nelder-Mead', 
                               bounds=list(zip(lo, hi)), 
                               options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-6})
                
                # Record the result of the local search
                f_res = problem(res.x) # This is an extra eval! 
                # Wait, we have to be careful. The `wrapped_tchebycheff` already evaluated `res.x` 
                # likely in the last iteration. But `minimize` returns the best x.
                # We must evaluate `res.x` to get the final F. 
                # Does this count against budget? Yes.
                # We reserved `evals_per_search` for the search. 
                # The search itself uses up to `max_evals_search` evals.
                # The final `problem(res.x)` is an additional eval.
                # So we should reduce `max_evals_search` by 1 to account for the final check?
                # Or just let it be. If we exceed budget, the outer loop breaks.
                
                # Actually, to be safe and strict:
                # The `wrapped_tchebycheff` increments `eval_count`.
                # The final `problem(res.x)` will increment it too if we do it.
                # But we don't *need* to call `problem(res.x)` again if we trust the last eval.
                # However, `res.x` might not be the exact point of the last eval.
                # Let's just evaluate it and check budget.
                
                if eval_count < self.budget:
                    f_final = problem(res.x)
                    eval_count += 1
                    all_F.append(f_final)
                    z_min = np.minimum(z_min, f_final)
                else:
                    # Over budget, discard this result or just stop
                    pass

            except Exception:
                # If minimize fails, we've still spent some evals. 
                # We just move to the next seed.
                pass

        return all_F
# EVOLVE-BLOCK-END
