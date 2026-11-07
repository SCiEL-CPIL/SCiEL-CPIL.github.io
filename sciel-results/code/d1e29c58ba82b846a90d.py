"""Multi-objective optimizer on WFG7: Tchebycheff Decomposition with Evolution Strategy."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer using Tchebycheff decomposition.
    
    Strategy:
    1. Define a set of weight vectors that cover the objective space.
    2. For each weight vector, run a local Evolution Strategy (ES) to minimize the 
       Tchebycheff scalarization function: max_i (w_i * (f_i(x) - z_i^min)).
    3. Use a global archive of non-dominated solutions to guide the search (local search 
       around existing good solutions).
    4. Use a deadline guard to stop before the 300s limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0  # 5s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        # Use a fixed seed for internal randomness to ensure reproducibility, 
        # but the evaluator sets np.random.seed(seed) before calling.
        # We should use np.random for the main search to align with evaluator's seed?
        # The evaluator calls np.random.seed(seed). 
        # If we use np.random.default_rng(42), we ignore the evaluator's seed.
        # The task says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)...".
        # It's safer to use the global np.random state if possible, or just use a local rng.
        # Given the previous program used default_rng(42) and got 0.91, let's stick to local rng 
        # for stability, but maybe mixing in global random is better?
        # Let's use np.random directly to respect the evaluator's seed if we want variation, 
        # but for a deterministic optimizer, a local seed is often better for consistent performance.
        # However, "seeds 0 and 1" implies the evaluator wants to test different initial conditions.
        # If I use a fixed internal seed, both seeds 0 and 1 will produce the same result.
        # The score is the mean over seeds. If they are identical, it's fine.
        # But usually, diversity helps. Let's use np.random for the initial population 
        # and mutations to leverage the evaluator's seed.
        
        used = 0
        archive_X = []
        archive_F = []

        def eval_point(x):
            nonlocal used
            if used >= self.budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            return x, f

        def update_archive(x, f):
            # Add to archive if not dominated by any existing point
            # Simple O(N) check for small N
            is_dominated = False
            for i, f_arch in enumerate(archive_F):
                if np.all(f_arch <= f) and np.any(f_arch < f):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Remove any archive points dominated by the new point
                new_archive_X = []
                new_archive_F = []
                for i, f_arch in enumerate(archive_F):
                    if not (np.all(f <= f_arch) and np.any(f < f_arch)):
                        new_archive_X.append(archive_X[i])
                        new_archive_F.append(f_arch)
                
                new_archive_X.append(x)
                new_archive_F.append(f)
                
                return new_archive_X, new_archive_F
            else:
                return archive_X, archive_F

        # --- Phase 1: Tchebycheff Decomposition ---
        # Create weight vectors. For 3 objectives, we can use a grid.
        # Let's use 5 points per dimension -> 5^3 = 125 weight vectors? 
        # That might be too many for 200 budget if each needs multiple evals.
        # Let's use 3 points per dimension -> 3^3 = 27 weight vectors.
        # Or a simplex grid.
        
        # Let's use a set of weights that cover the simplex.
        # Weights w such that sum(w) = 1, w >= 0.
        # For 3 objectives, we can use:
        # (1,0,0), (0,1,0), (0,0,1) - vertices
        # (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5) - edges
        # (1/3, 1/3, 1/3) - center
        # Plus some intermediate points.
        
        weights = []
        # Generate a grid on the simplex
        n_grid = 4 # 4 points per axis -> (4-1)^2 = 9 points? No, number of points in simplex with n_grid steps is C(n_grid + n_obj - 1, n_obj - 1)
        # For n_obj=3, n_grid=4 (steps), number of points = C(4+2, 2) = C(6,2) = 15.
        # Let's use n_grid=5 -> C(7,2) = 21 points.
        n_grid = 5
        for w1 in np.linspace(0, 1, n_grid):
            for w2 in np.linspace(0, 1 - w1, n_grid):
                w3 = 1 - w1 - w2
                if w3 >= 0:
                    weights.append([w1, w2, w3])
        
        # Convert to numpy array
        W = np.array(weights)
        
        # Normalize weights? Tchebycheff usually uses normalized weights.
        # Sum of weights is already 1.
        
        # Initial z-vector (ideal point). Start with a large value or first eval.
        # We need an initial population to get a rough idea of the front.
        
        # Initialize with random points to establish z_min
        n_init = 10
        X_init = lo + (hi - lo) * np.random.rand(n_init, self.dim)
        F_init = []
        for x in X_init:
            res = eval_point(x)
            if res is None: break
            _, f = res
            F_init.append(f)
        
        if not F_init:
            return []
            
        F_init = np.array(F_init)
        z_min = np.min(F_init, axis=0)
        z_max = np.max(F_init, axis=0)
        
        # Update archive with initial points
        for i in range(len(F_init)):
            archive_X, archive_F = update_archive(X_init[i], F_init[i])

        # --- Phase 2: Local Search per Weight Vector ---
        # For each weight vector, perform a local search (e.g., CMA-ES or simple gradient-free local search)
        # minimizing the Tchebycheff scalarization:
        # g(x, w, z) = max_i ( w_i * (f_i(x) - z_i) )
        
        # Since we have limited budget, we'll do a simple coordinate-wise local search 
        # or a small population ES for each weight vector.
        
        # Let's use a simple "hill climbing" with random perturbations for each weight.
        # Or better, use a small population of candidates for each weight, evolving them.
        
        # Given 200 budget and ~21 weights, we have ~9 evals per weight.
        # This is tight. 
        # Alternative: Use a global population where each individual is assigned a weight.
        
        # Let's switch to a population-based approach where each individual has a weight.
        # This is similar to MOEA/D.
        
        pop_size = 30
        # Assign weights to individuals. If pop_size > len(W), repeat weights.
        # If pop_size < len(W), select a subset of weights.
        # Let's use all 21 weights and add 9 more random weights to make 30.
        # Or just use 21 individuals.
        
        n_ind = min(len(W), 25) # Keep it manageable
        # Select first n_ind weights
        W_sub = W[:n_ind]
        
        # Initialize population at the ideal point (or center) + noise
        # Start from the best initial point? Or center of bounds?
        # Center is safe.
        center = (lo + hi) / 2
        
        X_pop = []
        F_pop = []
        for i in range(n_ind):
            # Start near center with some noise
            x = center + (hi - lo) * 0.1 * np.random.randn(self.dim)
            x = np.clip(x, lo, hi)
            res = eval_point(x)
            if res is None: break
            _, f = res
            X_pop.append(x)
            F_pop.append(f)
            
            # Update archive
            archive_X, archive_F = update_archive(x, f)
            
            # Update z_min
            z_min = np.minimum(z_min, f)
            
        if not X_pop:
            return []
            
        X_pop = np.array(X_pop)
        F_pop = np.array(F_pop)
        
        # Define scalarization function
        def tchebycheff(x, w, z):
            f = np.array(problem(x), dtype=np.float64)
            # Note: problem is called inside, but we need to count evals.
            # This function is used for comparison, but we need to be careful with eval counting.
            # For local search, we will manually manage evals.
            return np.max(w * (f - z))
        
        # Local search loop
        # For each individual, try to improve it using a local search method.
        # We'll use a simple "random restart" local search or a small ES.
        
        # Let's use a simple coordinate descent with random steps.
        # For each individual i:
        #   For a few steps:
        #     Generate a perturbation
        #     If it improves the scalarized value, accept.
        
        n_local_steps = 5
        step_size = 0.1 # Relative to range
        
        while used < self.budget and time.time() < deadline:
            improved = False
            for i in range(len(X_pop)):
                if used >= self.budget or time.time() > deadline:
                    break
                    
                w = W_sub[i]
                x_best = X_pop[i]
                f_best = F_pop[i]
                
                # Try to improve x_best
                for _ in range(n_local_steps):
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    # Generate neighbor
                    # Use a Gaussian perturbation scaled by the remaining range
                    noise = np.random.randn(self.dim) * step_size * (hi - lo)
                    x_new = x_best + noise
                    x_new = np.clip(x_new, lo, hi)
                    
                    # Check if this point is already in archive? 
                    # If yes, we might skip evaluation to save budget? 
                    # But we don't have a hash. Let's just evaluate.
                    
                    res = eval_point(x_new)
                    if res is None: break
                    _, f_new = res
                    
                    # Calculate scalarized values
                    # g(x, w, z) = max(w * (f - z))
                    # Note: z_min should be updated as we find better points.
                    # However, using a fixed z_min from initialization might be okay for the first pass.
                    # Let's use the current global z_min.
                    
                    g_best = np.max(w * (f_best - z_min))
                    g_new = np.max(w * (f_new - z_min))
                    
                    if g_new < g_best:
                        x_best = x_new
                        f_best = f_new
                        improved = True
                        
                        # Update global z_min if this point is better in any objective
                        z_min = np.minimum(z_min, f_new)
                        
                        # Update archive
                        archive_X, archive_F = update_archive(x_best, f_best)
                        
                        # If we found a new best, maybe reduce step size?
                        # step_size *= 0.8
                    else:
                        # If not improved, maybe try a smaller step?
                        # For now, just continue.
                        pass
                
                # Update population
                X_pop[i] = x_best
                F_pop[i] = f_best
            
            # If no improvement in the whole round, we might be stuck.
            # We can restart some individuals or increase diversity.
            if not improved:
                # Randomly restart 20% of the population
                n_restart = max(1, int(0.2 * len(X_pop)))
                for _ in range(n_restart):
                    if used >= self.budget or time.time() > deadline:
                        break
                    idx = np.random.randint(len(X_pop))
                    x_new = lo + (hi - lo) * np.random.rand(self.dim)
                    res = eval_point(x_new)
                    if res is None: break
                    _, f_new = res
                    X_pop[idx] = x_new
                    F_pop[idx] = f_new
                    archive_X, archive_F = update_archive(x_new, f_new)
                    z_min = np.minimum(z_min, f_new)
            
            # Break if we've used enough budget or time
            if used > self.budget * 0.9:
                break

        # --- Phase 3: Polishing ---
        # Take the best non-dominated points from the archive and try to polish them.
        # We can use a simple local search on the original objectives.
        
        # Select top 5 non-dominated points from archive
        # (They are all non-dominated by construction of update_archive, but we can pick the ones with best sum)
        
        if len(archive_F) > 0:
            F_arch = np.array(archive_F)
            sums = np.sum(F_arch, axis=1)
            # Pick indices with lowest sums
            top_k = min(5, len(archive_F))
            top_indices = np.argsort(sums)[:top_k]
            
            for idx in top_indices:
                if used >= self.budget or time.time() > deadline:
                    break
                
                x_start = archive_X[idx]
                f_start = archive_F[idx]
                
                # Simple local search to minimize sum of objectives
                x_best = x_start.copy()
                f_best = f_start.copy()
                sum_best = np.sum(f_best)
                
                for _ in range(10): # 10 steps
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    noise = np.random.randn(self.dim) * 0.01 * (hi - lo)
                    x_try = x_best + noise
                    x_try = np.clip(x_try, lo, hi)
                    
                    res = eval_point(x_try)
                    if res is None: break
                    _, f_try = res
                    sum_try = np.sum(f_try)
                    
                    if sum_try < sum_best:
                        x_best = x_try
                        f_best = f_try
                        sum_best = sum_try
                        archive_X, archive_F = update_archive(x_best, f_best)
                        z_min = np.minimum(z_min, f_best)
        
        # Return the archive (though return value is ignored, it's good practice)
        return archive_X

# EVOLVE-BLOCK-END
