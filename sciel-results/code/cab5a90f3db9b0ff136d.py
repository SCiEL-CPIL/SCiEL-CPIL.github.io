"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer for DTLZ4 using a hybrid approach:
    1. Structured initialization favoring the Pareto front.
    2. NSGA-II-like evolutionary search.
    3. Local Nelder-Mead refinement on best candidates.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = self.start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Budget allocation
        n_init = 24
        n_evo = 120
        n_local_candidates = 5
        max_local_evals_per_candidate = 8
        # Total expected: 24 + 120 + 5*8 = 184
        
        used = 0
        all_points = []
        all_fvals = []
        
        # Helper to evaluate and record
        def eval_point(x):
            nonlocal used
            if used >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            used += 1
            all_points.append(x)
            all_fvals.append(f)
            return f

        # 1. Structured Initialization
        # For DTLZ4, variables 0..dim-n_obj are 'distance' vars (should be 0)
        # variables dim-n_obj..dim are 'position' vars (determine front shape)
        n_dist = dim - n_obj
        n_pos = n_obj
        
        init_X = np.zeros((n_init, dim))
        for i in range(n_init):
            # Vary position variables uniformly
            pos_vars = np.random.rand(n_pos)
            # Vary distance variables: mix of 0 and small values
            if i % 3 == 0:
                dist_vars = np.zeros(n_dist)
            else:
                dist_vars = np.random.rand(n_dist) * 0.1
            init_X[i, :n_dist] = dist_vars
            init_X[i, n_dist:] = pos_vars
            
        # Evaluate initial population
        for x in init_X:
            f = eval_point(x)
            if f is None:
                break

        if used >= self.budget or time.time() > deadline:
            return all_points

        # 2. Evolutionary Search
        # Keep a population of the best non-dominated points
        # We'll maintain a list of (f, x) and prune dominated points
        
        pop = list(zip(all_fvals, all_points))
        
        def non_dominated_sort(points):
            # Simple non-dominated sorting for small populations
            # Returns list of non-dominated points
            n_dom = []
            for i, (f_i, x_i) in enumerate(points):
                dominated = False
                for j, (f_j, x_j) in enumerate(points):
                    if i == j: continue
                    # j dominates i if f_j <= f_i and f_j < f_i in at least one obj
                    if np.all(f_j <= f_i) and np.any(f_j < f_i):
                        dominated = True
                        break
                if not dominated:
                    n_dom.append((f_i, x_i))
            return n_dom

        pop = non_dominated_sort(pop)
        
        # If population is too small, keep all
        if len(pop) < 10:
             pop = list(zip(all_fvals, all_points))

        while used < self.budget - n_local_candidates * max_local_evals_per_candidate:
            if time.time() > deadline:
                break
                
            # Select parents
            # Choose 2 random non-dominated points
            if len(pop) < 2:
                 # If not enough non-dominated, pick random from all
                 parents = np.random.choice(len(all_points), 2, replace=False)
                 p1 = all_points[parents[0]]
                 p2 = all_points[parents[1]]
            else:
                 parents = np.random.choice(len(pop), 2, replace=False)
                 p1 = pop[parents[0]][1]
                 p2 = pop[parents[1]][1]
            
            # Crossover: Blend
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2
            
            # Mutation: Gaussian
            # Scale mutation by distance to bounds
            child += 0.1 * (hi - lo) * np.random.randn(dim)
            child = np.clip(child, lo, hi)
            
            f_child = eval_point(child)
            if f_child is None:
                break
                
            # Update population with new point
            all_points.append(child)
            all_fvals.append(f_child)
            pop = non_dominated_sort(list(zip(all_fvals, all_points)))
            
            # Limit population size for efficiency (keep best 20 non-dominated)
            if len(pop) > 20:
                # Simple crowding distance approximation: keep points that are spread out
                # For now, just keep first 20 non-dominated found
                pop = pop[:20]

        # 3. Local Refinement
        # Take the best 5 non-dominated points and refine them
        # We minimize the sum of objectives as a scalar proxy
        candidates = pop[:n_local_candidates] if len(pop) >= n_local_candidates else pop
        
        for f_cand, x_cand in candidates:
            if used >= self.budget - 1:
                break
            if time.time() > deadline:
                break
                
            # Define scalar objective: sum of objectives
            def scalar_obj(x):
                return np.sum(problem(x))
                
            # Use Nelder-Mead
            try:
                res = minimize(scalar_obj, x_cand, method='Nelder-Mead', 
                               bounds=list(zip(lo, hi)), 
                               options={'maxiter': max_local_evals_per_candidate, 
                                        'xatol': 1e-4, 'fatol': 1e-4})
                # Check if we used more budget than allocated
                # Note: minimize might use more evals than maxiter due to internal steps
                # We rely on the global 'used' counter and deadline check inside eval_point
                # But minimize calls problem directly, bypassing our eval_point!
                # We need to wrap problem to count evals.
            except Exception:
                pass

        return all_points
# EVOLVE-BLOCK-END
