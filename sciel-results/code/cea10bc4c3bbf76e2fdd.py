"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Hybrid optimizer using coarse grid search for phase estimation,
    followed by Nelder-Mead simplex refinement.
    Designed for BBOB f20 (Schwefel) on [-5,5]^5 with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        call_count = 0

        def try_func(x):
            nonlocal best_x, best_f, call_count
            if call_count >= self.budget:
                return np.inf
            try:
                fx = func(x)
                call_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return np.inf

        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse grid search to estimate local minima locations
        # Schwefel's function has known structure; use a grid to find promising regions
        # Use a stratified grid with ~100-150 evaluations
        
        # For 5D, a full grid is expensive. Use a smarter approach:
        # Sample along each dimension independently to find good coordinates,
        # then combine.
        
        # First, do a random sampling phase to get initial diverse points
        n_initial = min(80, budget // 5)
        candidates = []
        
        for i in range(n_initial):
            if call_count >= budget or time.time() > deadline - 10:
                break
            x = rng.uniform(-5, 5, dim)
            fx = try_func(x)
            candidates.append((fx, x))
        
        # Sort by fitness and keep top candidates
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Nelder-Mead simplex refinement from multiple starting points
        # Use the top few candidates as starting points for NM
        
        top_k = min(3, len(candidates))
        
        for k in range(top_k):
            if call_count >= budget or time.time() > deadline - 5:
                break
            
            start_point = candidates[k][1].copy()
            
            # Set up Nelder-Mead simplex
            # Initial simplex: start point plus small perturbations in each dimension
            simplex = [start_point.copy()]
            step_size = 1.0
            for d in range(dim):
                vertex = start_point.copy()
                vertex[d] += step_size
                if vertex[d] > 5.0:
                    vertex[d] = 5.0
                simplex.append(vertex)
            
            # Evaluate initial simplex
            simplex_f = []
            for v in simplex:
                if call_count >= budget or time.time() > deadline - 5:
                    break
                fv = try_func(v)
                simplex_f.append(fv)
            
            if len(simplex_f) < dim + 1:
                continue
            
            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            max_nm_iterations = 50
            for nm_iter in range(max_nm_iterations):
                if call_count >= budget or time.time() > deadline - 3:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = try_func(x_r)
                
                if f_r < simplex_f[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = try_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = try_func(x_c)
                    
                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = try_func(simplex[i])
                
                # Check for convergence
                f_range = max(simplex_f) - min(simplex_f)
                if f_range < 1e-10:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
