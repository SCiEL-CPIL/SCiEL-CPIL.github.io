import numpy as np
import time

class Optimizer:
    """
    Multi-start Nelder-Mead optimizer for BBOB f12 (Bent Cigar).
    
    Strategy:
    1. Multi-start initialization: Evaluate several random points to find promising regions.
    2. Nelder-Mead simplex method: Local refinement that adapts to the local geometry.
       - The simplex (set of points) can orient itself along the narrow valley of the Bent Cigar.
       - Reflection, Expansion, and Contraction steps allow efficient navigation.
    3. Budget management: Strict tracking of evaluations and time to ensure validity.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: 330s limit, aim for 300s to be safe
        deadline = start_time + 300.0

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            
            # Clip to box [-5, 5]
            x = np.clip(np.asarray(x, dtype=float).flatten(), -5, 5)
            
            try:
                f = float(func(x))
            except Exception:
                return float('inf')
                
            if not np.isfinite(f):
                return float('inf')

            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            return f

        def nelder_mead(start_x, initial_f=None, max_iter=1000):
            """
            Nelder-Mead simplex method.
            """
            nonlocal eval_count
            
            if eval_count >= budget - 10:
                return

            # Initialize simplex
            # Start with the given point and d other points slightly perturbed
            # Perturbation size: 0.05 * range + 0.001
            # For Bent Cigar, we might want to align the initial simplex, 
            # but standard random perturbation usually works well enough with multi-start.
            
            x0 = np.clip(start_x, -5, 5).copy()
            if initial_f is not None:
                f0 = initial_f
            else:
                f0 = eval_func(x0)
                if eval_count >= budget:
                    return

            simplex = [x0]
            simplex_f = [f0]
            
            # Initialize other points
            # Use a small step for the initial simplex size
            # 0.05 is a reasonable default for [-5, 5]
            init_step = 0.05
            for i in range(d):
                xi = x0.copy()
                # Standard Nelder-Mead initialization: perturb one dimension
                # xi[i] += init_step if x0[i] + init_step <= 5 else x0[i] - init_step
                # Simpler: just add a small vector
                xi[i] += init_step
                simplex.append(np.clip(xi, -5, 5))
                fi = eval_func(simplex[i])
                simplex_f.append(fi)
                
                if eval_count >= budget:
                    return

            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrinkage
            
            # Max iterations to avoid infinite loops if budget is large but progress is slow
            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                # We have d+1 points
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                # If the best point hasn't changed in a while and steps are tiny, break?
                # For now, rely on budget and time.
                
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                
                # Centroid of all points except the worst
                x_c = np.mean(simplex[:-1], axis=0)
                
                # 1. Reflection
                x_r = x_c + alpha * (x_c - x_worst)
                f_r = eval_func(x_r)
                if eval_count >= budget:
                    break

                if f_r < simplex_f[0]:
                    # 2. Expansion
                    x_e = x_c + gamma * (x_c - x_worst)
                    f_e = eval_func(x_e)
                    if eval_count >= budget:
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Just accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # 3. Contraction
                    # Outside contraction if f_worst <= f_r, else inside
                    if f_r < f_worst:
                        x_c_out = x_c + rho * (x_r - x_c)
                        f_c_out = eval_func(x_c_out)
                        if eval_count >= budget:
                            break
                        
                        if f_c_out <= f_r:
                            simplex[-1] = x_c_out
                            simplex_f[-1] = f_c_out
                        else:
                            # 4. Shrinkage
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = eval_func(simplex[i])
                                if eval_count >= budget:
                                    break
                    else:
                        x_c_in = x_c - rho * (x_c - x_worst)
                        f_c_in = eval_func(x_c_in)
                        if eval_count >= budget:
                            break
                        
                        if f_c_in <= f_worst:
                            simplex[-1] = x_c_in
                            simplex_f[-1] = f_c_in
                        else:
                            # 4. Shrinkage
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = eval_func(simplex[i])
                                if eval_count >= budget:
                                    break
                            
                            if eval_count >= budget:
                                break

        # --- Phase 1: Multi-Start Initialization ---
        rng = np.random.default_rng(42)
        
        # Generate initial candidates
        # Include origin and random points in the likely range of the minimizer [-4, 4]
        init_points = [np.zeros(d)]
        # Add a few more diverse points
        for _ in range(19):
            init_points.append(rng.uniform(-4, 4, d))
            
        # Evaluate all initial points
        point_scores = []
        for p in init_points:
            if eval_count >= budget:
                break
            fp = eval_func(p)
            point_scores.append((fp, p))
            
        # Sort by function value
        point_scores.sort(key=lambda x: x[0])
        
        # --- Phase 2: Local Refinement on the best starting points ---
        # Refine the top K points. 
        # K=5 is a good balance. Nelder-Mead is more expensive per iteration than HJ,
        # but converges faster on this function.
        top_k = min(5, len(point_scores))
        
        for i in range(top_k):
            if eval_count >= budget - 20:
                break
            
            # Refine the i-th best point
            # Pass the known function value to save one evaluation
            nelder_mead(point_scores[i][1], point_scores[i][0])
            
            # Update best if NM found a better point
            # (eval_func already updates best_x, best_f)
            
            # If we found a very good solution, we can stop early to save time
            # But AOCC rewards early progress, so continuing to refine is good.
            
        return best_x, best_f
