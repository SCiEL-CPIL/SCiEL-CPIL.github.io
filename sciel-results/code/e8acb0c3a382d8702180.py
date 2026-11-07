"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES runs with Nelder-Mead final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        call_count = [0]

        def make_call(x):
            if call_count[0] >= total_budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Nelder-Mead Simplex Method ---
        def nelder_mead(start_x, max_iter):
            nonlocal call_count, best_x, best_f
            
            # Initialize simplex
            simplex = []
            f_start = make_call(start_x)
            if f_start is None:
                return
            update_best(start_x, f_start)
            simplex.append((start_x.copy(), f_start))
            
            # Other points of the simplex
            for i in range(dim):
                x_i = start_x.copy()
                step = 0.5
                if x_i[i] + step > 5.0:
                    step = -0.5
                x_i[i] += step
                x_i = np.clip(x_i, -5.0, 5.0)
                
                f_i = make_call(x_i)
                if f_i is None:
                    return
                update_best(x_i, f_i)
                simplex.append((x_i.copy(), f_i))
                
                if time.time() > deadline - 5:
                    return

            # NM Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_sigma = 10.0
            min_sigma = 1e-8

            for _ in range(max_iter):
                if call_count[0] >= total_budget - 5:
                    break
                if time.time() > deadline - 5:
                    break

                # Sort simplex by function value
                simplex.sort(key=lambda x: x[1])
                
                x_worst = simplex[-1][0]
                f_worst = simplex[-1][1]
                x_best = simplex[0][0]
                f_best = simplex[0][1]
                
                # Check for convergence
                if f_worst - f_best < 1e-8:
                    break

                # Calculate centroid of all but worst point
                x_centroid = np.mean([x for x, _ in simplex[:-1]], axis=0)

                # Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = make_call(x_ref)
                if f_ref is None:
                    break
                update_best(x_ref, f_ref)

                if f_ref < f_best:
                    # Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = make_call(x_exp)
                    if f_exp is None:
                        break
                    update_best(x_exp, f_exp)
                    
                    if f_exp < f_ref:
                        simplex[-1] = (x_exp.copy(), f_exp)
                    else:
                        simplex[-1] = (x_ref.copy(), f_ref)
                elif f_ref < simplex[-2][1]:
                    simplex[-1] = (x_ref.copy(), f_ref)
                else:
                    # Contraction
                    if f_ref < f_worst:
                        # Outside contraction
                        x_con = x_centroid + rho * (x_ref - x_centroid)
                    else:
                        # Inside contraction
                        x_con = x_centroid + rho * (x_worst - x_centroid)
                    
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = make_call(x_con)
                    if f_con is None:
                        break
                    update_best(x_con, f_con)

                    if f_con < min(f_ref, f_worst):
                        simplex[-1] = (x_con.copy(), f_con)
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            x_shrink = x_best + sigma * (simplex[i][0] - x_best)
                            x_shrink = np.clip(x_shrink, -5.0, 5.0)
                            f_shrink = make_call(x_shrink)
                            if f_shrink is None:
                                return
                            update_best(x_shrink, f_shrink)
                            simplex[i] = (x_shrink.copy(), f_shrink)
                        
                        # Check size
                        size = np.linalg.norm(simplex[-1][0] - x_best)
                        if size < 1e-6:
                            break

        # --- (1+1)-ES Exploration ---
        def run_es(start_x, budget_left):
            nonlocal call_count, best_x, best_f
            
            x = start_x.copy()
            fx = make_call(x)
            if fx is None:
                return x, fx
            
            update_best(x, fx)
            
            sigma = 1.0
            for _ in range(budget_left):
                if call_count[0] >= total_budget:
                    break
                if time.time() > deadline - 5:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break
                
                update_best(y, fy)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    
                if sigma < 1e-8:
                    break
                    
            return x, fx

        # --- Main Loop: Portfolio Strategy ---
        
        # Phase 1: Broad exploration with multiple ES runs
        # Allocate 60% of budget to exploration
        exploration_budget = int(total_budget * 0.6)
        
        # Define diverse starting points
        start_points = []
        start_points.append(np.zeros(dim))
        
        # Random points
        for _ in range(10):
            start_points.append(rng.uniform(-5, 5, dim))
        
        # Points near corners
        for d in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[d] = sign * 4.0
                start_points.append(p)
        
        rng.shuffle(start_points)
        
        # Run ES from each start point, allocating remaining exploration budget
        n_starts = len(start_points)
        budget_per_es = exploration_budget // n_starts
        
        for i in range(n_starts):
            if call_count[0] >= total_budget - 50: # Keep 50 for NM
                break
            if time.time() > deadline - 30:
                break
                
            start_x = start_points[i]
            run_es(start_x, budget_per_es)
            
            # Check if we have a good candidate for NM
            # We'll run NM on the best found so far after exploration
            
        # Phase 2: Refine the best point found with Nelder-Mead
        if best_x is not None:
            nm_budget = total_budget - call_count[0]
            if nm_budget > 2 * dim: # Need at least dim+1 calls for simplex
                nelder_mead(best_x, nm_budget)
        
        # Return the best result
        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(dim)
            best_f = make_call(best_x)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
