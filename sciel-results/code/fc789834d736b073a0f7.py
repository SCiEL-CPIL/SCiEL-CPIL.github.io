import numpy as np
import time

class Optimizer:
    """Multi-start Hooke-Jeeves pattern search for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(start_x, max_iter=500):
            nonlocal eval_count
            if eval_count >= budget - 10:
                return

            x = np.clip(start_x, -5, 5).copy()
            f = eval_func(x)
            if eval_count >= budget:
                return

            # Initialize simplex
            simplex = [x.copy()]
            fvals = [f]

            # Create other vertices
            for i in range(d):
                new_x = x.copy()
                if new_x[i] == 0:
                    new_x[i] = 0.5
                else:
                    new_x[i] += 0.5
                new_x = np.clip(new_x, -5, 5)
                f_new = eval_func(new_x)
                if eval_count >= budget:
                    return
                simplex.append(new_x)
                fvals.append(f_new)

            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink

            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if abs(fvals[0] - fvals[-1]) < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = eval_func(x_ref)
                if eval_count >= budget:
                    break

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = eval_func(x_exp)
                    if eval_count >= budget:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - 0.5 * (centroid - simplex[-1])
                    x_con = np.clip(x_con, -5, 5)
                    f_con = eval_func(x_con)
                    if eval_count >= budget:
                        break
                    if f_con < min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = eval_func(simplex[i])
                            if eval_count >= budget:
                                return

        # Phase 1: Global Coarse Search / Multi-Start Initialization
        rng = np.random.default_rng(42)
        
        # Generate initial candidates: Origin and random points
        init_points = [np.zeros(d)]
        for _ in range(9):
            # Sample in [-4, 4] as per problem statement minimizer location
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
        
        # Phase 2: Local Refinement on the best starting points
        # Try to refine the top 3 points if budget allows
        top_k = min(3, len(point_scores))
        for i in range(top_k):
            if eval_count >= budget - 20:
                break
            nelder_mead(point_scores[i][1])

        return best_x, best_f
