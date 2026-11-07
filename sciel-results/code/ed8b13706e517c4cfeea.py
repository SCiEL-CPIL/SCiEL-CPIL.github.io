"""Multi-start Nelder-Mead ensemble with Latin Hypercube initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead ensemble for global exploration and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Generate diverse starting points using Latin Hypercube sampling
        n_starts = 5
        lhd_points = []
        
        # Latin Hypercube: divide each dimension into n_starts equal intervals
        for d in range(dim):
            # Create n_starts intervals, pick one random point from each
            intervals = np.linspace(-5, 5, n_starts + 1)
            for i in range(n_starts):
                low = intervals[i]
                high = intervals[i + 1]
                point = rng.uniform(low, high)
                lhd_points.append(point)
        
        # Reshape into n_starts points of dim dimensions
        lhd_points = np.array(lhd_points).reshape(n_starts, dim)
        
        # Add origin as an additional starting point
        all_starts = np.vstack([lhd_points, np.zeros(dim)])
        n_starts = len(all_starts)
        
        # Initialize simplexes for each starting point
        alpha = 0.5  # Step size for initial simplex
        simplexes = []
        f_simplexes = []
        
        for start_idx in range(n_starts):
            if not time_ok() or not budget_ok():
                break
                
            x0 = all_starts[start_idx].copy()
            f0 = func(x0)
            calls += 1
            record(x0, f0)
            
            simplex = [x0.copy()]
            f_simplex = [f0]
            
            # Build simplex by perturbing each dimension
            for i in range(dim):
                if not time_ok() or not budget_ok():
                    break
                x_i = x0.copy()
                x_i[i] += alpha
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                simplex.append(x_i)
                f_simplex.append(f_i)
                record(x_i, f_i)
            
            if len(simplex) >= 2:
                simplexes.append(simplex)
                f_simplexes.append(f_simplex)
        
        if not simplexes:
            return best_x, best_f

        # Nelder-Mead parameters
        gamma = 1.0
        rho = 2.0
        sigma_nm = 0.5
        shrink = 0.5

        # Round-robin: process each simplex in turn
        active_simplexes = list(range(len(simplexes)))
        
        while time_ok() and budget_ok() and active_simplexes:
            for si in list(active_simplexes):
                if not time_ok() or not budget_ok():
                    break
                    
                simplex = simplexes[si]
                f_simplex = f_simplexes[si]
                
                # Sort simplex by function value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                simplexes[si] = simplex
                f_simplexes[si] = f_simplex
                
                x_best = simplex[0]
                f_best = f_simplex[0]
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_second_worst = simplex[-2]
                f_second_worst = f_simplex[-2]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + gamma * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                if not time_ok() or not budget_ok():
                    break
                f_r = func(x_r)
                calls += 1
                record(x_r, f_r)

                if f_r < f_best:
                    # Expansion
                    x_e = centroid + rho * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if not time_ok() or not budget_ok():
                        break
                    f_e = func(x_e)
                    calls += 1
                    record(x_e, f_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_second_worst:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + sigma_nm * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - sigma_nm * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)

                    if not time_ok() or not budget_ok():
                        break
                    f_c = func(x_c)
                    calls += 1
                    record(x_c, f_c)

                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if not time_ok() or not budget_ok():
                                break
                            simplex[i] = x_best + shrink * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            record(simplex[i], f_simplex[i])
                        if not time_ok():
                            break

                # Check for convergence of this simplex
                f_spread = np.max(f_simplex) - np.min(f_simplex)
                if f_spread < 1e-12 or np.max(np.abs(f_simplex)) < 1e-14:
                    active_simplexes.remove(si)
                    continue

                # Also check if simplex has collapsed in space
                max_dist = 0
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        d = np.max(np.abs(simplex[i] - simplex[j]))
                        max_dist = max(max_dist, d)
                if max_dist < 1e-10:
                    active_simplexes.remove(si)

        return best_x, best_f
# EVOLVE-BLOCK-END
