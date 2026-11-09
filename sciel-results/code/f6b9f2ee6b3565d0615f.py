import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_f, best_x
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Coarse-to-fine grid search to bracket the minimum
        # This systematically explores the space and finds a good starting point
        # for the local optimizer, which is crucial for AOCC (early progress)
        
        # Start with a coarse grid: 3 points per dimension = 3^5 = 243 evaluations
        # This covers [-4, 4] with step ~4
        grid_resolutions = [3, 4, 5]
        grid_ranges = [(-4.5, 4.5), (-4, 4), (-3.5, 3.5)]
        
        current_best_x = np.zeros(dim)
        current_best_f = np.inf
        
        for res, (lo, hi) in zip(grid_resolutions, grid_ranges):
            if calls_made >= budget - 20 or time.time() > deadline - 10:
                break
            
            # Create grid points
            coords = np.linspace(lo, hi, res)
            grid_points = list(np.meshgrid(*([coords] * dim), indexing='ij'))
            grid_points = np.array(grid_points).reshape(dim, -1).T
            
            # Evaluate all grid points
            for x in grid_points:
                if calls_made >= budget - 20 or time.time() > deadline - 10:
                    break
                fx = call_func(x)
                if fx < current_best_f:
                    current_best_f = fx
                    current_best_x = x.copy()
        
        start_x = current_best_x.copy()
        best_f = current_best_f
        best_x = current_best_x.copy()

        # Phase 2: Nelder-Mead with adaptive step size based on gradient magnitude
        # Estimate gradient via finite differences to adapt step size
        simplex = [start_x.copy()]
        simplex_f = [best_f]
        
        # Initial step size: use the grid resolution as a hint
        # The last grid had step ~ (hi-lo)/(res-1)
        initial_step = (grid_ranges[-1][1] - grid_ranges[-1][0]) / max(grid_resolutions[-1] - 1, 1)
        initial_step = max(0.5, min(1.5, initial_step))
        
        for i in range(dim):
            v = start_x.copy()
            v[i] += initial_step
            fv = call_func(v)
            simplex.append(v.copy())
            simplex_f.append(fv)
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        max_iter = 300
        tol = 1e-10
        no_improve_count = 0
        best_f_nm = best_f
        step_size = initial_step
        
        # Estimate gradient magnitude at current best point
        def estimate_gradient_magnitude(x, h=0.01):
            grad = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                # We don't want to spend too many calls on gradient estimation
                # Use central difference but only if we have budget
                if calls_made >= budget - 10:
                    return 1.0
                fp = call_func(xp)
                fm = call_func(xm)
                grad[i] = (fp - fm) / (2 * h)
            return np.linalg.norm(grad)
        
        for iteration in range(max_iter):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Sort simplex by fitness
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Check convergence
            if abs(simplex_f[-1] - simplex_f[0]) < tol:
                break
            
            # Track no-improvement
            if simplex_f[0] < best_f_nm - 1e-15:
                no_improve_count = 0
                best_f_nm = simplex_f[0]
            else:
                no_improve_count += 1
            
            # Adaptive step size based on gradient magnitude
            # Every 10 iterations, re-estimate gradient to adjust step size
            if iteration % 10 == 5 and calls_made < budget - 20 and time.time() < deadline - 5:
                grad_mag = estimate_gradient_magnitude(simplex[0], h=0.005)
                # If gradient is large, we can take bigger steps
                # If gradient is small, we need finer steps
                if grad_mag > 1.0:
                    step_size = min(step_size * 1.2, 2.0)
                elif grad_mag < 0.1:
                    step_size = max(step_size * 0.7, 0.1)
                else:
                    step_size = max(step_size * 0.9, 0.2)
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = call_func(x_ref)
            
            if f_ref < simplex_f[0]:
                # Expansion
                x_exp = centroid + gamma * (centroid - simplex[-1])
                f_exp = call_func(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    simplex_f[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                # Simple reflection
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Contraction
                if f_ref < simplex_f[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (centroid - simplex[-1])
                f_con = call_func(x_con)
                if f_con < min(f_ref, simplex_f[-1]):
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex_f[i] = call_func(simplex[i])
            
            # Adaptive restart: if stuck, try a new point near current best
            if no_improve_count >= 20 and calls_made < budget - 15:
                # Try points near current best with decreasing perturbation
                restart_found = False
                for attempt in range(5):
                    if calls_made >= budget - 15 or time.time() > deadline - 5:
                        break
                    perturbation = max(0.2, step_size * (0.5 ** attempt))
                    x_new = simplex[0] + rng.normal(0, perturbation, dim)
                    x_new = np.clip(x_new, -4.5, 4.5)
                    f_new = call_func(x_new)
                    if f_new < simplex_f[0] - 1e-8:
                        simplex = [x_new.copy()]
                        simplex_f = [f_new]
                        step_size = max(0.3, step_size * 0.8)
                        for i in range(dim):
                            v = x_new.copy()
                            v[i] += step_size
                            fv = call_func(v)
                            simplex.append(v.copy())
                            simplex_f.append(fv)
                        no_improve_count = 0
                        best_f_nm = f_new
                        restart_found = True
                        break
                if not restart_found:
                    no_improve_count = 10

        # Phase 3: Final polish with very tight tolerance
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline - 3:
            polish_tol = 1e-12
            polish_iter = 0
            while polish_iter < min(remaining // 2, 100):
                if calls_made >= budget or time.time() > deadline:
                    break
                polish_iter += 1
                
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                if abs(simplex_f[-1] - simplex_f[0]) < polish_tol:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = call_func(x_ref)
                
                if f_ref < simplex_f[0]:
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    f_exp = call_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    if f_ref < simplex_f[-1]:
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        x_con = centroid - rho * (centroid - simplex[-1])
                    f_con = call_func(x_con)
                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = call_func(simplex[i])

        return best_x, best_f
