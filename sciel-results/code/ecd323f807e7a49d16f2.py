"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based global exploration followed by Nelder-Mead local refinement.

    Strategy:
      1. Systematic grid sampling to identify the global basin of attraction.
      2. Coarse-to-fine grid refinement around the best grid point.
      3. Nelder-Mead local search from the best refined point.
      4. Explicit wall-clock deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin

        call_count = [0]

        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def nelder_mead(x0, max_iter=100):
            """
            Nelder-Mead simplex method starting from x0.
            Uses shared call_count and deadline.
            """
            dim = len(x0)
            
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.1
                simplex.append(xi)
            
            # Evaluate initial simplex
            fvals = []
            for i in range(dim + 1):
                if call_count[0] >= budget or time.time() > deadline:
                    return
                fi = make_call(simplex[i])
                if fi is None:
                    return
                fvals.append(fi)
                update_best(simplex[i], fi)
            
            if len(fvals) < dim + 1:
                return
            
            # Main loop
            for _ in range(max_iter):
                if call_count[0] >= budget or time.time() > deadline:
                    break
                
                # Sort by function value
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check convergence
                if abs(fvals[0] - fvals[-1]) < 1e-10:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = fvals[-1]
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                
                if call_count[0] >= budget or time.time() > deadline:
                    break
                f_ref = make_call(x_ref)
                if f_ref is None:
                    break
                update_best(x_ref, f_ref)
                
                if f_ref < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - worst)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    f_exp = make_call(x_exp)
                    if f_exp is None:
                        break
                    update_best(x_exp, f_exp)
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Simple reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (worst - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    f_con = make_call(x_con)
                    if f_con is None:
                        break
                    update_best(x_con, f_con)
                    
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            if call_count[0] >= budget or time.time() > deadline:
                                return
                            f_sh = make_call(simplex[i])
                            if f_sh is None:
                                return
                            update_best(simplex[i], f_sh)
                            fvals[i] = f_sh
                        break

        # Phase 1: Coarse grid exploration
        # Use a 3x3x3x3x3 grid over [-4, 4]^5
        # That's 3^5 = 243 points, well within budget
        grid_step = 8.0 / 2  # Step of 4.0, so points at -4, 0, 4
        grid_points = []
        
        # Generate 3^dim grid points
        for i in range(3 ** dim):
            point = np.zeros(dim)
            for d in range(dim):
                digit = (i // (3 ** d)) % 3
                point[d] = -4.0 + digit * 4.0
            grid_points.append(point)
        
        # Evaluate all grid points
        for gp in grid_points:
            if call_count[0] >= budget or time.time() > deadline:
                break
            f = make_call(gp)
            if f is None:
                break
            update_best(gp, f)
        
        # Phase 2: Fine grid around best point
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            # Create a fine grid around the best point
            fine_radius = 2.0
            fine_step = 1.0
            fine_points = []
            
            # Generate grid points within fine_radius of best_x
            for d in range(dim):
                for offset in [-fine_radius, -fine_step, 0.0, fine_step, fine_radius]:
                    if d == 0:
                        pass  # Will be handled by the loop below
            
            # Better approach: generate a 5x5x5x5x5 grid centered at best_x
            # But that's 3125 points, too many. Use 3x3x3x3x3 = 243 points
            fine_grid_step = 1.0
            fine_points = []
            for i in range(3 ** dim):
                point = best_x.copy()
                for d in range(dim):
                    digit = (i // (3 ** d)) % 3
                    point[d] = best_x[d] + (digit - 1) * fine_grid_step
                    point[d] = np.clip(point[d], -5.0, 5.0)
                fine_points.append(point)
            
            # Remove duplicates and points already evaluated
            for fp in fine_points:
                if call_count[0] >= budget or time.time() > deadline:
                    break
                f = make_call(fp)
                if f is None:
                    break
                update_best(fp, f)
        
        # Phase 3: Nelder-Mead refinement from best point
        if best_x is not None and call_count[0] < budget - 50 and time.time() < deadline - 10:
            nelder_mead(best_x, max_iter=150)
        
        # Phase 4: If we still have budget, do another round of Nelder-Mead
        # from a slightly perturbed best point
        if best_x is not None and call_count[0] < budget - 30 and time.time() < deadline - 5:
            perturbed = best_x + np.random.default_rng(123).uniform(-0.5, 0.5, dim)
            perturbed = np.clip(perturbed, -5.0, 5.0)
            nelder_mead(perturbed, max_iter=100)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
