"""Rewritten optimizer using Nelder-Mead Simplex with restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead Simplex optimizer for the Bent Cigar (f12) landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Nelder-Mead parameters
        # Standard parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrinkage

        # Initial simplex setup
        # Start at 0
        # Create a simplex of dim+1 points
        # We will use a scaled initial step
        init_step = 0.5
        
        def create_simplex(center, step):
            """Create a simplex centered at 'center' with given step size."""
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = center
            for i in range(dim):
                simplex[i + 1] = center.copy()
                simplex[i + 1][i] += step
            return simplex

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        # Global best tracking
        global_best_x = np.zeros(dim)
        global_best_f = float('inf')

        def eval_point(x):
            nonlocal calls, best_f, best_x, global_best_f, global_best_x
            # Clip to box to maintain simplex geometry consistency
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls += 1
            if f_val < global_best_f:
                global_best_f = f_val
                global_best_x = x_clipped.copy()
            return x_clipped, f_val

        # Initial simplex
        simplex = create_simplex(np.zeros(dim), init_step)
        f_vals = np.zeros(dim + 1)
        
        # Evaluate initial simplex
        for i in range(dim + 1):
            simplex[i], f_vals[i] = eval_point(simplex[i])
            if f_vals[i] < best_f:
                best_f = f_vals[i]
                best_x = simplex[i].copy()

        # Main loop
        max_stagnation = 2 * dim  # Restart if no improvement for this many iterations
        stagnation_count = 0
        current_step = init_step

        while calls < budget:
            # Sort simplex by fitness
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            # Update local best
            if f_vals[0] < best_f:
                best_f = f_vals[0]
                best_x = simplex[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Check for convergence or stagnation
            if stagnation_count >= max_stagnation:
                # Restart around the best point with smaller step
                current_step *= 0.5
                if current_step < 1e-4:
                    # If step is too small, try random perturbation around best
                    if calls < budget:
                         # Do a few random steps
                         while calls < budget:
                             # Random walk
                             direction = rng.standard_normal(dim)
                             direction /= np.linalg.norm(direction)
                             step_size = current_step * 10 # Use a slightly larger random step
                             new_x = best_x + step_size * direction
                             new_x, new_f = eval_point(new_x)
                             if new_f < best_f:
                                 best_f = new_f
                                 best_x = new_x
                                 current_step = step_size
                                 break
                    break
                simplex = create_simplex(best_x, current_step)
                f_vals = np.zeros(dim + 1)
                for i in range(dim + 1):
                    simplex[i], f_vals[i] = eval_point(simplex[i])
                stagnation_count = 0
                continue

            # Nelder-Mead steps
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            
            # Reflection
            reflected = centroid + alpha * (centroid - worst)
            reflected, f_reflected = eval_point(reflected)
            
            if f_reflected < f_vals[0]:
                # Expansion
                expanded = centroid + gamma * (centroid - worst)
                expanded, f_expanded = eval_point(expanded)
                if f_expanded < f_reflected:
                    simplex[-1] = expanded
                    f_vals[-1] = f_expanded
                else:
                    simplex[-1] = reflected
                    f_vals[-1] = f_reflected
            elif f_reflected < f_vals[-2]:
                # Accept reflection
                simplex[-1] = reflected
                f_vals[-1] = f_reflected
            else:
                # Contraction
                if f_reflected < f_vals[-1]:
                    # Outside contraction
                    contracted = centroid + rho * (reflected - centroid)
                else:
                    # Inside contraction
                    contracted = centroid - rho * (centroid - worst)
                
                contracted, f_contracted = eval_point(contracted)
                
                if f_contracted < f_vals[-1]:
                    simplex[-1] = contracted
                    f_vals[-1] = f_contracted
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i], f_vals[i] = eval_point(simplex[i])
                        if f_vals[i] < best_f:
                             best_f = f_vals[i]
                             best_x = simplex[i].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
