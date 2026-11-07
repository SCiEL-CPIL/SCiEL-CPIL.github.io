"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex optimization with adaptive restarts and local polishing."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track best solution found overall
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma_nm = 0.5 # shrink

        # Strategy:
        # 1. Run Nelder-Mead with a relatively large initial step to explore the space.
        # 2. If it converges or stagnates, restart from the best point with a smaller step.
        # 3. Use remaining budget for a fine-grained local search (polishing).

        # We will allocate budget dynamically. 
        # Start with a generous NM budget, keep some for polishing.
        # If NM finishes early, we can restart with smaller steps.
        
        max_restarts = 15
        initial_step = 2.0
        current_step = initial_step
        
        # Reserve 10% for final polishing
        polish_budget = int(budget * 0.1)
        nm_budget = budget - polish_budget

        for restart in range(max_restarts):
            if calls_made >= nm_budget:
                break

            # Initialize simplex
            if best_x is None:
                center = rng.uniform(-4.0, 4.0, dim)
            else:
                # Perturb the best point slightly to avoid exact repetition
                center = best_x.copy()
                center += rng.normal(0, current_step * 0.05, dim)
                center = np.clip(center, -5.0, 5.0)

            # Create simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = center
            for i in range(dim):
                simplex[i + 1] = center.copy()
                simplex[i + 1][i] += current_step

            # Evaluate initial simplex
            fx_values = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= nm_budget:
                    break
                fx_values[i] = evaluate(simplex[i])

            if calls_made >= nm_budget:
                break

            # Nelder-Mead loop
            stagnation_count = 0
            last_best_f = best_f
            converged = False

            while calls_made < nm_budget:
                # Sort simplex
                order = np.argsort(fx_values)
                simplex = simplex[order]
                fx_values = fx_values[order]

                # Convergence check: function values are very close
                f_range = fx_values[-1] - fx_values[0]
                if f_range < 1e-8:
                    converged = True
                    break

                # Centroid of best dim points
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)

                if calls_made >= nm_budget:
                    break

                if f_r < fx_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)
                    
                    if calls_made >= nm_budget:
                        break

                    if f_e < f_r:
                        simplex[-1] = x_e
                        fx_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fx_values[-1] = f_r
                elif f_r < fx_values[-2]:
                    # Reflection
                    simplex[-1] = x_r
                    fx_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < fx_values[-1]:
                        # Outer contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = evaluate(x_c)

                    if calls_made >= nm_budget:
                        break

                    if f_c < min(f_r, fx_values[-1]):
                        simplex[-1] = x_c
                        fx_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if calls_made < nm_budget:
                                fx_values[i] = evaluate(simplex[i])
                        if calls_made >= nm_budget:
                            break

                # Check for improvement in global best
                if best_f < last_best_f - 1e-6:
                    stagnation_count = 0
                    last_best_f = best_f
                else:
                    stagnation_count += 1

                if stagnation_count > 20:
                    break

            # If we broke out due to stagnation or convergence, reduce step for next restart
            # If we converged, we can reduce step significantly to polish locally
            # If we stagnated, we might need to escape, but since we restart from best_x,
            # reducing step is usually safer for local minima refinement unless we are sure
            # we are in a wrong basin. Given the box and single minimum likely, reducing is good.
            
            if converged:
                current_step *= 0.5
            else:
                # Stagnation: try a slightly smaller step to refine, or if step is already small, stop
                if current_step > 0.1:
                    current_step *= 0.7
                else:
                    break # Step is very small, further restarts unlikely to help

        # Final Polishing: Local Hill Climbing
        if best_x is not None and calls_made < budget:
            step = 0.5
            # Start with a reasonable step and reduce
            while calls_made < budget and step > 1e-5:
                improved = False
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    # Positive step
                    x_try = best_x.copy()
                    x_try[i] += step
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = evaluate(x_try)
                    if f_try < best_f:
                        best_x = x_try
                        best_f = f_try
                        improved = True

                if calls_made >= budget:
                    break

                # Negative step
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    x_try = best_x.copy()
                    x_try[i] -= step
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = evaluate(x_try)
                    if f_try < best_f:
                        best_x = x_try
                        best_f = f_try
                        improved = True

                if not improved:
                    step *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
