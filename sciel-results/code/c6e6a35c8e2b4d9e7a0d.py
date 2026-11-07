"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start Nelder-Mead optimizer.
    Uses multiple restarts to explore the search space, with each restart
    performing a local Nelder-Mead optimization within a sub-budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit

        # Count function evaluations
        eval_count = 0
        budget_exceeded = False

        def eval_func(x):
            nonlocal eval_count, budget_exceeded
            if budget_exceeded:
                return float('inf')
            if eval_count >= budget:
                budget_exceeded = True
                return float('inf')
            eval_count += 1
            return func(np.clip(x, -5.0, 5.0))

        rng = np.random.default_rng(42)
        
        # Global best tracker
        x_best_global = None
        f_best_global = float('inf')

        def update_global(x, f):
            nonlocal x_best_global, f_best_global
            if f < f_best_global:
                f_best_global = f
                x_best_global = x.copy()

        # Nelder-Mead implementation
        def nelder_mead(x0, f0, max_evals):
            # Initialize simplex
            simplex = [x0.copy()]
            simplex_f = [f0]
            
            # Initial step size
            delta = 0.5
            for i in range(dim):
                point = x0.copy()
                point[i] += delta
                simplex.append(point)
                simplex_f.append(eval_func(point))
                if budget_exceeded:
                    return simplex[0], simplex_f[0]

            # NM parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            # Track best in this NM run
            x_best_nm = x0.copy()
            f_best_nm = f0

            while eval_count < max_evals and time.time() < deadline:
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Update local best
                if simplex_f[0] < f_best_nm:
                    f_best_nm = simplex_f[0]
                    x_best_nm = simplex[0].copy()
                    update_global(x_best_nm, f_best_nm)

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                x_ref = centroid + alpha * (centroid - worst)
                f_ref = eval_func(x_ref)
                if budget_exceeded:
                    break

                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - worst)
                    f_exp = eval_func(x_exp)
                    if budget_exceeded:
                        break
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
                    # Contraction
                    x_con = centroid + rho * (worst - centroid)
                    f_con = eval_func(x_con)
                    if budget_exceeded:
                        break
                    if f_con < worst_f:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = eval_func(simplex[i])
                            if budget_exceeded:
                                break
                        if budget_exceeded:
                            break

                # Check for convergence (simple heuristic)
                if max(simplex_f) - min(simplex_f) < 1e-8:
                    break

            return x_best_nm, f_best_nm

        # Multi-start strategy
        # Divide budget into N restarts
        # We reserve some budget for the first restart to establish a baseline
        n_restarts = 10
        evals_per_restart = budget // n_restarts
        
        # Ensure at least 10 evals per restart
        if evals_per_restart < 10:
            n_restarts = max(1, budget // 10)
            evals_per_restart = budget // n_restarts

        # First restart: Start at origin or random
        x_start = np.zeros(dim)
        f_start = eval_func(x_start)
        update_global(x_start, f_start)
        
        if not budget_exceeded:
            x_best_nm, f_best_nm = nelder_mead(x_start, f_start, evals_per_restart)
            update_global(x_best_nm, f_best_nm)

        # Subsequent restarts
        for i in range(1, n_restarts):
            if budget_exceeded or time.time() > deadline:
                break
            
            # Choose starting point
            # Mix of random and perturbed best
            if rng.random() < 0.5 and x_best_global is not None:
                # Local exploration around best
                x_start = x_best_global + rng.normal(0, 1.0, dim)
            else:
                # Global exploration
                x_start = rng.uniform(-5.0, 5.0, dim)
            
            f_start = eval_func(x_start)
            if budget_exceeded:
                break
            update_global(x_start, f_start)
            
            x_best_nm, f_best_nm = nelder_mead(x_start, f_start, evals_per_restart)
            update_global(x_best_nm, f_best_nm)

        if x_best_global is None:
            # Fallback if no valid evaluation was made (should not happen)
            x_best_global = np.zeros(dim)
            f_best_global = eval_func(x_best_global)

        return x_best_global, f_best_global
# EVOLVE-BLOCK-END
