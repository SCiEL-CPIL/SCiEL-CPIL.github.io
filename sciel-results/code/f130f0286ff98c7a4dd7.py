"""MODEL-AND-OPTIMIZE approach: Halton exploration + Stochastic Gradient Descent + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Halton space-filling + Stochastic Gradient Descent (SFGD) + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = 1e10
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def halton_sequence(n, dim, base_start=2):
            """Generate Halton sequence points in [0,1]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = base_start + d
                for i in range(n):
                    f = 1.0
                    r = 0.0
                    idx = i + 1
                    while idx > 0:
                        idx, rem = divmod(idx, base)
                        f /= base
                        r += rem * f
                    points[i, d] = r
            return points

        def nelder_mead(x0, max_evals, seed=42):
            """Nelder-Mead simplex method for local refinement."""
            nonlocal best_x, best_f, calls_made
            n = dim
            rng = np.random.default_rng(seed)

            # Build initial simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1, i] += 0.25 * max(0.1, abs(x0[i]))

            # Evaluate simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                fx = eval_func(simplex[i])
                if fx is None:
                    return
                fvals[i] = fx

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for iteration in range(max_evals):
                if time.time() > deadline:
                    break
                if calls_made >= budget - 2:
                    break

                # Sort by fitness
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                f_reflect = eval_func(x_reflect)
                if f_reflect is None:
                    break

                if f_reflect < fvals[0]:
                    # Expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    f_expand = eval_func(x_expand)
                    if f_expand is None:
                        break
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        fvals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        fvals[-1] = f_reflect
                elif f_reflect < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
                else:
                    # Contraction
                    if f_reflect < fvals[-1]:
                        # Outside contraction
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
                        # Inside contraction
                        x_contract = centroid - rho * (x_reflect - centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    f_contract = eval_func(x_contract)
                    if f_contract is None:
                        break
                    if f_contract < min(f_reflect, fvals[-1]):
                        simplex[-1] = x_contract
                        fvals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fx = eval_func(simplex[i])
                            if fx is None:
                                return
                            fvals[i] = fx
                        break

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-12 * (1 + np.abs(fvals[0])):
                    break

            return simplex[0], fvals[0]

        def sfgd_trajectory(x0, max_evals, seed=42):
            """Stochastic Gradient Descent with central finite differences."""
            nonlocal best_x, best_f, calls_made
            rng = np.random.default_rng(seed)
            n = dim
            
            # Initial parameters
            step_size = 0.5
            epsilon = 0.05
            x = np.asarray(x0, dtype=np.float64).copy()
            fx = eval_func(x)
            if fx is None:
                return x, fx
            
            no_improve_count = 0
            max_no_improve = 20

            for step in range(max_evals):
                if time.time() > deadline:
                    break
                if calls_made >= budget - 2:
                    break
                
                # Estimate gradient using central differences
                grad = np.zeros(n)
                for i in range(n):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += epsilon
                    x_minus[i] -= epsilon
                    
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    x_minus = np.clip(x_minus, -5.0, 5.0)
                    
                    f_plus = eval_func(x_plus)
                    if f_plus is None: break
                    f_minus = eval_func(x_minus)
                    if f_minus is None: break
                    
                    grad[i] = (f_plus - f_minus) / (2 * epsilon)
                
                # Normalize gradient to avoid huge steps
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-8:
                    grad = grad / grad_norm
                else:
                    # If gradient is zero, random direction
                    grad = rng.standard_normal(n)
                    grad = grad / np.linalg.norm(grad)
                
                # Take step
                x_trial = x - step_size * grad
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = eval_func(x_trial)
                if f_trial is None:
                    break
                
                if f_trial < fx:
                    # Success
                    x = x_trial
                    fx = f_trial
                    no_improve_count = 0
                    # Adaptively increase step size slightly
                    step_size *= 1.05
                    step_size = min(step_size, 2.0)
                else:
                    # Failure
                    no_improve_count += 1
                    # Decrease step size
                    step_size *= 0.8
                    step_size = max(step_size, 0.01)
                    
                    # If stuck for too long, restart from best known point with perturbation
                    if no_improve_count > max_no_improve:
                        x = best_x + 0.5 * rng.standard_normal(n)
                        x = np.clip(x, -5.0, 5.0)
                        fx = eval_func(x)
                        if fx is None: break
                        no_improve_count = 0
                        step_size = 0.5
                
                # Adapt epsilon based on progress
                if step % 10 == 0 and step > 0:
                    if fx < best_f:
                        epsilon *= 0.9
                    else:
                        epsilon *= 1.1
                    epsilon = np.clip(epsilon, 1e-4, 1.0)

            return x, fx

        # Phase 1: Halton space-filling for systematic early exploration
        n_halton = min(60, max(30, budget // 8))

        # Add center point first
        x0 = np.zeros(dim)
        fx0 = eval_func(x0)
        if fx0 is None:
            return best_x, best_f

        # Halton points mapped to [-5, 5]^dim
        halton_pts = halton_sequence(n_halton - 1, dim)
        for i in range(n_halton - 1):
            if time.time() > deadline:
                break
            if calls_made >= budget - 10:
                break
            x = halton_pts[i] * 10.0 - 5.0  # map [0,1] to [-5,5]
            x = np.clip(x, -5.0, 5.0)
            fx = eval_func(x)
            if fx is None:
                break

        if best_x is None:
            return best_x, best_f

        # Phase 2: Stochastic Gradient Descent
        remaining = budget - calls_made
        if remaining > 10:
            sfgd_budget = int(remaining * 0.7)
            sfgd_trajectory(best_x.copy(), max_evals=sfgd_budget, seed=1234)

        # Phase 3: Final Nelder-Mead polish from best point
        if best_x is not None and calls_made < budget - 5:
            remaining_polish = budget - calls_made - 5
            if remaining_polish > 10:
                nelder_mead(best_x.copy(), max_evals=remaining_polish, seed=5000)

        # Phase 4: Very fine local search if budget remains
        if best_x is not None and calls_made < budget - 2:
            remaining_fine = budget - calls_made - 2
            if remaining_fine > 5:
                # Tiny step SFGD for final precision
                sfgd_trajectory(best_x.copy(), max_evals=remaining_fine, seed=6000)

        return best_x, best_f
# EVOLVE-BLOCK-END
