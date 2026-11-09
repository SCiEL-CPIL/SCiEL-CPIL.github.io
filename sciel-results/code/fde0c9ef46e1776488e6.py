"""MODEL-AND-OPTIMIZE approach: Halton exploration + multi-fidelity (1+1)-ES + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Halton space-filling + parallel multi-fidelity (1+1)-ES trajectories + Nelder-Mead refinement."""

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
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

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
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
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

        def es_trajectory(x0, sigma0, max_evals, seed=42):
            """(1+1)-ES with adaptive step size for a single trajectory."""
            nonlocal best_x, best_f, calls_made
            rng = np.random.default_rng(seed)
            n = dim

            x = np.asarray(x0, dtype=np.float64).copy()
            sigma = sigma0
            fx = eval_func(x)
            if fx is None:
                return x, fx

            best_in_run = fx
            success_count = 0
            total_steps = 0

            for _ in range(max_evals):
                if time.time() > deadline:
                    break
                if calls_made >= budget - 2:
                    break

                # Generate mutant
                y = rng.standard_normal(n)
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5.0, 5.0)
                fx_new = eval_func(x_new)
                if fx_new is None:
                    break

                total_steps += 1

                # Accept if better
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    if fx < best_in_run:
                        best_in_run = fx
                else:
                    # Reject, shrink sigma slightly
                    sigma *= 0.95

                # Adapt sigma based on success rate (1/5 success rule)
                if total_steps >= 10:
                    rate = success_count / total_steps
                    if rate > 0.2:
                        sigma *= 1.05
                    elif rate < 0.1:
                        sigma *= 0.9
                    # Reset counters
                    success_count = 0
                    total_steps = 0

                # Clamp sigma
                sigma = np.clip(sigma, 1e-12, 5.0)

                # Early stop if converged
                if sigma < 1e-10:
                    break

            return x, best_in_run

        # Phase 1: Extensive Halton space-filling for systematic early exploration
        # Use more points to ensure good AOCC early
        n_halton = min(80, max(40, budget // 6))

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
            if calls_made >= budget - 30:
                break
            x = halton_pts[i] * 10.0 - 5.0  # map [0,1] to [-5,5]
            x = np.clip(x, -5.0, 5.0)
            fx = eval_func(x)
            if fx is None:
                break

        if best_x is None:
            return best_x, best_f

        # Phase 2: Multi-fidelity (1+1)-ES with different step sizes
        # Run multiple independent trajectories with different sigma values
        if calls_made < budget - 40:
            remaining_es = budget - calls_made - 40
            n_trajectories = 4  # number of parallel ES trajectories
            evals_per_traj = max(10, remaining_es // n_trajectories)

            # Different starting sigmas for multi-fidelity
            sigmas = [1.5, 0.8, 0.4, 0.15]
            seeds = [100, 200, 300, 400]

            for t in range(n_trajectories):
                if time.time() > deadline:
                    break
                if calls_made >= budget - 20:
                    break
                # Start from best point with different perturbations
                rng_t = np.random.default_rng(seeds[t])
                x_start = best_x + sigmas[t] * 0.3 * rng_t.standard_normal(dim)
                x_start = np.clip(x_start, -5.0, 5.0)
                es_trajectory(x_start, sigma0=sigmas[t], max_evals=evals_per_traj, seed=seeds[t])

        # Phase 3: Final Nelder-Mead polish from best point
        if best_x is not None and calls_made < budget - 5:
            remaining_polish = budget - calls_made - 5
            if remaining_polish > 10:
                nelder_mead(best_x.copy(), max_evals=remaining_polish, seed=5000)

        # Phase 4: Very fine local search if budget remains
        if best_x is not None and calls_made < budget - 2:
            remaining_fine = budget - calls_made - 2
            if remaining_fine > 5:
                es_trajectory(best_x.copy(), sigma0=0.01, max_evals=remaining_fine, seed=6000)

        return best_x, best_f
# EVOLVE-BLOCK-END
