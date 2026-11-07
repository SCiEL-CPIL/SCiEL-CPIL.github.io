import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # 1. Initial coarse scan: structured sampling for better coverage
        # Use origin + stratified random points to find promising basins early
        init_points = [np.zeros(dim)]
        # Add stratified points: divide the box into regions
        n_strata = 2
        for _ in range(10):
            init_points.append(np.random.uniform(-5, 5, dim))

        # Evaluate initial points
        for x in init_points:
            if not time_ok() or calls >= budget:
                break
            f = func(x)
            calls += 1
            record(x, f)

        if best_x is None:
            return None, np.inf

        # 2. Multi-start Nelder-Mead ensemble for fast local convergence
        # Initialize multiple simplexes from the best initial points
        n_starts = min(4, max(1, (budget - calls) // 50))  # Limit to available budget
        
        # Select diverse starting points from initial evaluations
        # Use the best few distinct points
        candidate_starts = [best_x.copy()]
        # Add a few random perturbations of the best point for diversity
        for _ in range(n_starts - 1):
            perturbed = best_x + np.random.randn(dim) * 1.5
            perturbed = np.clip(perturbed, -5, 5)
            candidate_starts.append(perturbed)

        # Initialize Nelder-Mead simplexes
        alpha = 0.5  # Initial simplex spread
        simplexes = []
        f_simplexes = []

        for s in range(n_starts):
            if calls >= budget or not time_ok():
                break

            simplex = []
            f_vals = []

            # First vertex: the starting point
            x0 = candidate_starts[s].copy()
            x0 = np.clip(x0, -5, 5)
            
            # Check if we've already evaluated this point (approximate)
            f0 = func(x0)
            calls += 1
            record(x0, f0)
            simplex.append(x0)
            f_vals.append(f0)

            # Additional vertices: perturb each dimension
            for d in range(dim):
                if calls >= budget or not time_ok():
                    break
                x_i = x0.copy()
                x_i[d] += alpha
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                record(x_i, f_i)
                simplex.append(x_i)
                f_vals.append(f_i)

            if len(simplex) >= 2:
                simplexes.append(simplex)
                f_simplexes.append(f_vals)

        if len(simplexes) == 0:
            # Fallback to (1+1)-ES if Nelder-Mead couldn't initialize
            x = best_x.copy()
            f_current = best_f
            sigma = 0.5
            success_ema = 0.0
            target_success = 0.25
            stall_count = 0
            max_stall = 30

            while calls < budget and time_ok():
                z = np.random.randn(dim)
                norm_z = np.linalg.norm(z)
                if norm_z > 1e-10:
                    z = z / norm_z
                else:
                    z = np.zeros(dim)
                    z[0] = 1.0

                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)

                if not time_ok() or calls >= budget:
                    break
                f_new = func(x_new)
                calls += 1
                record(x_new, f_new)

                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

                success = (f_new < f_current)
                success_ema = 0.5 * success_ema + 0.5 * success

                if success_ema > target_success:
                    sigma *= 1.1
                elif success_ema < target_success:
                    sigma *= 0.9

                sigma = max(1e-6, min(10.0, sigma))

                if f_new < f_current:
                    x = x_new
                    f_current = f_new
                    stall_count = 0
                else:
                    stall_count += 1

                if stall_count >= max_stall:
                    sigma = max(sigma * 2.0, 1.0)
                    stall_count = 0

                if sigma < 1e-4:
                    break

            return best_x, best_f

        # Nelder-Mead parameters
        gamma = 1.0
        rho = 2.0
        sigma_nm = 0.5
        shrink = 0.5

        # Interleave evaluations across all active simplexes
        active_indices = list(range(len(simplexes)))
        max_iterations = 500  # Safety limit for NM iterations

        nm_iter = 0
        while calls < budget and time_ok() and active_indices and nm_iter < max_iterations:
            nm_iter += 1
            # Pick the next simplex to update (round-robin)
            idx = active_indices.pop(0)
            active_indices.append(idx)

            simplex = simplexes[idx]
            f_vals = f_simplexes[idx]

            # Sort by fitness
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]

            # Check convergence
            if len(f_vals) >= 2 and (f_vals[-1] - f_vals[0]) < 1e-10:
                continue

            x_best = simplex[0]
            f_best_s = f_vals[0]
            x_worst = simplex[-1]
            f_worst_s = f_vals[-1]

            # Compute centroid of all but worst
            x_centroid = np.mean([simplex[i] for i in range(len(simplex) - 1)], axis=0)

            # Reflection
            x_reflect = x_centroid + gamma * (x_centroid - x_worst)
            x_reflect = np.clip(x_reflect, -5, 5)

            if calls >= budget or not time_ok():
                break
            f_reflect = func(x_reflect)
            calls += 1
            record(x_reflect, f_reflect)

            if f_reflect < f_best_s:
                # Expansion
                x_expand = x_centroid + rho * (x_reflect - x_centroid)
                x_expand = np.clip(x_expand, -5, 5)
                if calls >= budget or not time_ok():
                    break
                f_expand = func(x_expand)
                calls += 1
                record(x_expand, f_expand)

                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    f_vals[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    f_vals[-1] = f_reflect
            elif f_reflect < f_worst_s:
                # Accept reflection
                simplex[-1] = x_reflect
                f_vals[-1] = f_reflect
            else:
                # Contraction
                x_contract = x_centroid + sigma_nm * (x_worst - x_centroid)
                x_contract = np.clip(x_contract, -5, 5)
                if calls >= budget or not time_ok():
                    break
                f_contract = func(x_contract)
                calls += 1
                record(x_contract, f_contract)

                if f_contract < f_worst_s:
                    simplex[-1] = x_contract
                    f_vals[-1] = f_contract
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = x_best + shrink * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls >= budget or not time_ok():
                            break
                        f_shrink = func(simplex[i])
                        calls += 1
                        record(simplex[i], f_shrink)
                        f_vals[i] = f_shrink

            # Update global best
            if f_best_s < best_f:
                best_f = f_best_s
                best_x = x_best.copy()

            # Remove converged simplexes
            if (f_vals[-1] - f_vals[0]) < 1e-10:
                active_indices.remove(idx)

        # 3. Final polish phase: coordinate descent from best
        if calls < budget and time_ok():
            x_polish = best_x.copy()
            f_polish = best_f
            step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]

            for step in step_sizes:
                if not time_ok() or calls >= budget:
                    break
                improved = True
                while improved and time_ok() and calls < budget:
                    improved = False
                    for d in range(dim):
                        # Try positive step
                        x_test = x_polish.copy()
                        x_test[d] += step
                        x_test = np.clip(x_test, -5, 5)
                        if calls >= budget or not time_ok():
                            break
                        f_test = func(x_test)
                        calls += 1
                        record(x_test, f_test)
                        if f_test < f_polish:
                            x_polish = x_test
                            f_polish = f_test
                            improved = True
                        # Try negative step
                        x_test = x_polish.copy()
                        x_test[d] -= step
                        x_test = np.clip(x_test, -5, 5)
                        if calls >= budget or not time_ok():
                            break
                        f_test = func(x_test)
                        calls += 1
                        record(x_test, f_test)
                        if f_test < f_polish:
                            x_polish = x_test
                            f_polish = f_test
                            improved = True
                if f_polish < best_f:
                    best_f = f_polish
                    best_x = x_polish.copy()

        return best_x, best_f
