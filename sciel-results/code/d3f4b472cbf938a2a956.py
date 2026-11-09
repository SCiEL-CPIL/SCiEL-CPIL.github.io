"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Parallel Nelder-Mead simplex ensemble for BBOB f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Initialize multiple simplexes at different locations
        # We'll use 5 independent Nelder-Mead simplexes
        n_simplexes = 5
        simplex_size = 0.5  # Initial perturbation size for simplex vertices

        # Define simplex structures
        # Each simplex is a list of (dim+1) points
        simplexes = []
        simplex_fvals = []

        # Initialize simplexes at corners and center
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Corners of [-4, 4]^5 (sample a few)
        corners = []
        for _ in range(4):
            corner = rng.uniform(-4, 4, dim)
            corners.append(corner)
        init_points.extend(corners)

        for i, center in enumerate(init_points):
            if call_count >= budget or time.time() > deadline:
                break

            # Create simplex: center + perturbations along each axis
            simplex = [center.copy()]
            for d in range(dim):
                v = center.copy()
                v[d] += simplex_size
                simplex.append(np.clip(v, -5, 5))

            # Evaluate all vertices
            fvals = []
            for v in simplex:
                if call_count >= budget or time.time() > deadline:
                    break
                fv = call_func(v)
                if fv is None:
                    break
                fvals.append(fv)

            if len(fvals) == dim + 1:
                simplexes.append(simplex)
                simplex_fvals.append(fvals)

        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink

        max_iter_per_simplex = 50  # Limit iterations per simplex to allow switching
        stagnation_limit = 10

        # Track which simplex is active and its state
        active_simplex_idx = 0
        simplex_iters = [0] * len(simplexes)
        simplex_stagnation = [0] * len(simplexes)
        best_simplex_f = [min(fv) for fv in simplex_fvals]

        # Main loop: cycle through simplexes
        while call_count < budget and time.time() < deadline:
            # Select next simplex to work on (round-robin with some bias toward best)
            # Simple strategy: cycle through all simplexes
            for idx in range(len(simplexes)):
                if call_count >= budget or time.time() > deadline:
                    break

                simplex = simplexes[idx]
                fvals = np.array(simplex_fvals[idx])
                simplex_iters[idx] += 1

                if simplex_iters[idx] > max_iter_per_simplex:
                    # Reset this simplex: reinitialize around current best
                    if call_count >= budget:
                        break
                    center = best_x + rng.normal(0, 0.3, dim)
                    center = np.clip(center, -5, 5)
                    new_simplex = [center.copy()]
                    for d in range(dim):
                        v = center.copy()
                        v[d] += 0.2
                        new_simplex.append(np.clip(v, -5, 5))
                    new_fvals = []
                    for v in new_simplex:
                        if call_count >= budget:
                            break
                        fv = call_func(v)
                        if fv is None:
                            break
                        new_fvals.append(fv)
                    if len(new_fvals) == dim + 1:
                        simplexes[idx] = new_simplex
                        simplex_fvals[idx] = new_fvals
                        simplex_iters[idx] = 0
                        simplex_stagnation[idx] = 0
                        best_simplex_f[idx] = min(new_fvals)
                    continue

                # Sort vertices by function value
                sorted_idx = np.argsort(fvals)
                sorted_simplex = [simplex[i] for i in sorted_idx]
                sorted_fvals = [fvals[i] for i in sorted_idx]

                # Check convergence
                if sorted_fvals[0] < best_simplex_f[idx] - 1e-10:
                    best_simplex_f[idx] = sorted_fvals[0]
                    simplex_stagnation[idx] = 0
                else:
                    simplex_stagnation[idx] += 1

                if simplex_stagnation[idx] > stagnation_limit:
                    # Reinitialize this simplex
                    center = best_x + rng.normal(0, 0.5, dim)
                    center = np.clip(center, -5, 5)
                    new_simplex = [center.copy()]
                    for d in range(dim):
                        v = center.copy()
                        v[d] += 0.3
                        new_simplex.append(np.clip(v, -5, 5))
                    new_fvals = []
                    for v in new_simplex:
                        if call_count >= budget:
                            break
                        fv = call_func(v)
                        if fv is None:
                            break
                        new_fvals.append(fv)
                    if len(new_fvals) == dim + 1:
                        simplexes[idx] = new_simplex
                        simplex_fvals[idx] = new_fvals
                        simplex_iters[idx] = 0
                        simplex_stagnation[idx] = 0
                        best_simplex_f[idx] = min(new_fvals)
                    continue

                # Nelder-Mead step
                # Centroid of all points except worst
                centroid = np.mean(sorted_simplex[:-1], axis=0)

                # Reflection
                x_reflect = centroid + alpha * (centroid - sorted_simplex[-1])
                x_reflect = np.clip(x_reflect, -5, 5)
                if call_count >= budget:
                    break
                f_reflect = call_func(x_reflect)
                if f_reflect is None:
                    break

                if f_reflect < sorted_fvals[0]:
                    # Expansion
                    x_expand = centroid + gamma * (centroid - sorted_simplex[-1])
                    x_expand = np.clip(x_expand, -5, 5)
                    if call_count >= budget:
                        break
                    f_expand = call_func(x_expand)
                    if f_expand is None:
                        break

                    if f_expand < f_reflect:
                        new_point = x_expand
                        new_fval = f_expand
                    else:
                        new_point = x_reflect
                        new_fval = f_reflect
                elif f_reflect < sorted_fvals[-2]:
                    # Accept reflection
                    new_point = x_reflect
                    new_fval = f_reflect
                else:
                    # Contraction
                    if f_reflect < sorted_fvals[-1]:
                        # Outside contraction
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
                        # Inside contraction
                        x_contract = centroid - rho * (x_reflect - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    if call_count >= budget:
                        break
                    f_contract = call_func(x_contract)
                    if f_contract is None:
                        break

                    if f_contract < min(f_reflect, sorted_fvals[-1]):
                        new_point = x_contract
                        new_fval = f_contract
                    else:
                        # Shrink
                        # Move all points except best toward best
                        for i in range(1, dim + 1):
                            simplex[i] = sorted_simplex[0] + sigma * (simplex[i] - sorted_simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if call_count >= budget:
                                break
                            fvals[i] = call_func(simplex[i])
                            if fvals[i] is None:
                                break

                        # Update sorted structures
                        sorted_idx = np.argsort(fvals)
                        sorted_simplex = [simplex[i] for i in sorted_idx]
                        sorted_fvals = [fvals[i] for i in sorted_idx]
                        simplex[idx] = simplex
                        simplex_fvals[idx] = fvals
                        continue

                # Replace worst point
                simplex[-1] = new_point
                fvals[-1] = new_fval

                # Update sorted structures
                sorted_idx = np.argsort(fvals)
                sorted_simplex = [simplex[i] for i in sorted_idx]
                sorted_fvals = [fvals[i] for i in sorted_idx]
                simplex[idx] = simplex
                simplex_fvals[idx] = fvals

        # Final local polish using coordinate descent on the best point
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f

            for step in [0.1, 0.05, 0.02, 0.01, 0.005]:
                if call_count >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if call_count >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            fx_new = call_func(x_new)
                            if fx_new is None:
                                break
                            if fx_new < fx - 1e-12:
                                x = x_new
                                fx = fx_new
                                improved = True
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
