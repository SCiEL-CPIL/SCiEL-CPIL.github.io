"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5 using adaptive Nelder-Mead with periodic-aware restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Rastrigin-aware optimizer: lattice sampling + adaptive Nelder-Mead with periodic restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        call_count = [0]
        best_x = np.zeros(dim)
        best_f = np.inf

        def do_call(x):
            nonlocal best_f, best_x
            if call_count[0] >= budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            call_count[0] += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        # Phase 1: Lattice-aware global search
        # Rastrigin has period 2 in each dimension. Sample at key lattice positions.
        structured_points = [np.zeros(dim)]

        # Axis-aligned: 5 dims × 4 values = 20 points
        for i in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(dim)
                p[i] = val
                structured_points.append(p)

        # Pairwise: 10 pairs × 4 values = 40 points
        for i in range(dim):
            for j in range(i + 1, dim):
                for vi in [-1.0, 1.0]:
                    for vj in [-1.0, 1.0]:
                        p = np.zeros(dim)
                        p[i] = vi
                        p[j] = vj
                        structured_points.append(p)

        # Total so far: 1 + 20 + 40 = 61 points
        # Add random points to reach ~120
        global_budget = min(120, budget // 4)
        num_random = max(0, global_budget - len(structured_points))

        all_points = structured_points[:]
        if num_random > 0:
            random_points = rng.uniform(-5, 5, (num_random, dim))
            all_points.extend(random_points)

        # Evaluate all points
        for p in all_points:
            if not time_ok() or call_count[0] >= global_budget:
                break
            do_call(p)

        # Phase 2: Adaptive Nelder-Mead with periodic-aware restarts
        remaining = budget - call_count[0]
        if remaining < 20 or not time_ok():
            return best_x, best_f

        # NM parameters
        nm_calls = [0]
        max_stagnation = 15  # lower than 20 to trigger restarts sooner
        stagnation_count = [0]
        current_best = best_f
        current_best_x = best_x.copy()
        jump_size = 1.5  # initial jump size for restarts

        def nm_func(x):
            nonlocal current_best, current_best_x
            if nm_calls[0] >= remaining or call_count[0] >= budget or not time_ok():
                return current_best
            nm_calls[0] += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            call_count[0] += 1
            if val < current_best:
                current_best = val
                current_best_x = x_clipped.copy()
                stagnation_count[0] = 0
            else:
                stagnation_count[0] += 1
            return val

        # Initial simplex: larger initial simplex (1.0) better captures Rastrigin's periodic structure
        initial_simplex = np.vstack([
            current_best_x,
            current_best_x + np.array([1.0] + [0.0]*(dim-1)),
            current_best_x + np.array([0.0, 1.0] + [0.0]*(dim-2)),
            current_best_x + np.array([0.0, 0.0, 1.0] + [0.0]*(dim-3)),
            current_best_x + np.array([0.0, 0.0, 0.0, 1.0]),
            current_best_x + np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        ])

        # Evaluate initial simplex
        simplex_values = np.zeros(dim + 1)
        for i in range(dim + 1):
            if not time_ok() or call_count[0] >= budget:
                break
            simplex_values[i] = nm_func(initial_simplex[i])

        # Adaptive Nelder-Mead implementation
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5   # contraction
        sigma = 0.5  # shrink

        while call_count[0] < budget and time_ok() and nm_calls[0] < remaining:
            # Sort simplex by function value
            idx = np.argsort(simplex_values)
            initial_simplex = initial_simplex[idx]
            simplex_values = simplex_values[idx]

            # Check for convergence
            if np.max(simplex_values) - np.min(simplex_values) < 1e-10:
                break

            # Centroid of all but worst point
            centroid = np.mean(initial_simplex[:-1], axis=0)
            worst = initial_simplex[-1]

            # Reflection
            xr = centroid + alpha * (centroid - worst)
            fr = nm_func(xr)
            if fr < simplex_values[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst)
                fe = nm_func(xe)
                if fe < fr:
                    initial_simplex[-1] = xe
                    simplex_values[-1] = fe
                else:
                    initial_simplex[-1] = xr
                    simplex_values[-1] = fr
            elif fr < simplex_values[-2]:
                initial_simplex[-1] = xr
                simplex_values[-1] = fr
            else:
                # Contraction
                if fr < simplex_values[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    fc = nm_func(xc)
                    if fc <= fr:
                        initial_simplex[-1] = xc
                        simplex_values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            initial_simplex[i] = initial_simplex[0] + sigma * (initial_simplex[i] - initial_simplex[0])
                            simplex_values[i] = nm_func(initial_simplex[i])
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - xr)
                    fc = nm_func(xc)
                    if fc <= simplex_values[-1]:
                        initial_simplex[-1] = xc
                        simplex_values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            initial_simplex[i] = initial_simplex[0] + sigma * (initial_simplex[i] - initial_simplex[0])
                            simplex_values[i] = nm_func(initial_simplex[i])

            # Update best
            min_idx = np.argmin(simplex_values)
            if simplex_values[min_idx] < current_best:
                current_best = simplex_values[min_idx]
                current_best_x = initial_simplex[min_idx].copy()

            # Check for stagnation and restart
            if stagnation_count[0] >= max_stagnation:
                # Adaptive restart: jump from current best with increasing size
                if nm_calls[0] >= remaining - 10:
                    break
                # Jump to a new location based on periodic structure
                jump_direction = rng.normal(0, 1, dim)
                jump_direction = jump_direction / (np.linalg.norm(jump_direction) + 1e-10)
                new_center = current_best_x + jump_size * jump_direction
                new_center = np.clip(new_center, -5, 5)

                # Create new simplex around new center
                initial_simplex = np.vstack([
                    new_center,
                    new_center + np.array([jump_size] + [0.0]*(dim-1)),
                    new_center + np.array([0.0, jump_size] + [0.0]*(dim-2)),
                    new_center + np.array([0.0, 0.0, jump_size] + [0.0]*(dim-3)),
                    new_center + np.array([0.0, 0.0, 0.0, jump_size]),
                    new_center + np.array([0.0, 0.0, 0.0, 0.0, jump_size])
                ])

                # Evaluate new simplex
                simplex_values = np.zeros(dim + 1)
                for i in range(dim + 1):
                    if not time_ok() or call_count[0] >= budget:
                        break
                    simplex_values[i] = nm_func(initial_simplex[i])

                # Increase jump size for next restart
                jump_size *= 1.3
                stagnation_count[0] = 0

        # Update global best
        if current_best < best_f:
            best_f = current_best
            best_x = current_best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
