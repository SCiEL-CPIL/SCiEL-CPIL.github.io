"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with adaptive step size for f7 step ellipsoidal."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        call_count = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling with Latin Hypercube for good early coverage
        n_lhs = 20
        # Generate LHS points
        lhs_points = []
        for i in range(n_lhs):
            point = np.zeros(dim)
            for d in range(dim):
                # Stratified sampling per dimension
                stratum = (i + rng.random()) / n_lhs
                point[d] = -5.0 + 10.0 * stratum
            lhs_points.append(point)

        # Evaluate LHS points
        for point in lhs_points:
            if call_count >= budget:
                break
            if time.time() > deadline - 5:
                break
            fx = func(point)
            call_count += 1
            record(point, fx)

        # Phase 2: Simulated Annealing
        if call_count < budget:
            # SA parameters
            T_start = 15.0  # high enough to jump over step discontinuities
            T_end = 1e-8
            # We'll run SA until budget or time runs out
            # Adaptive step size: start large, shrink as T decreases

            current_x = best_x.copy()
            current_fx = best_fx
            T = T_start

            # Number of iterations per temperature step
            iters_per_temp = 50
            n_temps = 200  # total temperature steps

            for t_step in range(n_temps):
                if call_count >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Compute current temperature (geometric cooling)
                if t_step < n_temps:
                    T = T_start * (T_end / T_start) ** (t_step / max(1, n_temps - 1))
                else:
                    T = T_end

                # Step size proportional to temperature
                step_scale = 0.1 + 3.0 * (T / T_start)

                for _ in range(iters_per_temp):
                    if call_count >= budget:
                        break
                    if time.time() > deadline - 5:
                        break

                    # Generate candidate
                    candidate = current_x + rng.normal(0, step_scale, size=dim)
                    candidate = np.clip(candidate, -5, 5)

                    fx = func(candidate)
                    call_count += 1

                    if fx < current_fx:
                        current_x = candidate
                        current_fx = fx
                        record(candidate, fx)
                    else:
                        # Accept with probability exp(-delta/T)
                        delta = fx - current_fx
                        if delta < 0:
                            current_x = candidate
                            current_fx = fx
                            record(candidate, fx)
                        else:
                            prob = np.exp(-delta / max(T, 1e-10))
                            if rng.random() < prob:
                                current_x = candidate
                                current_fx = fx

        # Phase 3: Local refinement with simple gradient-free method
        # Use a simple coordinate descent or Nelder-Mead-like refinement
        if call_count < budget and time.time() < deadline - 10:
            # Simple local search: try small perturbations in each dimension
            local_step = 0.5
            for _ in range(20):
                if call_count >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                improved = False
                for d in range(dim):
                    if call_count >= budget:
                        break
                    for direction in [-1, 1]:
                        candidate = best_x.copy()
                        candidate[d] += direction * local_step
                        candidate = np.clip(candidate, -5, 5)
                        if np.allclose(candidate, best_x):
                            continue

                        fx = func(candidate)
                        call_count += 1

                        if fx < best_fx:
                            best_x = candidate
                            best_fx = fx
                            improved = True

                if not improved:
                    local_step *= 0.5
                    if local_step < 0.01:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
