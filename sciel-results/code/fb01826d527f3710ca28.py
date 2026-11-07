"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with adaptive step size for f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin

        call_count = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Simulated Annealing parameters
        T_start = 10.0
        T_end = 1e-6
        alpha = 0.9995  # cooling rate
        step_scale_start = 2.0
        step_scale_end = 0.01

        # Initial point: center of the box
        current_x = np.zeros(dim)
        current_fx = func(current_x)
        call_count += 1
        record(current_x, current_fx)

        T = T_start
        iteration = 0
        no_improve_count = 0

        while call_count < budget:
            if time.time() > deadline - 5:
                break

            # Adaptive step size based on temperature
            progress = (T_start - T) / (T_start - T_end)
            progress = min(1.0, max(0.0, progress))
            step_scale = step_scale_start * (step_scale_end / step_scale_start) ** progress

            # Generate candidate: mix of random jumps and local steps
            if rng.random() < 0.1:
                # Large random jump to escape plateaus
                candidate_x = rng.uniform(-4, 4, size=dim)
            else:
                # Local step with adaptive scale
                candidate_x = current_x + rng.normal(0, step_scale, size=dim)
                candidate_x = np.clip(candidate_x, -5, 5)

            candidate_fx = func(candidate_x)
            call_count += 1

            # Acceptance criterion
            if candidate_fx < current_fx:
                current_x = candidate_x
                current_fx = candidate_fx
                no_improve_count = 0
            else:
                delta = candidate_fx - current_fx
                if T > T_end:
                    acceptance_prob = np.exp(-delta / T)
                    if rng.random() < acceptance_prob:
                        current_x = candidate_x
                        current_fx = candidate_fx
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

            record(current_x, current_fx)

            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end

            # If stuck for too long, restart from best with a kick
            if no_improve_count > 100:
                current_x = best_x + rng.normal(0, 1.0, size=dim)
                current_x = np.clip(current_x, -5, 5)
                current_fx = func(current_x)
                call_count += 1
                record(current_x, current_fx)
                no_improve_count = 0
                # Reheat
                T = max(T, 1.0)

            iteration += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
