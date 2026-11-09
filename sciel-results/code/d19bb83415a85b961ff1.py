"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate Descent with adaptive step sizes and periodic perturbations."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution found
        x_best = rng.uniform(-5, 5, dim)
        fx_best = func(x_best)
        calls_used = 1

        # Per-axis step sizes
        steps = np.ones(dim) * 1.0

        # Global step decay
        global_step = 1.0

        no_improve_count = 0
        perturb_every = 30

        while calls_used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Periodic random perturbation to escape local minima
            if no_improve_count >= perturb_every:
                x_best = np.clip(x_best + rng.standard_normal(dim) * 2.0, -5, 5)
                fx_best = func(x_best)
                calls_used += 1
                no_improve_count = 0
                global_step = 1.0
                steps = np.ones(dim) * 1.0
                if calls_used >= budget:
                    break

            # Cyclic coordinate descent sweep
            improved_this_sweep = False
            for i in range(dim):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Try positive step
                x_trial = x_best.copy()
                x_trial[i] = np.clip(x_best[i] + steps[i] * global_step, -5, 5)
                fx_trial = func(x_trial)
                calls_used += 1

                if fx_trial < fx_best:
                    x_best = x_trial
                    fx_best = fx_trial
                    steps[i] *= 1.1  # grow step if successful
                    improved_this_sweep = True
                else:
                    steps[i] *= 0.85  # shrink step if not successful

                    # Try negative step
                    if calls_used < budget:
                        x_trial = x_best.copy()
                        x_trial[i] = np.clip(x_best[i] - steps[i] * global_step, -5, 5)
                        fx_trial = func(x_trial)
                        calls_used += 1

                        if fx_trial < fx_best:
                            x_best = x_trial
                            fx_best = fx_trial
                            steps[i] *= 1.1
                            improved_this_sweep = True
                        else:
                            steps[i] *= 0.85

            # Update no-improve counter
            if improved_this_sweep:
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Decay global step
            global_step *= 0.97

            # Ensure minimum step size
            global_step = max(global_step, 1e-6)

        return x_best, fx_best
# EVOLVE-BLOCK-END
