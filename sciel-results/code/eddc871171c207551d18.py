"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: (1+1)-ES with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def try_update(x, fx):
            nonlocal global_best_x, global_best_fx
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
                return True
            return False

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            return fx

        def time_ok():
            return time.time() < deadline - 5.0

        # Phase 1: Initial diverse sampling to get a rough landscape
        n_initial = min(15, budget // 10)
        for i in range(n_initial):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-4, 4, size=dim)
            evaluate(x)

        if global_best_x is None:
            x0 = np.zeros(dim)
            evaluate(x0)

        # Phase 2: (1+1)-ES with adaptive step size and stagnation-based restarts
        x = global_best_x.copy()
        fx = global_best_fx
        sigma = 1.0  # Initial step size
        stagnation_count = 0
        max_stagnation = 30  # Number of non-improving iterations before restart
        sigma_min = 1e-4
        sigma_max = 5.0
        success_ratio_target = 0.2  # Target success rate for step size adaptation
        window_size = 10  # Window for tracking success rate

        # Track success rate in a sliding window
        successes = []

        while calls_made < budget and time_ok():
            # Generate a single trial point
            direction = rng.standard_normal(dim)
            trial = x + sigma * direction
            trial = np.clip(trial, -5, 5)

            fx_trial = evaluate(trial)
            if fx_trial is None:
                break

            # Check for improvement
            if fx_trial < fx:
                # Success: accept the trial and adapt sigma
                x = trial.copy()
                fx = fx_trial
                successes.append(1)
                stagnation_count = 0
            else:
                # Failure: reject the trial
                successes.append(0)
                stagnation_count += 1

            # Maintain sliding window
            if len(successes) > window_size:
                successes.pop(0)

            # Adapt sigma based on success rate
            if len(successes) >= window_size // 2:
                success_rate = np.mean(successes)
                # Simple adaptation: increase sigma if success rate too high, decrease if too low
                if success_rate > success_ratio_target * 1.5:
                    sigma *= 1.2
                elif success_rate < success_ratio_target * 0.5:
                    sigma *= 0.8
                else:
                    sigma *= 1.0
                sigma = np.clip(sigma, sigma_min, sigma_max)

            # Restart if stagnation detected
            if stagnation_count >= max_stagnation:
                # Restart from a new random point or from global best
                if rng.random() < 0.5:
                    x = global_best_x.copy()
                    fx = global_best_fx
                else:
                    x = rng.uniform(-4, 4, size=dim)
                    fx = evaluate(x)
                    if fx is None:
                        break
                sigma = 1.0
                stagnation_count = 0
                successes = []

        # Phase 3: Final local refinement with coordinate descent if budget remains
        if global_best_x is not None and calls_made < budget and time_ok():
            x = global_best_x.copy()
            fx = global_best_fx
            step_sizes = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]

            for step in step_sizes:
                if calls_made >= budget or not time_ok():
                    break
                improved = True
                while improved:
                    improved = False
                    for j in range(dim):
                        if calls_made >= budget or not time_ok():
                            break
                        for direction in [-1, 1]:
                            x_trial = x.copy()
                            x_trial[j] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            fx_trial = evaluate(x_trial)
                            if fx_trial is not None and fx_trial < fx:
                                x = x_trial
                                fx = fx_trial
                                improved = True
                                break

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
