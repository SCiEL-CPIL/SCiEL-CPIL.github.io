"""Hooke-Jeeves pattern search for BBOB f12 (Bent cigar)."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s limit

        # Start at origin
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Hooke-Jeeves parameters
        step_sizes = np.ones(dim) * 2.0  # initial step size per coordinate
        min_step = 1e-10
        max_step = 10.0
        step_shrink = 0.5
        step_expand = 1.5

        # Pattern search loop
        while calls < budget and time.time() < deadline:
            progress = False
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break

                # Try positive direction
                trial = best_x.copy()
                trial[i] += step_sizes[i]
                trial = np.clip(trial, -5, 5)
                trial_f = func(trial)
                calls += 1

                if trial_f < best_f:
                    best_x = trial
                    best_f = trial_f
                    progress = True
                    step_sizes[i] *= step_expand
                    step_sizes[i] = min(step_sizes[i], max_step)
                    # Continue in the same direction (pattern move)
                    while calls < budget and time.time() < deadline:
                        trial = best_x.copy()
                        trial[i] += step_sizes[i]
                        trial = np.clip(trial, -5, 5)
                        trial_f = func(trial)
                        calls += 1
                        if trial_f < best_f:
                            best_x = trial
                            best_f = trial_f
                        else:
                            break
                else:
                    # Try negative direction
                    trial = best_x.copy()
                    trial[i] -= step_sizes[i]
                    trial = np.clip(trial, -5, 5)
                    trial_f = func(trial)
                    calls += 1

                    if trial_f < best_f:
                        best_x = trial
                        best_f = trial_f
                        progress = True
                        step_sizes[i] *= step_expand
                        step_sizes[i] = min(step_sizes[i], max_step)
                        # Continue in the same direction (pattern move)
                        while calls < budget and time.time() < deadline:
                            trial = best_x.copy()
                            trial[i] -= step_sizes[i]
                            trial = np.clip(trial, -5, 5)
                            trial_f = func(trial)
                            calls += 1
                            if trial_f < best_f:
                                best_x = trial
                                best_f = trial_f
                            else:
                                break
                    else:
                        # No improvement in either direction, shrink step size
                        step_sizes[i] *= step_shrink
                        step_sizes[i] = max(step_sizes[i], min_step)

            # If no progress was made in any coordinate, check if we've converged
            if not progress:
                # Check if all step sizes are at minimum
                if np.all(step_sizes <= min_step * 1.1):
                    break
                # Otherwise, try a larger random perturbation to escape local minima
                if calls < budget and time.time() < deadline:
                    rng = np.random.default_rng(calls)  # deterministic seed based on call count
                    trial = best_x + rng.uniform(-1, 1, dim)
                    trial = np.clip(trial, -5, 5)
                    trial_f = func(trial)
                    calls += 1
                    if trial_f < best_f:
                        best_x = trial
                        best_f = trial_f
                        # Reset step sizes to moderate values
                        step_sizes = np.ones(dim) * 0.5
                    else:
                        # Shrink all step sizes
                        step_sizes *= step_shrink
                        step_sizes = np.maximum(step_sizes, min_step)

        return best_x, best_f
