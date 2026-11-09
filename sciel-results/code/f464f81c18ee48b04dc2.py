import numpy as np
import time

class Optimizer:
    """Valley-following coordinate descent for Rosenbrock-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 285.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = [
            np.ones(self.dim),
            np.zeros(self.dim),
            np.full(self.dim, 2.0),
            np.full(self.dim, -1.0),
            np.full(self.dim, 3.0),
        ]
        for _ in range(6):
            starts.append(rng.uniform(-3, 5, self.dim))
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, self.dim))

        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-5):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        for x0 in unique_starts:
            if self.calls_made >= self.budget - 30:
                break
            if time.time() > self.deadline:
                break

            x0 = np.clip(x0, -5, 5)
            x = x0.copy()
            f = wrapped_func(x)

            # Adaptive step sizes per coordinate
            step_sizes = np.full(self.dim, 0.5)
            prev_f = f

            # Coordinate descent with adaptive steps
            max_rounds = 80
            for _ in range(max_rounds):
                if self.calls_made >= self.budget:
                    break
                if time.time() > self.deadline:
                    break

                improved = False
                for i in range(self.dim):
                    if self.calls_made >= self.budget:
                        break
                    if time.time() > self.deadline:
                        break

                    # Try positive and negative steps
                    for sign in [1, -1]:
                        x_trial = x.copy()
                        x_trial[i] += sign * step_sizes[i]
                        x_trial = np.clip(x_trial, -5, 5)
                        
                        if np.allclose(x_trial, x, atol=1e-10):
                            continue

                        f_trial = wrapped_func(x_trial)

                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            # Increase step size slightly (momentum along valley)
                            step_sizes[i] *= 1.3
                            improved = True
                        else:
                            # Decrease step size
                            step_sizes[i] *= 0.6
                            # Cap step size
                            step_sizes[i] = min(step_sizes[i], 2.0)
                            step_sizes[i] = max(step_sizes[i], 0.01)

                    # If both directions failed, shrink step further
                    if not improved:
                        step_sizes[i] *= 0.5
                        step_sizes[i] = max(step_sizes[i], 0.005)

                # Check convergence
                if not improved and all(s < 0.01 for s in step_sizes):
                    break

                # If no improvement in this full round, break
                if not improved:
                    break

            # If we found a better point, continue; otherwise try next start
            if f >= self.best_f - 1e-10:
                # No improvement from this start, move on
                pass

        # Final evaluation if nothing was found
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = func(np.clip(x0, -5, 5))

        return self.best_x, self.best_f
