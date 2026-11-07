"""Pattern search with adaptive step size for ill-conditioned Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive pattern search with orthogonal and diagonal directions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._start_time = None

    def __call__(self, func):
        self._start_time = time.time()
        deadline = self._start_time + 280.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Build direction set: axes + all diagonals (2^dim - 2 for dim=5 is 30,
        # but we use a smaller set for efficiency: axes + a few key diagonals)
        # For dim=5: 10 axis directions + 10 diagonal directions = 20 directions
        dirs = []
        # Axis directions
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            dirs.append(d)
            dirs.append(-d)
        # Diagonal directions (all combinations of +/-1)
        # For dim=5, 2^5 = 32, subtract 2 (all + and all -) = 30, but that's a lot
        # Use a subset: pick random diagonal directions
        rng = np.random.default_rng(42)
        n_diag = 10
        for _ in range(n_diag):
            d = rng.choice([-1.0, 1.0], dim)
            # Avoid duplicates with axes
            if np.sum(np.abs(d)) > 1:  # not an axis
                dirs.append(d)
        dirs = np.array(dirs)
        n_dirs = len(dirs)

        # Adaptive step size
        step = 2.0
        min_step = 1e-6
        max_step = 10.0

        # Best tracking
        best_x = None
        best_f = np.inf
        calls = 0

        # Multiple restarts with diverse starting points
        n_restarts = 5
        points_per_restart = budget // n_restarts

        for r in range(n_restarts):
            # Check time
            if time.time() > deadline:
                break

            # Starting point: diverse for each restart
            if r == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Pattern search loop
            for _ in range(points_per_restart - 1):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break

                improved = False
                # Try all directions
                for d in dirs:
                    if time.time() > deadline:
                        break
                    if calls >= budget:
                        break

                    y = np.clip(x + step * d, -5, 5)
                    fy = func(y)
                    calls += 1

                    if fy < fx:
                        x = y
                        fx = fy
                        improved = True
                        # Expand step
                        step = min(step * 1.5, max_step)
                        break  # restart direction loop with new x
                    else:
                        # Shrink step slightly
                        step = max(step * 0.85, min_step)

                if not improved:
                    # No direction improved; shrink step more aggressively
                    step = max(step * 0.5, min_step)
                    if step < min_step:
                        break  # converged, move to next restart

        # Final: use remaining budget for fine-tuning around best
        if calls < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            fine_step = step * 0.1 if step > min_step else 1e-4
            # Local coordinate descent
            for i in range(dim):
                for sign in [1, -1]:
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    y = x.copy()
                    y[i] = np.clip(y[i] + sign * fine_step, -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
