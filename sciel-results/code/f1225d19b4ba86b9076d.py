"""Optimizer for BBOB f5 (Linear slope) with structured probing and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured probing + Nelder-Mead + coordinate refinement for linear slope."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        d = self.dim
        budget = self.budget
        calls = 0

        def call(x):
            nonlocal calls
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = func(x)
            calls += 1
            return val

        # Phase 1: Structured probing to find the optimal vertex
        # For linear slope f(x) = c^T x + b, the minimum is at a vertex of [-5,5]^d
        # We probe strategically to identify the sign of each component of c

        # First, evaluate center
        center = np.zeros(d)
        f_center = call(center)

        # Probe each axis: evaluate center +/- small step in each direction
        # This tells us the sign of the gradient component
        probe_size = 1.0
        signs = np.zeros(d)
        for i in range(d):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            pos = center.copy()
            pos[i] = probe_size
            neg = center.copy()
            neg[i] = -probe_size
            f_pos = call(pos)
            if calls >= budget or time.time() > deadline:
                signs[i] = 1.0  # default
                break
            f_neg = call(neg)
            if f_pos < f_neg:
                signs[i] = 1.0  # gradient positive, go to +5
            else:
                signs[i] = -1.0  # gradient negative, go to -5

        # Jump to the suspected optimal vertex
        vertex = np.where(signs > 0, 5.0, -5.0)
        f_vertex = call(vertex)

        # Phase 2: Local refinement around the best point found so far
        # Use Nelder-Mead with a small simplex
        best_x = vertex.copy()
        best_f = f_vertex

        # If the vertex is on the boundary, the true optimum might be nearby
        # Do a local search to refine
        if calls < budget - 10 and time.time() < deadline:
            # Create a small simplex around the best point
            # For linear functions, the optimum is at a vertex, but let's verify
            # with a local search in case the function isn't perfectly linear
            # or there's numerical noise

            # Coordinate descent refinement
            step = 0.5
            for _ in range(20):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(d):
                    if calls >= budget or time.time() > deadline:
                        break
                    # Try moving in negative direction (toward lower f)
                    trial = best_x.copy()
                    trial[i] = np.clip(best_x[i] - step, -5.0, 5.0)
                    f_trial = call(trial)
                    if f_trial < best_f:
                        best_x = trial
                        best_f = f_trial
                        improved = True
                    else:
                        # Try positive direction
                        trial = best_x.copy()
                        trial[i] = np.clip(best_x[i] + step, -5.0, 5.0)
                        f_trial = call(trial)
                        if f_trial < best_f:
                            best_x = trial
                            best_f = f_trial
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        # Phase 3: Final polish with very small steps
        if calls < budget and time.time() < deadline:
            step = 1e-3
            for _ in range(5):
                if calls >= budget or time.time() > deadline:
                    break
                for i in range(d):
                    if calls >= budget or time.time() > deadline:
                        break
                    trial = best_x.copy()
                    trial[i] = np.clip(best_x[i] - step, -5.0, 5.0)
                    f_trial = call(trial)
                    if f_trial < best_f:
                        best_x = trial
                        best_f = f_trial
                    else:
                        trial = best_x.copy()
                        trial[i] = np.clip(best_x[i] + step, -5.0, 5.0)
                        f_trial = call(trial)
                        if f_trial < best_f:
                            best_x = trial
                            best_f = f_trial
                step *= 0.5

        # Use remaining budget for further refinement if possible
        while calls < budget and time.time() < deadline:
            # Random perturbation around best
            step = 0.1 * (1 - calls / budget) + 1e-4
            trial = best_x + step * np.random.randn(d)
            trial = np.clip(trial, -5.0, 5.0)
            f_trial = call(trial)
            if f_trial < best_f:
                best_x = trial
                best_f = f_trial
            else:
                # No improvement, shrink search
                step *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
