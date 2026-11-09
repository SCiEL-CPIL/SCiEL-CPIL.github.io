"""Rewritten optimizer: coordinate descent with adaptive per-axis steps + Sobol restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _sobol_sequence(n, dim, seed=0):
    """Generate n Sobol-like points in [0,1)^dim using a simple quasi-random approach."""
    rng = np.random.default_rng(seed)
    # Use Halton-like or just stratified random for simplicity and speed
    # Actually, let's use a simple Halton sequence for quasi-randomness
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = np.zeros((n, dim))
    for i in range(n):
        for d in range(dim):
            # Use prime bases for Halton
            base = 2 + d  # 2, 3, 5, 7, 11 for dim=5
            points[i, d] = halton(i + 1, base)
    return points


class Optimizer:
    """Coordinate descent with adaptive step sizes per dimension."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Sobol initialization for robust basin finding
        n_init = min(20, budget // 10)
        if n_init < 2:
            n_init = 2

        sobol_pts = _sobol_sequence(n_init, dim, seed=42)
        # Map to [-5, 5]^dim
        candidates = sobol_pts * 10.0 - 5.0

        best_x = candidates[0].copy()
        best_f = func(best_x)
        calls_used = 1

        for i in range(1, n_init):
            if calls_used >= budget:
                break
            x = candidates[i].copy()
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        # Adaptive coordinate descent
        # Step sizes per dimension, initialized to cover the box
        steps = np.full(dim, 2.0)
        min_step = 1e-6
        max_step = 10.0

        # Powell-like adaptive coordinate descent
        # We'll do full sweeps over dimensions, adapting step sizes based on improvement
        sweep = 0
        no_improve_sweeps = 0
        max_no_improve = 5

        while calls_used < budget:
            if time.time() > deadline:
                break

            improved_this_sweep = False
            for d in range(dim):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Try positive step
                x_plus = best_x.copy()
                x_plus[d] = np.clip(best_x[d] + steps[d], -5, 5)
                if np.allclose(x_plus, best_x):
                    # Step too small or at boundary
                    steps[d] *= 0.5
                    if steps[d] < min_step:
                        steps[d] = min_step
                    continue

                f_plus = func(x_plus)
                calls_used += 1

                if f_plus < best_f:
                    best_x = x_plus
                    best_f = f_plus
                    improved_this_sweep = True
                    # Increase step if we made good progress
                    steps[d] = np.clip(steps[d] * 1.3, min_step, max_step)
                else:
                    # Try negative step
                    x_minus = best_x.copy()
                    x_minus[d] = np.clip(best_x[d] - steps[d], -5, 5)
                    if np.allclose(x_minus, best_x):
                        steps[d] *= 0.5
                        if steps[d] < min_step:
                            steps[d] = min_step
                        continue

                    f_minus = func(x_minus)
                    calls_used += 1

                    if f_minus < best_f:
                        best_x = x_minus
                        best_f = f_minus
                        improved_this_sweep = True
                        steps[d] = np.clip(steps[d] * 1.3, min_step, max_step)
                    else:
                        # No improvement in either direction, shrink step
                        steps[d] *= 0.5
                        if steps[d] < min_step:
                            steps[d] = min_step

            sweep += 1
            if not improved_this_sweep:
                no_improve_sweeps += 1
                if no_improve_sweeps >= max_no_improve:
                    # Restart from a new Sobol point
                    if calls_used >= budget:
                        break
                    new_idx = (sweep % 100)
                    if new_idx < n_init:
                        new_x = candidates[new_idx].copy()
                    else:
                        # Generate a new random point
                        rng = np.random.default_rng(sweep)
                        new_x = rng.uniform(-5, 5, dim)
                    f_new = func(new_x)
                    calls_used += 1
                    if f_new < best_f:
                        best_x = new_x
                        best_f = f_new
                    no_improve_sweeps = 0
                    # Reset steps to a moderate value
                    steps = np.full(dim, 1.0)
            else:
                no_improve_sweeps = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
