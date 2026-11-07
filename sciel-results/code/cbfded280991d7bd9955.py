"""Particle Swarm Optimization for f24 (Lunacek bi-Rastrigin) with adaptive parameters."""
import numpy as np
import time


class Optimizer:
    """
    PSO-based derivative-free optimizer for the Lunacek bi-Rastrigin function.
    Uses a swarm of particles with velocity clamping, adaptive inertia,
    and periodic re-seeding to escape local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = [0]

        def counted_func(x):
            calls_made[0] += 1
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            return float(func(x))

        rng = np.random.default_rng(42)

        # PSO parameters
        n_particles = 12
        w_initial = 0.9
        w_final = 0.4
        c1 = 1.5
        c2 = 1.5
        v_max = 2.0

        # Initialize swarm
        # Spread particles across the search space
        positions = rng.uniform(-4.5, 4.5, (n_particles, dim))
        velocities = rng.uniform(-1.0, 1.0, (n_particles, dim))

        # Evaluate initial positions
        personal_bests = np.empty((n_particles, dim))
        personal_best_f = np.empty(n_particles)

        for i in range(n_particles):
            if calls_made[0] >= budget or time.time() > deadline:
                break
            fx = counted_func(positions[i])
            personal_bests[i] = positions[i].copy()
            personal_best_f[i] = fx
            if fx < best_f:
                best_f = fx
                best_x = positions[i].copy()

        # Global best
        global_best_idx = np.argmin(personal_best_f)
        global_best = personal_bests[global_best_idx].copy()
        global_best_f = personal_best_f[global_best_idx]

        # Adaptive re-seeding: if no improvement for a while, re-seed worst particles
        no_improve_count = 0
        max_no_improve = 30

        # Main PSO loop
        iter_count = 0
        while calls_made[0] < budget and time.time() < deadline:
            iter_count += 1

            # Adaptive inertia weight
            progress = min(1.0, iter_count / 100.0)
            w = w_initial + (w_final - w_initial) * progress

            # Update velocities and positions
            for i in range(n_particles):
                if calls_made[0] >= budget or time.time() > deadline:
                    break

                r1 = rng.random(dim)
                r2 = rng.random(dim)

                # Velocity update
                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (personal_bests[i] - positions[i])
                    + c2 * r2 * (global_best - positions[i])
                )

                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Position update
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5, 5)

                # Evaluate
                fx = counted_func(positions[i])

                # Update personal best
                if fx < personal_best_f[i]:
                    personal_best_f[i] = fx
                    personal_bests[i] = positions[i].copy()

                    # Update global best
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best = positions[i].copy()
                        no_improve_count = 0

                        if fx < best_f:
                            best_f = fx
                            best_x = positions[i].copy()
                else:
                    no_improve_count += 1

            # Periodic re-seeding of worst particles if stuck
            if no_improve_count >= max_no_improve:
                no_improve_count = 0
                # Re-seed the worst 3 particles
                worst_indices = np.argsort(-personal_best_f)[:3]
                for idx in worst_indices:
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    # Random new position near the global best or random
                    if rng.random() < 0.5:
                        positions[idx] = global_best + rng.normal(0, 1.0, dim)
                    else:
                        positions[idx] = rng.uniform(-4.5, 4.5, dim)
                    positions[idx] = np.clip(positions[idx], -5, 5)
                    velocities[idx] = rng.uniform(-0.5, 0.5, dim)

                    fx = counted_func(positions[idx])
                    personal_best_f[idx] = fx
                    personal_bests[idx] = positions[idx].copy()

                    if fx < global_best_f:
                        global_best_f = fx
                        global_best = positions[idx].copy()
                        if fx < best_f:
                            best_f = fx
                            best_x = positions[idx].copy()

        # Final local polish with coordinate descent around the best point
        if calls_made[0] < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            step = 0.3
            for _ in range(30):
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls_made[0] >= budget or time.time() > deadline:
                            break
                        x_trial = x.copy()
                        x_trial[d] += sign * step
                        x_trial = np.clip(x_trial, -5, 5)
                        fx_trial = counted_func(x_trial)
                        if fx_trial < fx:
                            x = x_trial
                            fx = fx_trial
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break
                if not improved:
                    step *= 0.7
                    if step < 0.01:
                        break
                else:
                    step = max(0.05, step * 0.9)

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made[0] < budget:
                best_f = counted_func(best_x)

        return best_x, best_f
