"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derandomized Discrete Particle Swarm Optimization (ddPSO) with ring topology."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        max_time = 325.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)

        # Phase 1: Global Sampling with LHS for initial swarm
        n_particles = 30
        samples = np.zeros((n_particles, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_particles + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            samples[:, d] = points

        f_samples = np.array([func(x) for x in samples])
        calls = n_particles

        if time.time() - start_time > max_time:
            idx = np.argmin(f_samples)
            return samples[idx], f_samples[idx]

        # Initialize particle states
        positions = samples.copy()
        velocities = np.zeros((n_particles, dim))
        personal_best = samples.copy()
        personal_best_f = f_samples.copy()

        global_best_idx = np.argmin(personal_best_f)
        global_best = personal_best[global_best_idx].copy()
        global_best_f = personal_best_f[global_best_idx]

        # Phase 2: ddPSO with ring topology
        # Ring topology: particle i communicates with particle (i-1) % n and (i+1) % n
        # Derandomized: fixed step sizes based on budget progression
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        w_max = 0.9
        w_min = 0.4
        v_max = 1.0  # Maximum velocity bound

        remaining_budget = budget - calls
        total_steps = remaining_budget
        step = 0

        while calls < budget and time.time() - start_time < max_time:
            # Dynamic inertia weight: decreases linearly from w_max to w_min
            progress = step / max(1, total_steps)
            w = w_max - (w_max - w_min) * progress

            for i in range(n_particles):
                if calls >= budget or time.time() - start_time > max_time:
                    break

                # Ring topology neighbors
                neighbor1 = (i - 1) % n_particles
                neighbor2 = (i + 1) % n_particles

                # Social component uses the best of neighbors (ring topology)
                neighbor_best_idx = neighbor1 if personal_best_f[neighbor1] < personal_best_f[neighbor2] else neighbor2
                neighbor_best = personal_best[neighbor_best_idx]

                # Velocity update (derandomized: fixed coefficients)
                r1 = 0.5  # Fixed random coefficient (derandomized)
                r2 = 0.5  # Fixed random coefficient (derandomized)

                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best[i] - positions[i]) +
                                 c2 * r2 * (neighbor_best - positions[i]))

                # Clip velocity to v_max
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Position update
                new_pos = positions[i] + velocities[i]
                new_pos = np.clip(new_pos, -5, 5)

                # Evaluate new position
                new_f = func(new_pos)
                calls += 1
                step += 1

                # Update personal best
                if new_f < personal_best_f[i]:
                    personal_best[i] = new_pos.copy()
                    personal_best_f[i] = new_f
                    positions[i] = new_pos.copy()

                    # Update global best
                    if new_f < global_best_f:
                        global_best = new_pos.copy()
                        global_best_f = new_f
                else:
                    # Keep position if not improved (velocity will be updated next iteration)
                    positions[i] = new_pos.copy()

        return global_best, global_best_f
# EVOLVE-BLOCK-END
