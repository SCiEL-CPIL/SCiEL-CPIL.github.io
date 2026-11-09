"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-Lineage Parallel (1+1)-ES with Structured Initialization.
    
    Maintains multiple concurrent lineages at different sigma scales for
    simultaneous multi-resolution exploration. Each lineage independently
    adapts its step size. The best lineage is periodically reinforced.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_exceeded():
            return time.time() > deadline

        # ---------------------------------------------------------
        # Phase 1: Structured Initialization (random + lattice)
        # ---------------------------------------------------------
        n_init = 50
        init_points = []

        # Lattice points for Rastrigin-like functions
        lattice_vals = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        for _ in range(15):
            pt = np.array([rng.choice(lattice_vals) for _ in range(dim)])
            init_points.append(pt)

        # Random uniform
        for _ in range(n_init - 15):
            pt = rng.uniform(-5, 5, dim)
            init_points.append(pt)

        # High-priority points for early AOCC
        high_priority = [
            np.zeros(dim),
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
        ]

        all_init = high_priority + init_points
        rng.shuffle(all_init)

        for pt in all_init:
            if calls >= budget or time_exceeded():
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            update_best(pt, f)

        # ---------------------------------------------------------
        # Phase 2: Multi-Lineage Parallel (1+1)-ES
        # ---------------------------------------------------------
        # Multiple lineages at different scales
        n_lineages = 8
        sigma_scales = [3.0, 2.0, 1.5, 1.0, 0.7, 0.4, 0.2, 0.1]
        min_sigma = 0.005
        max_sigma = 5.0

        # Initialize lineages
        lineages = []
        for i in range(n_lineages):
            if i == 0:
                # First lineage starts at global best
                x = best_x.copy()
                fx = best_f
            else:
                # Other lineages start at random points
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls += 1
                update_best(x, fx)
            sigma = sigma_scales[i]
            stagnation = 0
            lineages.append({'x': x, 'fx': fx, 'sigma': sigma, 'stagnation': stagnation})

        # Track best lineage index for reinforcement
        best_lineage_idx = 0
        last_global_best = best_f
        no_improvement_rounds = 0
        max_no_improve_rounds = 10

        while calls < budget and not time_exceeded():
            # Round-robin: evaluate one offspring per lineage
            for i in range(n_lineages):
                if calls >= budget or time_exceeded():
                    break

                lin = lineages[i]
                x = lin['x']
                fx = lin['fx']
                sigma = lin['sigma']

                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    lin['x'] = y
                    lin['fx'] = fy
                    lin['stagnation'] = 0
                    sigma *= 1.15
                    update_best(y, fy)
                else:
                    lin['stagnation'] += 1
                    sigma *= 0.9

                lin['sigma'] = np.clip(sigma, min_sigma, max_sigma)

                # Update best lineage tracking
                if lin['fx'] < best_f:
                    best_lineage_idx = i

            # Check for global stagnation
            if best_f < last_global_best - 1e-10:
                last_global_best = best_f
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1

            # Reinforce best lineage: copy best point to stagnant lineages
            if no_improvement_rounds >= max_no_improve_rounds:
                no_improvement_rounds = 0
                # Find stagnant lineages (those far from best)
                for i in range(n_lineages):
                    if i == best_lineage_idx:
                        continue
                    lin = lineages[i]
                    dist = np.linalg.norm(lin['x'] - best_x)
                    if dist > 1.0:
                        # Reset this lineage near the best with moderate sigma
                        lin['x'] = np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5)
                        lin['fx'] = func(lin['x'])
                        calls += 1
                        update_best(lin['x'], lin['fx'])
                        lin['sigma'] = 0.5
                        lin['stagnation'] = 0
                        if calls >= budget or time_exceeded():
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
