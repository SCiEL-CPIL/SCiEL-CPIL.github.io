# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Adaptive (1+1)-ES with Halton-based coarse-to-fine initialization
    and adaptive restarts for BBOB-style optimization.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Helper: Halton sequence
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q /= base
                r += q * (index % base)
                index //= base
            return r

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        def generate_halton(n, dim):
            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for j in range(dim):
                    base = primes[j % len(primes)]
                    points[i - 1, j] = halton_sequence(i, base)
            return points

        # Global state
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Phase 1: Coarse-to-fine Halton sampling
        # Start with coarse grid, refine around best if improvement found
        n_coarse = 30
        halton_coarse = generate_halton(n_coarse, dim)
        halton_coarse = halton_coarse * 10.0 - 5.0

        # Also add strategic points
        strategic = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
        ]

        all_init_points = list(halton_coarse) + strategic
        # Shuffle to interleave Halton and strategic
        rng.shuffle(all_init_points)

        for pt in all_init_points:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 2: Fine Halton refinement around current best
        # Generate fine Halton points in a smaller box around best_x
        n_fine = 20
        fine_radius = 3.0
        center = best_x.copy()
        halton_fine = generate_halton(n_fine, dim)
        # Map Halton [0,1]^d to [center - radius, center + radius]
        halton_fine = center + (halton_fine * 2.0 - 1.0) * fine_radius
        halton_fine = np.clip(halton_fine, -5, 5)

        for pt in halton_fine:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 3: (1+1)-ES with adaptive sigma and restarts
        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        # Track success rate for 1/5th rule
        success_window = []
        window_size = 15

        stagnation = 0
        no_improve_count = 0

        sigma_min = 0.005
        sigma_max = 4.0

        # Budget allocation: use last 10% for polish
        polish_start = int(budget * 0.90)

        # Pre-generate a pool of Halton-inspired restart points
        n_restart_pool = 50
        restart_pool = generate_halton(n_restart_pool, dim)
        restart_pool = restart_pool * 10.0 - 5.0
        restart_idx = 0

        while calls < budget:
            if time.time() > deadline:
                break

            in_polish = calls >= polish_start
            if in_polish:
                sigma = min(sigma, 0.1)

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            success = fy < fx
            success_window.append(success)
            if len(success_window) > window_size:
                success_window.pop(0)

            # 1/5th success rule
            if len(success_window) >= window_size:
                success_rate = sum(success_window) / len(success_window)
                if success_rate > 0.3:
                    sigma = min(sigma * 1.2, sigma_max)
                elif success_rate < 0.1:
                    sigma = max(sigma * 0.7, sigma_min)

            if success:
                x, fx = y, fy
                stagnation = 0
            else:
                stagnation += 1

            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Adaptive restart conditions
            progress_ratio = calls / budget
            stagnation_limit = int(40 * (1.0 - 0.5 * progress_ratio))
            no_improve_limit = int(60 * (1.0 - 0.5 * progress_ratio))

            if in_polish:
                stagnation_limit = 15
                no_improve_limit = 25

            # Restart if stagnated
            if stagnation >= stagnation_limit:
                explore_prob = min(0.3 + 0.01 * no_improve_count, 0.8)

                if rng.random() < explore_prob:
                    # Global exploration using Halton-inspired point
                    if restart_idx < len(restart_pool):
                        pt = restart_pool[restart_idx].copy()
                        restart_idx += 1
                    else:
                        idx = int(rng.integers(1, n_coarse + 1))
                        pt = np.zeros(dim)
                        for j in range(dim):
                            base = primes[j % len(primes)]
                            pt[j] = halton_sequence(idx + calls, base)
                        pt = pt * 10.0 - 5.0
                    x = np.clip(pt, -5, 5)
                    sigma = 2.0
                else:
                    # Local exploration around best
                    local_sigma = max(sigma * 2.0, 0.5)
                    x = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                    sigma = local_sigma

                fx = func(x)
                calls += 1
                stagnation = 0
                no_improve_count = 0
                success_window = []

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                if calls >= budget:
                    break

            # Aggressive restart if no global improvement
            if no_improve_count >= no_improve_limit:
                best_restart_x = x
                best_restart_f = fx
                n_restarts = 3
                for _ in range(n_restarts):
                    if calls >= budget:
                        break
                    cand = rng.uniform(-5, 5, dim)
                    cand_f = func(cand)
                    calls += 1
                    if cand_f < best_restart_f:
                        best_restart_f = cand_f
                        best_restart_x = cand
                    if cand_f < best_f:
                        best_f = cand_f
                        best_x = cand.copy()

                x = best_restart_x
                fx = best_restart_f
                sigma = 1.5
                stagnation = 0
                no_improve_count = 0
                success_window = []

                if calls >= budget:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
