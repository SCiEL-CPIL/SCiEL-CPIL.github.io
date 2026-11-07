"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with lattice-based initialization and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Phase 1: Lattice-based deterministic grid search to exploit
        # the periodic structure of Rastrigin. Evaluate points at integer
        # coordinates and half-integer offsets within the box.
        # Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Global minima at integer coordinates (x_i = 0, +/-1, +/-2, ...)
        # With hidden shift/rotation, the lattice structure is transformed,
        # but we can still benefit from a structured initial exploration.
        
        # Generate lattice points: integers and half-integers in [-5, 5]
        # That's 11 * 11 = 121 points per dimension... too many.
        # Let's use a coarser grid: integers only, which is 11 points per dim.
        # 11^5 = 161051 points - way too many.
        
        # Better approach: evaluate a small set of structured points.
        # Use integer coordinates: -4, -3, -2, -1, 0, 1, 2, 3, 4 (9 values)
        # 9^5 = 59049 - still too many.
        
        # Let's use a smaller structured set:
        # Evaluate at origin and unit vectors, plus some random points.
        # Actually, for Rastrigin with hidden shift/rotation, the lattice
        # structure is rotated, so pure lattice search won't align well.
        
        # Instead, use a hybrid: a small number of well-spread deterministic
        # points (Sobol-like or grid) plus random points for the initial phase.
        
        # Phase 1: Coarse exploration - use ~25% of budget
        n_explore = max(10, budget // 4)
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Generate structured initial points: combine a few grid points
        # with random points to cover the space
        # Use a 3x3x3x3x3 grid centered at 0: values in {-2, 0, 2}
        grid_vals = np.array([-2.0, 0.0, 2.0])
        grid_points = np.array(np.meshgrid(*[grid_vals]*dim)).T.reshape(-1, dim)
        
        # Shuffle and take up to n_explore points
        perm = rng.permutation(len(grid_points))
        n_grid = min(n_explore, len(grid_points))
        
        for i in range(n_grid):
            if not time_ok() or calls_made >= budget:
                break
            x = grid_points[perm[i]]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Fill remaining exploration budget with random points
        while calls_made < n_explore and time_ok():
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: (1+1)-ES with adaptive step size using 1/5 success rule
        # Use remaining budget for main optimization
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        
        # Track successes for 1/5 rule
        successes = 0
        total = 0
        window_size = 10

        remaining_budget = budget - calls_made
        for i in range(remaining_budget):
            if not time_ok():
                break
            if calls_made >= budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            total += 1

            if fy < fx:
                x = y
                fx = fy
                successes += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Keep x unchanged
                pass

            # Adaptive sigma update every window_size steps using 1/5 rule
            if total >= window_size:
                rate = successes / total
                if rate > 0.2:
                    sigma *= 1.5
                elif rate < 0.2:
                    sigma *= 0.87
                successes = 0
                total = 0

            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)

            # If sigma becomes too small, we're converged
            if sigma < 1e-10:
                break

        # Phase 3: Final polish with smaller sigma for high-precision refinement
        if calls_made < budget and time_ok():
            sigma = max(1e-4, sigma * 0.1)
            successes = 0
            total = 0
            polish_budget = min(50, budget - calls_made)
            for i in range(polish_budget):
                if not time_ok() or calls_made >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                total += 1

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                if total >= window_size:
                    rate = successes / total
                    if rate > 0.2:
                        sigma *= 1.3
                    elif rate < 0.15:
                        sigma *= 0.7
                    successes = 0
                    total = 0
                sigma = np.clip(sigma, 1e-8, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
