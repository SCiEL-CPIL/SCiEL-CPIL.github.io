"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        n = self.dim
        max_calls = self.budget
        total_calls = 0
        best_x = None
        best_f = np.inf

        def time_check():
            return time.time() < deadline

        def safe_func(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= max_calls:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            total_calls += 1
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        rng = np.random.default_rng(7)

        # --- Phase 1: Quick random scan to find a good starting region ---
        n_scan = min(25, max_calls // 6)
        scan_points = [np.zeros(n)]
        for _ in range(n_scan - 1):
            scan_points.append(rng.uniform(-4, 4, n))

        scan_fitness = []
        for p in scan_points:
            if total_calls >= max_calls or not time_check():
                break
            f = safe_func(p)
            scan_fitness.append(f)

        # --- Phase 2: Differential Evolution ---
        # DE parameters
        pop_size = max(10, min(30, max_calls // 20))
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        max_generations = 50

        # Initialize population using scan results and random points
        pop = []
        pop_fitness = []

        # Use best scan points as part of initial population
        if scan_fitness:
            sorted_scan = np.argsort(scan_fitness)
            n_from_scan = min(pop_size // 2, len(sorted_scan))
            for idx in sorted_scan[:n_from_scan]:
                pop.append(scan_points[idx].copy())
                pop_fitness.append(scan_fitness[idx])

        # Fill rest with random points
        while len(pop) < pop_size and total_calls < max_calls and time_check():
            x = rng.uniform(-4, 4, n)
            f = safe_func(x)
            if np.isfinite(f):
                pop.append(x)
                pop_fitness.append(f)

        # Ensure we have at least some population
        if len(pop) < 4:
            # Fallback: generate more random points
            while len(pop) < 4 and total_calls < max_calls and time_check():
                x = rng.uniform(-4, 4, n)
                f = safe_func(x)
                if np.isfinite(f):
                    pop.append(x)
                    pop_fitness.append(f)

        if len(pop) < 4:
            # Not enough valid points, just return best found
            if best_x is None:
                best_x = np.zeros(n)
            return best_x, best_f

        pop = np.array(pop)
        pop_fitness = np.array(pop_fitness)

        # DE main loop
        gen = 0
        no_improve_count = 0
        best_f_de = np.inf
        best_x_de = None

        while gen < max_generations and total_calls < max_calls and time_check():
            # Sort population
            order = np.argsort(pop_fitness)
            pop = pop[order]
            pop_fitness = pop_fitness[order]

            # Track best in this generation
            if pop_fitness[0] < best_f_de - 1e-10:
                best_f_de = pop_fitness[0]
                best_x_de = pop[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count > 15:
                break

            # Generate trial population
            new_pop = []
            new_fitness = []

            for i in range(pop_size):
                if total_calls >= max_calls or not time_check():
                    break

                # DE/rand/1: x_trial = x_rand1 + F * (x_rand2 - x_rand3)
                while True:
                    r1 = rng.integers(0, pop_size)
                    r2 = rng.integers(0, pop_size)
                    r3 = rng.integers(0, pop_size)
                    if len(set([i, r1, r2, r3])) == 4:
                        break

                x_base = pop[r1]
                x_diff = pop[r2] - pop[r3]
                x_trial = x_base + F * x_diff

                # Binary crossover
                j_rand = rng.integers(0, n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        x_trial[j] = x_base[j] + F * x_diff[j]
                    else:
                        x_trial[j] = pop[i][j]

                # Clip to bounds
                x_trial = np.clip(x_trial, -5.0, 5.0)

                # Evaluate
                f_trial = safe_func(x_trial)

                if np.isfinite(f_trial):
                    new_pop.append(x_trial)
                    new_fitness.append(f_trial)
                else:
                    # Keep original if trial is invalid
                    new_pop.append(pop[i].copy())
                    new_fitness.append(pop_fitness[i])

            # Replace population with better of trial vs original
            for i in range(len(new_pop)):
                if new_fitness[i] < pop_fitness[i]:
                    pop[i] = new_pop[i]
                    pop_fitness[i] = new_fitness[i]

            gen += 1

        # Update global best from DE
        if best_x_de is not None and best_f_de < best_f:
            best_f = best_f_de
            best_x = best_x_de

        # --- Phase 3: Local refinement with Nelder-Mead ---
        if best_x is not None and total_calls < max_calls and time_check():
            try:
                from scipy.optimize import minimize

                def objective(x):
                    return safe_func(x)

                # Start Nelder-Mead from current best
                res = minimize(objective, best_x, method='Nelder-Mead',
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})

                if res.success and np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)

            except Exception:
                pass

        # --- Phase 4: Final coordinate descent refinement ---
        if best_x is not None and total_calls < max_calls and time_check():
            x_curr = best_x.copy()
            f_curr = best_f
            step = 0.3
            stall_count = 0

            for _ in range(80):
                if total_calls >= max_calls or not time_check():
                    break

                improved = False
                for dim_idx in range(n):
                    for direction in [-1, 1]:
                        x_trial = x_curr.copy()
                        x_trial[dim_idx] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = safe_func(x_trial)
                        if np.isfinite(f_trial) and f_trial < f_curr - 1e-12:
                            x_curr = x_trial
                            f_curr = f_trial
                            improved = True
                            stall_count = 0
                            break
                    if improved:
                        break

                if improved:
                    step = min(step * 1.03, 1.5)
                else:
                    step *= 0.7
                    stall_count += 1
                    if step < 1e-4 or stall_count > 12:
                        break

            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr

        if best_x is None:
            best_x = np.zeros(n)
            if total_calls < max_calls:
                best_f = safe_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
