"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid: multi-start exploration + (1+1)-ES with adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        calls_made = 0

        # Phase 1: Multi-start exploration
        # Sample a diverse set of points to find a good starting region
        n_explore = min(30, budget // 10)
        candidates = []

        # Always include origin as a candidate
        x0 = np.zeros(dim)
        fx0 = func(x0)
        calls_made += 1
        candidates.append((fx0, x0.copy()))

        # Add random points in [-4, 4]^5 (where minimizer is guaranteed to be)
        for i in range(n_explore - 1):
            if time.time() >= deadline or calls_made >= budget:
                break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            calls_made += 1
            candidates.append((fx, x.copy()))

        # Sort by function value
        candidates.sort(key=lambda t: t[0])
        best_x, best_f = candidates[0][1], candidates[0][0]

        # Phase 2: (1+1)-ES from the best starting point
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Start with a moderate step size
        n_success = 0
        n_total = 0
        window_size = 8

        # Also keep track of the global best
        global_best_x = best_x.copy()
        global_best_f = best_f

        for i in range(budget - calls_made):
            if time.time() >= deadline:
                break

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                n_success += 1
                # Update global best
                if fy < global_best_f:
                    global_best_f = fy
                    global_best_x = y.copy()
            n_total += 1

            # Adapt sigma based on success rate over the window
            if n_total >= window_size:
                success_rate = n_success / n_total
                # Target success rate around 1/5 = 0.2
                if success_rate > 0.25:
                    sigma *= 1.15
                elif success_rate < 0.15:
                    sigma *= 0.85
                # else: keep same
                n_success = 0
                n_total = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.001, 5.0)

            # Stagnation check: if no improvement in a while, try a restart
            # from another candidate
            if n_total > 0 and n_success == 0 and n_total > 20:
                # Try restarting from the next best candidate
                improved = False
                for cand_f, cand_x in candidates[1:]:
                    if cand_f < global_best_f * 1.01:  # within 1% of global best
                        # Perturb and try
                        x_new = np.clip(cand_x + 0.5 * rng.standard_normal(dim), -5, 5)
                        fx_new = func(x_new)
                        calls_made += 1
                        if fx_new < global_best_f:
                            global_best_f = fx_new
                            global_best_x = x_new.copy()
                            x = x_new.copy()
                            fx = fx_new
                            sigma = 1.0
                            n_success = 0
                            n_total = 0
                            improved = True
                            break
                if not improved:
                    # Shrink sigma aggressively
                    sigma *= 0.5
                    n_success = 0
                    n_total = 0

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
