"""Direct search pattern search with orthogonal/diagonal directions and binary step-size search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Pattern search with fixed orthogonal and diagonal directions, binary step-size refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s limit

        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)
        n = self.dim

        def make_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            fx = func(x)
            call_count += 1
            return fx

        def update_best(x, f):
            nonlocal best_x, best_f
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()

        # Build direction set: n orthogonal unit vectors + diagonal directions
        directions = []
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            directions.append(e)
        # Add diagonal directions (all +1, all -1, and mixed)
        diag_pos = np.ones(n) / np.sqrt(n)
        diag_neg = -diag_pos
        directions.append(diag_pos)
        directions.append(diag_neg)
        # Add some random diagonal-like directions for better coverage
        for _ in range(n):
            d = rng.standard_normal(n)
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                directions.append(d / norm)

        def pattern_search(x0, sigma0, max_calls):
            """
            Pattern search: for each direction, do binary search on step size.
            Returns (best_x_local, best_f_local, calls_made).
            """
            nonlocal best_x, best_f, call_count

            x = x0.copy()
            fx = make_call(x)
            if fx is None:
                return x, np.inf, 1
            if not np.isfinite(fx):
                fx = np.inf
            local_best_x = x.copy()
            local_best_f = fx
            calls = 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = sigma0
            min_sigma = 1e-10
            max_sigma = 10.0

            while calls < max_calls and call_count < self.budget:
                if time.time() > deadline - 2.0:
                    break

                improved = False

                for d in directions:
                    if calls >= max_calls or call_count >= self.budget:
                        break
                    if time.time() > deadline - 2.0:
                        break

                    # Try positive step
                    best_step = 0.0
                    best_val = fx
                    lo = 0.0
                    hi = sigma
                    # Binary search for best step in [0, sigma]
                    for _ in range(6):  # 6 iterations of binary search
                        mid = (lo + hi) / 2.0
                        if mid < min_sigma:
                            break
                        y = x + mid * d
                        y = np.clip(y, -5, 5)
                        fy = make_call(y)
                        calls += 1
                        if fy is None:
                            break
                        if np.isfinite(fy) and fy < best_val:
                            best_val = fy
                            best_step = mid
                            lo = mid
                        else:
                            hi = mid

                    # Try negative step
                    lo = 0.0
                    hi = sigma
                    for _ in range(6):
                        mid = (lo + hi) / 2.0
                        if mid < min_sigma:
                            break
                        y = x - mid * d
                        y = np.clip(y, -5, 5)
                        fy = make_call(y)
                        calls += 1
                        if fy is None:
                            break
                        if np.isfinite(fy) and fy < best_val:
                            best_val = fy
                            best_step = -mid
                            lo = mid
                        else:
                            hi = mid

                    if best_step != 0.0:
                        x = x + best_step * d
                        x = np.clip(x, -5, 5)
                        fx = best_val
                        improved = True

                        if fx < local_best_f:
                            local_best_f = fx
                            local_best_x = x.copy()
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

                if improved:
                    # Shrink sigma slightly to refine
                    sigma *= 0.7
                    if sigma < min_sigma:
                        sigma = min_sigma
                else:
                    # No improvement in any direction; shrink sigma more aggressively
                    sigma *= 0.5
                    if sigma < min_sigma:
                        # Try to escape: jump to a random nearby point
                        if call_count < self.budget - 5:
                            escape = x + rng.standard_normal(n) * 0.3
                            escape = np.clip(escape, -5, 5)
                            fe = make_call(escape)
                            calls += 1
                            if fe is not None and np.isfinite(fe) and fe < fx:
                                x = escape
                                fx = fe
                                if fx < local_best_f:
                                    local_best_f = fx
                                    local_best_x = x.copy()
                                    if fx < best_f:
                                        best_f = fx
                                        best_x = x.copy()
                        break

            return local_best_x, local_best_f, calls

        # Phase 1: Sparse random starts to find promising basins
        num_starts = 6
        calls_per_start = 40
        total_phase1 = num_starts * calls_per_start

        for s in range(num_starts):
            if call_count >= self.budget or time.time() > deadline - 5.0:
                break

            if s == 0:
                x_start = np.zeros(n)
            else:
                x_start = rng.uniform(-4, 4, n)

            sigma_init = 2.0
            pattern_search(x_start, sigma_init, calls_per_start)

        # Phase 2: Exploitation from the best point found
        if best_x is not None:
            remaining = self.budget - call_count
            if remaining > 20:
                # Multiple exploitation runs with decreasing sigma
                for exp_run in range(3):
                    if call_count >= self.budget or time.time() > deadline - 5.0:
                        break
                    max_calls_exp = min(100, remaining - 10)
                    if max_calls_exp < 20:
                        break
                    sigma_exp = 1.0 * (0.5 ** exp_run)
                    pattern_search(best_x, sigma_exp, max_calls_exp)
                    remaining = self.budget - call_count

        # Final fallback
        if best_x is None:
            x = np.zeros(n)
            fx = make_call(x)
            if fx is not None and np.isfinite(fx):
                best_x = x.copy()
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
