import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        calls_used = 0

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used
            if calls_used >= budget or check_deadline():
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            return fx

        # Phase 1: Random Coordinate Descent with adaptive step sizes
        # This is a structural change from population-based methods to
        # coordinate-wise search, which can be effective for Rosenbrock's valley

        # Initialize at origin
        x = np.zeros(n)
        best_x = x.copy()
        best_f = do_call(x)
        calls_used = 1

        # Per-coordinate step sizes, initialized to cover the box
        sigma = np.full(n, 2.0)
        min_sigma = 1e-10
        max_sigma = 6.0

        # Track consecutive failures per coordinate to adapt step size
        fail_count = np.zeros(n, dtype=int)
        max_fail = 30

        # Phase 1: RCD for ~70% of budget
        rcd_budget = int(budget * 0.70)
        remaining = rcd_budget - calls_used

        while remaining > 0 and not check_deadline():
            # Select coordinate: weighted by inverse step size (finer coords explored more)
            # But also ensure diversity: use a mix of random and weighted selection
            if rng.random() < 0.7:
                # Weighted selection: prefer coordinates with larger sigma (less explored)
                weights = sigma / (sigma + 0.1)
                weights /= weights.sum()
                coord = rng.choice(n, p=weights)
            else:
                coord = rng.randint(n)

            # Generate perturbation
            delta = sigma[coord] * rng.standard_normal()
            y = x.copy()
            y[coord] += delta
            y = np.clip(y, -5, 5)

            fy = do_call(y)
            remaining -= 1

            if fy < best_f - 1e-12:
                # Accept
                x = y
                best_x = x.copy()
                best_f = fy
                # Increase step size slightly on success
                sigma[coord] = min(sigma[coord] * 1.1, max_sigma)
                fail_count[coord] = 0
            elif fy < x_f + 1e-12 if hasattr(self, '_x_f') else fy < best_f + 1e-8:
                # Marginal: accept if not worse than current
                pass
            else:
                # Reject: shrink step size
                sigma[coord] = max(sigma[coord] * 0.9, min_sigma)
                fail_count[coord] += 1

            # If a coordinate has failed too many times, reset its step size
            if fail_count[coord] > max_fail:
                sigma[coord] = 2.0
                fail_count[coord] = 0

            # Periodically try a full random restart if stuck
            if remaining % 100 == 0 and best_f > 10:
                x_rand = rng.uniform(-4, 4, n)
                fr = do_call(x_rand)
                remaining -= 1
                if fr < best_f:
                    x = x_rand
                    best_x = x.copy()
                    best_f = fr
                    sigma = np.full(n, 2.0)
                    fail_count = np.zeros(n, dtype=int)

        # Phase 2: Local refinement with Nelder-Mead from best point
        remaining_budget = budget - calls_used
        if remaining_budget > 30 and not check_deadline():
            from scipy.optimize import minimize

            def objective(x):
                nonlocal calls_used
                if check_deadline() or calls_used >= budget:
                    raise ValueError("budget exceeded")
                val = func(np.clip(x, -5, 5))
                calls_used += 1
                return val

            try:
                res = minimize(
                    objective,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining_budget - 10, 300),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 3: If budget remains, do a few more RCD steps with tiny steps
        remaining_budget = budget - calls_used
        if remaining_budget > 5 and not check_deadline():
            sigma = np.full(n, 0.01)
            for _ in range(min(remaining_budget, 50)):
                if check_deadline() or calls_used >= budget - 1:
                    break
                coord = rng.randint(n)
                delta = sigma[coord] * rng.standard_normal()
                y = best_x.copy()
                y[coord] += delta
                y = np.clip(y, -5, 5)
                fy = do_call(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

        return best_x, best_f
