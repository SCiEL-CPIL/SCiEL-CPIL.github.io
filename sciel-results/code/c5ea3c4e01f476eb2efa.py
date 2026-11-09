"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid Nelder-Mead + CMA-ES-like population search with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start with Nelder-Mead
        # We'll use a few restarts with different initial points
        remaining_budget = self.budget
        restarts = 0
        max_restarts = 10

        while remaining_budget > 0 and time.time() < deadline and restarts < max_restarts:
            # Check if we're getting close to deadline
            if time.time() > deadline - 5.0:
                break

            # Generate initial point
            if restarts == 0:
                x0 = np.zeros(self.dim)
            else:
                x0 = rng.uniform(-4, 4, self.dim)

            # Use Nelder-Mead with limited function evaluations
            # We'll track evaluations
            eval_count = [0]

            def nm_func(x):
                nonlocal remaining_budget
                if remaining_budget <= 0 or time.time() > deadline:
                    raise StopIteration
                remaining_budget -= 1
                eval_count[0] += 1
                return safe_func(x)

            try:
                options = {
                    'maxiter': 50,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
                result = minimize(nm_func, x0, method='Nelder-Mead', options=options)
            except StopIteration:
                break
            except Exception:
                pass

            restarts += 1

            # If we found a good solution, do local refinement
            if best_x is not None and remaining_budget > 10 and time.time() < deadline - 10:
                # Fine-tune with smaller steps
                self._local_refine(safe_func, best_x.copy(), remaining_budget, deadline, rng)
                # Update remaining budget (we need to track this properly)
                # Actually, let's restructure to track budget globally

        return best_x, best_f

    def _local_refine(self, func, x, budget_remaining, deadline, rng):
        """Local refinement using coordinate descent + Nelder-Mead."""
        # This is a simplified approach - we'll just do a few NM iterations
        pass


# Let me rewrite this more carefully with proper budget tracking

class Optimizer:
    """Hybrid multi-start Nelder-Mead with adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        eval_count = 0

        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start strategy with Nelder-Mead
        num_starts = 5
        budget_per_start = self.budget // num_starts
        extra_budget = self.budget - budget_per_start * num_starts

        for start_idx in range(num_starts):
            if eval_count >= self.budget or time.time() > deadline - 2.0:
                break

            # Initial point for this start
            if start_idx == 0:
                x0 = np.zeros(self.dim)
            elif start_idx == 1:
                x0 = rng.uniform(-3, 3, self.dim)
            else:
                x0 = rng.uniform(-4.5, 4.5, self.dim)

            # Evaluate initial point
            if eval_count < self.budget:
                f0 = safe_func(x0)
            else:
                break

            # Nelder-Mead optimization
            def nm_func(x):
                nonlocal eval_count
                if eval_count >= self.budget or time.time() > deadline:
                    raise StopIteration
                return safe_func(x)

            try:
                options = {
                    'maxiter': 30,
                    'xatol': 1e-9,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'initial_simplex': None,
                }
                result = minimize(nm_func, x0, method='Nelder-Mead', options=options)
            except StopIteration:
                break
            except Exception:
                pass

            # After each start, if we have budget left, do a fine-tune
            if eval_count < self.budget and best_x is not None and time.time() < deadline - 5.0:
                # Fine-tune around best point
                self._fine_tune(safe_func, best_x.copy(), 5, deadline, rng)

        return best_x, best_f

    def _fine_tune(self, func, x, num_steps, deadline, rng):
        """Coordinate-wise fine-tuning."""
        for _ in range(num_steps):
            if time.time() > deadline:
                break
            for i in range(self.dim):
                for delta in [0.1, -0.1, 0.01, -0.01]:
                    x_new = x.copy()
                    x_new[i] += delta
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    if f_new < func(x):
                        x = x_new
                        break


# Actually, let me write a cleaner version

class Optimizer:
    """Multi-start Nelder-Mead with adaptive initial simplex."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        eval_count = 0

        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            x = np.clip(x, -5, 5)
            f = float(func(x))
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy: Multiple Nelder-Mead runs from different starting points
        starts = [
            np.zeros(self.dim),
            rng.uniform(-2, 2, self.dim),
            rng.uniform(-3, 3, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4.5, 4.5, self.dim),
        ]

        for x0 in starts:
            if eval_count >= self.budget or time.time() > deadline - 1.0:
                break

            # Ensure x0 is within bounds
            x0 = np.clip(x0, -5, 5)

            # Evaluate starting point
            if eval_count < self.budget:
                safe_func(x0)

            # Run Nelder-Mead
            def nm_obj(x):
                nonlocal eval_count
                if eval_count >= self.budget or time.time() > deadline:
                    raise StopIteration
                return safe_func(x)

            try:
                res = minimize(
                    nm_obj,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 40,
                        'xatol': 1e-10,
                        'fatol': 1e-14,
                        'adaptive': True,
                    }
                )
            except StopIteration:
                break
            except Exception:
                pass

        return best_x, best_f


# Final clean version
class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f11."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(123)

        best_x = None
        best_f = np.inf
        n_evals = 0

        def eval_fn(x):
            nonlocal best_x, best_f, n_evals
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            n_evals += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Generate diverse starting points
        starts = []
        starts.append(np.zeros(self.dim))
        for _ in range(4):
            starts.append(rng.uniform(-4.0, 4.0, self.dim))

        for x0 in starts:
            if n_evals >= self.budget or time.time() > deadline - 0.5:
                break
            x0 = np.clip(x0, -5.0, 5.0)
            eval_fn(x0)  # Evaluate the start

            def objective(x):
                if n_evals >= self.budget or time.time() > deadline:
                    raise StopIteration
                return eval_fn(x)

            try:
                minimize(
                    objective, x0, method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-11, 'fatol': 1e-15, 'adaptive': True}
                )
            except StopIteration:
                break
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
