"""STRUCTURAL CHANGE: Multi-start L-BFGS-B with Rastrigin-aware initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer with Rastrigin-aware basin initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s kill limit

        call_count = 0
        max_calls = self.budget
        best_x = np.zeros(self.dim)
        best_f = float('inf')

        def counted_func(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")

            call_count += 1
            x = np.asarray(x, dtype=np.float64).ravel()[:self.dim]
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))

            if f < best_f:
                best_f = f
                best_x = x.copy()

            return f

        # Phase 1: Rastrigin-aware initialization
        # Büche-Rastrigin has local minima near integer coordinates
        # Generate candidate starting points at integer lattice points and nearby
        rng = np.random.RandomState(42)

        # Integer lattice points in [-4, 4]^5
        lattice = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                for k in range(-4, 5):
                    for l in range(-4, 5):
                        for m in range(-4, 5):
                            lattice.append([float(i), float(j), float(k), float(l), float(m)])
        lattice = np.array(lattice)

        # Shuffle and select top candidates
        perm = rng.permutation(len(lattice))
        num_starts = min(100, len(lattice), max_calls // 5)

        # Evaluate lattice points to find promising basins
        evaluated_lattice = []
        for i in range(num_starts):
            if call_count >= max_calls or time.time() > deadline:
                break
            x = lattice[perm[i]]
            try:
                f = counted_func(x)
                evaluated_lattice.append((f, x))
            except Exception:
                break

        # Sort by function value
        evaluated_lattice.sort(key=lambda t: t[0])

        # Phase 2: Multi-start L-BFGS-B from top lattice points
        # Use the best lattice points as starting points for local optimization
        num_local = min(30, len(evaluated_lattice), max(10, max_calls // 10))

        for i in range(num_local):
            if call_count >= max_calls or time.time() > deadline:
                break

            f_init, x_init = evaluated_lattice[i]

            # Try L-BFGS-B from this starting point
            try:
                res = minimize(
                    lambda x: counted_func(x),
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * self.dim,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
            except Exception:
                pass

            # If we found a good solution, also try Nelder-Mead polish
            if call_count < max_calls and time.time() < deadline and best_f < float('inf'):
                try:
                    res2 = minimize(
                        lambda x: counted_func(x),
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                except Exception:
                    pass

        # Phase 3: Additional random starts if budget remains
        if call_count < max_calls and time.time() < deadline:
            num_random = min(20, max(5, (max_calls - call_count) // 20))
            for i in range(num_random):
                if call_count >= max_calls or time.time() > deadline:
                    break

                x_rand = rng.uniform(-5.0, 5.0, size=self.dim)
                try:
                    f = counted_func(x_rand)
                    # Quick local refinement
                    res = minimize(
                        lambda x: counted_func(x),
                        x_rand,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * self.dim,
                        options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6}
                    )
                except Exception:
                    pass

        # Phase 4: Final polish
        if call_count < max_calls and time.time() < deadline and best_f < float('inf'):
            try:
                res = minimize(
                    lambda x: counted_func(x),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12}
                )
            except Exception:
                pass

            # One more L-BFGS-B pass
            if call_count < max_calls and time.time() < deadline:
                try:
                    res = minimize(
                        lambda x: counted_func(x),
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * self.dim,
                        options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
