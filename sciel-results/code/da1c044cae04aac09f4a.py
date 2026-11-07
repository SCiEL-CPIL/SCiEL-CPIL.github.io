"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard for Katsuura (f23)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, fx0, budget, deadline, rng):
        """Simple CMA-ES variant with adaptive sigma."""
        dim = self.dim
        x = x0.copy()
        fx = fx0
        sigma = 1.0
        # CMA-ES parameters
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, (2 + 2.0 / (dim + 1.3) - (dim + 2) / (dim + 1.3)) / ((dim + 2) ** 2 + 2))
        cs = (dim + 1.7) / (dim + 9 + cs) if dim > 0 else 0
        cs = (dim + 1.7) / (dim + 9)
        damps = 1 + 2 * max(0, sqrt((dim - 1) / (dim + 3)) - 1) + cs
        ps = np.zeros(dim)
        C = np.eye(dim)
        inv_sqrt_C = np.eye(dim)
        chi_n = sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        calls = 0
        pop_size = 4 + int(3 * np.log(dim))
        for gen in range(budget // pop_size + 1):
            if time.time() > deadline:
                break
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (inv_sqrt_C @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y.copy(), fy
            # Update CMA-ES state (simplified)
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(max(sigma, 0.01), 10.0)
            if calls >= budget:
                break
        return x, fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # Katsuura f23 has known structure; try informed initial points
        # The global minimum of standard Katsuura is at a specific point
        # Try several strategies: random, grid-like, and informed starts
        candidates = []
        
        # Strategy 1: Pure random multi-start
        for i in range(10):
            x = rng.uniform(-5, 5, dim)
            candidates.append(x)
        
        # Strategy 2: Try points near origin and symmetric points
        for coord in [0.0, -1.0, 1.0, -2.0, 2.0]:
            x = np.full(dim, coord)
            candidates.append(x)
        
        # Strategy 3: Random with smaller variance
        for i in range(5):
            x = rng.normal(0, 1.5, dim)
            x = np.clip(x, -5, 5)
            candidates.append(x)
        
        # Evaluate all candidates, keep best
        best_x = None
        best_fx = float('inf')
        for x in candidates:
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Use remaining budget for CMA-ES refinement
        calls_used = len(candidates)
        remaining_budget = budget - calls_used
        if remaining_budget > 0:
            best_x, best_fx, _ = self._cma_es(func, best_x, best_fx, remaining_budget, deadline, rng)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
