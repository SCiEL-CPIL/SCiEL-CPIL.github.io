"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid search + fixed-step local hill climb."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        n = self.dim
        budget = self.budget

        # Track calls and best
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Deterministic grid search over vertices of hypercube
        # Evaluate all 2^5 = 32 vertices of [-5,5]^5 plus origin
        vertices = []
        for i in range(2 ** n):
            v = np.array([(i >> j) & 1 for j in range(n)], dtype=float)
            v = v * 10.0 - 5.0  # map {0,1}^n to {-5,5}^n
            vertices.append(v)
        vertices.append(np.zeros(n))  # origin

        for v in vertices:
            if time.time() > deadline:
                break
            safe_call(v)

        # Phase 2: Fixed-step local hill climb from best vertex
        # Use decreasing step sizes for refinement
        if best_x is not None:
            # Step sizes: coarse to fine
            step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            for step in step_sizes:
                if time.time() > deadline or calls >= budget:
                    break
                
                # Try all coordinate directions (positive and negative)
                improved = True
                while improved and time.time() < deadline and calls < budget:
                    improved = False
                    for d in range(n):
                        for sign in [1, -1]:
                            if calls >= budget or time.time() > deadline:
                                break
                            y = best_x.copy()
                            y[d] += sign * step
                            y = np.clip(y, -5.0, 5.0)
                            fy = safe_call(y)
                            if fy is not None and fy < best_f:
                                best_x = y.copy()
                                best_f = fy
                                improved = True

            # Additional refinement with smaller steps if budget remains
            if time.time() < deadline and calls < budget:
                for step in [0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]:
                    if time.time() > deadline or calls >= budget:
                        break
                    improved = True
                    while improved and time.time() < deadline and calls < budget:
                        improved = False
                        for d in range(n):
                            for sign in [1, -1]:
                                if calls >= budget or time.time() > deadline:
                                    break
                                y = best_x.copy()
                                y[d] += sign * step
                                y = np.clip(y, -5.0, 5.0)
                                fy = safe_call(y)
                                if fy is not None and fy < best_f:
                                    best_x = y.copy()
                                    best_f = fy
                                    improved = True

        # If we still have budget and time, do a final random exploration
        # to potentially escape local minima
        if time.time() < deadline and calls < budget:
            rng = np.random.default_rng(42)
            for _ in range(5):
                if calls >= budget or time.time() > deadline:
                    break
                # Random restart with small perturbation around best
                y = best_x + rng.standard_normal(n) * 0.5
                y = np.clip(y, -5.0, 5.0)
                fy = safe_call(y)
                if fy is not None and fy < best_f:
                    best_x = y.copy()
                    best_f = fy

        return best_x, best_f
# EVOLVE-BLOCK-END
