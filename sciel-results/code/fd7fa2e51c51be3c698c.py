"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES style optimizer for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Track best solution and function value
        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            if call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        rng = np.random.default_rng(12345)

        # Phase 1: Broad scout with diverse starting points
        # Rosenbrock has a long narrow valley; we need to find it first.
        # Use a mix of structured and random starts.
        starts = []
        # Nominal minima candidates
        starts.append(np.zeros(self.dim))
        starts.append(np.ones(self.dim))
        starts.append(np.full(self.dim, 1.0))
        starts.append(np.full(self.dim, 0.5))
        starts.append(np.full(self.dim, -0.5))
        # Grid-like points
        for v in [-3, -2, -1, 0, 1, 2, 3]:
            starts.append(np.full(self.dim, float(v)))
        # Random points
        for _ in range(12):
            starts.append(rng.uniform(-4.5, 4.5, self.dim))
        # Points along diagonal and anti-diagonal
        starts.append(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
        starts.append(np.array([1.0, 1.0, 0.5, 0.5, 0.5]))
        starts.append(np.array([0.5, 0.5, 1.0, 1.0, 1.0]))

        # Evaluate all starting points
        for s in starts:
            if call_count >= self.budget or time.time() > deadline:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            f_val = wrapped(s_clipped)

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        # Phase 2: Multiple Powell restarts from top diverse scout points
        # Collect top diverse points
        scored_starts = []
        for s in starts:
            if call_count >= self.budget or time.time() > deadline:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            # We already evaluated these, so just use them
            scored_starts.append((best_f, s_clipped))

        # Use the best few points as Powell starts
        # Powell is good for finding the valley floor in Rosenbrock
        powell_starts = [best_x.copy()]
        # Add a few other good candidates
        for s in starts:
            if len(powell_starts) >= 3:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            if np.linalg.norm(s_clipped - best_x) > 1.0:
                powell_starts.append(s_clipped)

        for ps in powell_starts:
            if call_count >= self.budget - 30 or time.time() > deadline:
                break
            # Run Powell from this start
            maxfev = max(10, (self.budget - call_count - 50) // len(powell_starts))
            result = minimize(
                wrapped,
                ps.copy(),
                method='Powell',
                options={
                    'maxiter': 100,
                    'maxfev': maxfev,
                    'xtol': 1e-10,
                    'ftol': 1e-12,
                }
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5.0, 5.0)

        # Phase 3: CMA-ES-like local refinement
        # Use a population-based approach for robust local optimization
        if call_count < self.budget - 20 and time.time() < deadline:
            dim = self.dim
            # Population size for CMA-ES
            lambda_es = max(4, int(4 + 3 * np.log(dim)))
            # Initial sigma based on problem scale
            sigma = 0.3
            x_mean = best_x.copy()
            f_mean = best_f
            C = np.eye(dim)
            # Simple covariance update
            step_size = sigma
            
            es_calls = 0
            max_es_calls = min(150, self.budget - call_count - 10)
            no_improve = 0
            best_sigma = sigma

            for gen in range(max_es_calls):
                if call_count >= self.budget - 5 or time.time() > deadline:
                    break
                
                # Generate population
                offspring = []
                for i in range(lambda_es):
                    y = rng.standard_normal(dim)
                    x_off = np.clip(x_mean + step_size * y, -5.0, 5.0)
                    offspring.append(x_off)
                
                # Evaluate offspring
                offspring_f = []
                for x_off in offspring:
                    if call_count >= self.budget or time.time() > deadline:
                        break
                    f_off = wrapped(x_off)
                    offspring_f.append(f_off)
                
                if len(offspring_f) == 0:
                    break
                
                # Sort by fitness
                idx = np.argsort(offspring_f)
                sorted_offspring = [offspring[i] for i in idx]
                sorted_f = [offspring_f[i] for i in idx]
                
                # Update mean (weighted average of top half)
                mu = lambda_es // 2
                weights = np.ones(mu)
                weights[0] = 2.0  # Extra weight for best
                weights /= np.sum(weights)
                
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * sorted_offspring[i]
                
                # Update sigma
                f_new_mean = sorted_f[0]
                if f_new_mean < f_mean:
                    step_size = min(step_size * 1.1, 5.0)
                    no_improve = 0
                else:
                    step_size = max(step_size * 0.9, 0.001)
                    no_improve += 1
                
                x_mean = new_mean
                f_mean = sorted_f[0]
                
                if f_mean < best_f:
                    best_f = f_mean
                    best_x = sorted_offspring[0].copy()
                
                es_calls += 1
                if no_improve > 30:
                    break

        # Phase 4: Final Nelder-Mead polish
        if call_count < self.budget - 5 and time.time() < deadline:
            result2 = minimize(
                wrapped,
                best_x.copy(),
                method='Nelder-Mead',
                options={
                    'maxiter': 50,
                    'maxfev': max(5, self.budget - call_count),
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                }
            )
            if result2.fun < best_f:
                best_f = result2.fun
                best_x = np.clip(result.x, -5.0, 5.0)

        # Phase 5: Final micro-step refinement
        if call_count < self.budget and time.time() < deadline:
            x_es = best_x.copy()
            fx_es = best_f
            sigma = 0.005
            max_es_calls = min(20, self.budget - call_count)
            
            for _ in range(max_es_calls):
                if call_count >= self.budget or time.time() > deadline:
                    break
                y = np.clip(x_es + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped(y)
                if fy < fx_es:
                    x_es, fx_es = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                if fx_es < best_f:
                    best_f = fx_es
                    best_x = x_es.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
