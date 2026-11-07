"""Hybrid axis-based initialization + aggressive (1+1)-ES optimizer for BBOB f11 Discus."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Axis-based initialization followed by aggressive (1+1)-ES for BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Axis-based initialization to identify anisotropic directions
        # Evaluate along each axis at multiple distances to find the narrow "needle"
        x0 = np.zeros(dim)
        f0 = try_eval(x0)
        if f0 is None:
            return best_x, best_f

        # For each axis, probe at several distances to identify where the function drops
        # This helps identify the direction of the narrow axis in Discus
        probe_distances = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        axis_info = []
        
        for i in range(dim):
            axis_min_f = f0
            axis_min_dist = 0.0
            for d in probe_distances:
                for sign in [1.0, -1.0]:
                    xi = np.zeros(dim)
                    xi[i] = sign * d
                    fi = try_eval(xi)
                    if fi is None:
                        break
                    if fi < axis_min_f:
                        axis_min_f = fi
                        axis_min_dist = abs(d)
                        axis_min_sign = sign
                if fi is None:
                    break
            axis_info.append((axis_min_f, i, axis_min_dist, axis_min_sign))
            if call_count >= budget:
                break

        # Sort axes by how much they reduce the function value
        axis_info.sort(key=lambda x: x[0])
        
        # Start from the best point found so far
        x = best_x.copy() if best_x is not None else x0.copy()
        fx = best_f

        # Phase 2: Aggressive (1+1)-ES with adaptive step-size
        # Use a larger initial sigma and more aggressive adaptation
        sigma = 0.5
        success_count = 0
        total_count = 0
        sigma_min = 1e-6
        sigma_max = 10.0
        
        remaining_budget = budget - call_count
        max_es_iter = min(remaining_budget, 300)
        
        for _ in range(max_es_iter):
            if time.time() > deadline - 5:
                break
            if call_count >= budget:
                break
            
            # Generate mutation
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = try_eval(y)
            if fy is None:
                break
            
            total_count += 1
            
            if fy < fx:
                x, fx = y, fy
                success_count += 1
                # Aggressive increase on success
                sigma = min(sigma * 1.3, sigma_max)
            else:
                # Conservative decrease on failure
                sigma = max(sigma * 0.85, sigma_min)
            
            # Adaptive sigma based on success rate over recent iterations
            if total_count > 20:
                success_rate = success_count / total_count
                # Target success rate around 1/3 for (1+1)-ES
                target_rate = 1.0 / 3.0
                if success_rate > target_rate * 1.5:
                    sigma = min(sigma * 1.1, sigma_max)
                elif success_rate < target_rate * 0.5:
                    sigma = max(sigma * 0.9, sigma_min)
            
            # Reset counters periodically to adapt to changing landscape
            if total_count % 50 == 0:
                success_count = 0
                total_count = 0

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
