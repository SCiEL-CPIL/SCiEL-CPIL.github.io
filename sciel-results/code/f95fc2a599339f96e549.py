"""Model-and-optimize optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lightweight (1+lambda)-ES with adaptive step size for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Phase 1: Initial Sampling to find a good start and estimate scale ---
        # We use a mix of random points and a sparse grid to cover the space better
        # than pure random, helping to identify the general direction of the optimum.
        n_samples = 40  # Slightly more than inspiration to be safer
        points = []
        
        # 1. Random points
        for _ in range(20):
            points.append(rng.uniform(-4, 4, dim))
        
        # 2. Center and axis-aligned points (sparse grid)
        points.append(np.zeros(dim))
        for i in range(dim):
            for d in [1.5, 3.0, -1.5, -3.0]:
                p = np.zeros(dim)
                p[i] = d
                points.append(p.copy())
        
        # 3. A few more random points near the center
        for _ in range(10):
            points.append(rng.uniform(-2, 2, dim))

        evals = 0
        values = []
        for p in points:
            if evals >= budget:
                break
            fp = func(np.clip(p, -5, 5))
            values.append(fp)
            evals += 1
        
        if evals >= budget or not values:
            return np.zeros(dim), 0.0

        # Find best so far
        best_idx = int(np.argmin(values))
        best_x = np.array(points[best_idx])
        best_f = values[best_idx]

        # Estimate initial sigma from spread of top points
        sorted_idx = np.argsort(values)
        top_k = min(10, len(values))
        top_points = np.array([points[i] for i in sorted_idx[:top_k]])
        
        if len(top_points) > 1:
            # Calculate average distance from best to other top points
            distances = np.linalg.norm(top_points[1:] - top_points[0], axis=1)
            sigma = max(0.2, np.mean(distances) * 0.5)
        else:
            sigma = 1.0

        # --- Phase 2: Main Optimization Loop (1+lambda)-ES ---
        # This is the core engine. It is fast and effective for smooth functions like Discus.
        lambda_pop = 8  # Population size per generation
        
        # Adaptive sigma parameters
        c_sigma_up = 1.1
        c_sigma_down = 0.9
        success_window = 10
        successes = []
        
        stall_count = 0
        stall_threshold = 50 # Generations without improvement before restarting

        while evals < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(lambda_pop):
                if evals >= budget:
                    break
                # Sample around current mean
                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                offspring.append(y)
                offspring_f.append(fy)
            
            if not offspring:
                break
            
            # Find best offspring
            best_off_idx = int(np.argmin(offspring_f))
            best_off_x = offspring[best_off_idx]
            best_off_f = offspring_f[best_off_idx]
            
            # Update best
            improved = False
            if best_off_f < best_f:
                best_x = best_off_x
                best_f = best_off_f
                improved = True
                successes.append(1)
                # Increase sigma on success
                sigma *= c_sigma_up
                stall_count = 0
            else:
                successes.append(0)
                # Decrease sigma on failure
                sigma *= c_sigma_down
                stall_count += 1

            # Trim success window
            if len(successes) > success_window:
                successes.pop(0)
            
            # Fine-tune sigma based on success rate (1/5 rule variant)
            if len(successes) == success_window:
                rate = np.mean(successes)
                if rate > 0.3:
                    sigma *= 1.02  # Slight boost if many successes
                elif rate < 0.1:
                    sigma *= 0.98  # Slight reduction if many failures
            
            # Ensure sigma stays in reasonable bounds
            sigma = np.clip(sigma, 0.001, 10.0)

            # Restart mechanism if stalled
            if stall_count >= stall_threshold:
                # Perturb the best point and reset sigma
                best_x = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                sigma = 1.0
                stall_count = 0
                successes = []
                
                # Evaluate the new starting point to ensure we have a valid best_f
                # (Optional, but good for consistency. We just use the previous best_f as reference)
                # Actually, we don't need to evaluate immediately, the next loop will.
                # But to be safe, let's just reset the stall counter.

        return best_x, best_f
# EVOLVE-BLOCK-END
