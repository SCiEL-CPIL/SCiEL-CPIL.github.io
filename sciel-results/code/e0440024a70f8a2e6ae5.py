"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and strict wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Counting wrapper
        call_count = 0

        def counted_func(x):
            nonlocal call_count
            call_count += 1
            return func(np.clip(x, -5.0, 5.0))

        def remaining_time():
            return deadline - time.time()

        # Phase 1: Efficient diverse initial evaluation
        # Use a smaller set of well-chosen starting points to conserve budget
        starts = []
        starts.append(np.zeros(dim))
        
        # Key vertices and midpoints (not all 32 vertices)
        key_points = [
            np.array([4.0] * dim),
            np.array([-4.0] * dim),
            np.array([3.0] * dim),
            np.array([-3.0] * dim),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0]),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0]),
            np.array([1.0, -1.0, 1.0, -1.0, 1.0]),
            np.array([-1.0, 1.0, -1.0, 1.0, -1.0]),
        ]
        starts.extend(key_points)
        
        # Fewer random points to save budget
        for _ in range(5):
            starts.append(rng.uniform(-4.0, 4.0, dim))

        best_x = None
        best_f = np.inf
        evaluated = []

        for x0 in starts:
            if call_count >= budget or remaining_time() < 10.0:
                break
            f0 = counted_func(x0)
            evaluated.append((x0.copy(), f0))
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = counted_func(best_x)
            evaluated.append((best_x.copy(), best_f))

        # Phase 2: Local refinement using (1+1)-ES from the best starting points
        # Sort evaluated points by fitness and try to refine the top few
        evaluated.sort(key=lambda t: t[1])
        
        tried_starts = set()
        max_starts_to_refine = min(5, len(evaluated))  # Limit to top 5 to save budget
        
        for idx, (x0, f0) in enumerate(evaluated[:max_starts_to_refine]):
            if call_count >= budget or remaining_time() < 15.0:
                break
            
            # Skip if we've already tried a nearby point
            key = tuple(np.round(x0, 2))
            if key in tried_starts:
                continue
            tried_starts.add(key)
            
            # Run (1+1)-ES refinement from this point with better adaptation
            x = x0.copy()
            fx = f0
            sigma = 0.8
            min_sigma = 1e-10
            max_sigma = 5.0
            success_count = 0
            total_count = 0
            target_success_rate = 1.0 / 5.0  # 1/5th success rule
            
            while call_count < budget and remaining_time() > 5.0:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = counted_func(y)
                total_count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Try a reflection if we hit a boundary
                    if np.any(y == np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)):
                        pass  # Just continue with normal update
                
                # Adapt sigma based on success rate over last 10 steps
                if total_count % 10 == 0:
                    recent_success_rate = success_count / total_count
                    if recent_success_rate > target_success_rate:
                        sigma *= 1.1
                    elif recent_success_rate < target_success_rate:
                        sigma *= 0.9
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    # Reset counters to track recent performance
                    success_count = 0
                    total_count = 0
                    
                if sigma < min_sigma * 10:
                    break
        
        # Phase 3: Final high-precision polish from the best point
        if call_count < budget and remaining_time() > 10.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            min_sigma = 1e-12
            max_sigma = 1.0
            success_count = 0
            total_count = 0
            target_success_rate = 1.0 / 5.0
            
            while call_count < budget and remaining_time() > 5.0:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = counted_func(y)
                total_count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                
                # Adapt sigma
                if total_count % 10 == 0:
                    recent_success_rate = success_count / total_count
                    if recent_success_rate > target_success_rate:
                        sigma *= 1.05
                    elif recent_success_rate < target_success_rate:
                        sigma *= 0.95
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    success_count = 0
                    total_count = 0
                    
                if sigma < min_sigma * 10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
