"""
Optimization strategy for BBOB f3 (Rastrigin) on [-5,5]^5:
1. Integer-grid sampling to target the known structure where Rastrigin's
   global minima occur at integer coordinates (after hidden shift/rotation).
2. Multi-restart adaptive (1+1)-ES for local refinement of top candidates.
3. Final high-precision polish around the global best.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64).flatten()
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_deadline():
            return time.time() >= deadline or calls_made >= budget

        # Phase 1: Integer grid sampling
        # Rastrigin's global minimum is at integer points. With hidden shift
        # and rotation, the true minimum is at some point in [-4,4]^5.
        # We sample integer and near-integer points to find the right basin.
        
        # Generate candidate integer points in [-4, 4]^5
        # Full grid would be 9^5 = 59049 points, too many.
        # Use random sampling of integer points plus some fractional points.
        
        n_integer_samples = min(80, budget // 5)
        n_fractional_samples = min(30, budget // 10)
        
        candidates = []
        
        # Sample integer points
        for _ in range(n_integer_samples):
            x_int = rng.integers(-4, 5, size=dim).astype(np.float64)
            candidates.append(x_int)
        
        # Sample near-integer points (integer + small offset)
        for _ in range(n_fractional_samples):
            x_int = rng.integers(-4, 5, size=dim).astype(np.float64)
            offset = rng.uniform(-0.3, 0.3, size=dim)
            x_near = np.clip(x_int + offset, -5, 5)
            candidates.append(x_near)
        
        # Add a few random points for diversity
        n_random = min(20, budget // 10)
        for _ in range(n_random):
            x_rand = rng.uniform(-5, 5, size=dim)
            candidates.append(x_rand)
        
        # Evaluate all candidates
        candidate_values = []
        for x_cand in candidates:
            if check_deadline():
                break
            fx = evaluate(x_cand)
            candidate_values.append(fx)
        
        # Sort candidates by fitness
        if candidate_values:
            sorted_indices = np.argsort(candidate_values)
            # Take top K candidates for local refinement
            k_top = min(10, len(sorted_indices))
            top_candidates = [candidates[i] for i in sorted_indices[:k_top]]
        else:
            top_candidates = [rng.uniform(-5, 5, size=dim)]

        # Phase 2: Multi-restart (1+1)-ES for local refinement
        remaining_budget = budget - calls_made
        n_restarts = min(len(top_candidates), max(3, remaining_budget // 50))
        
        for start_idx in range(n_restarts):
            if check_deadline():
                break
            
            x = top_candidates[start_idx].copy()
            fx = evaluate(x)
            
            # Adaptive (1+1)-ES
            sigma = 0.5
            success_count = 0
            total_steps = 0
            no_improve = 0
            max_no_improve = 30
            
            while not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                total_steps += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Adapt sigma every 5 steps
                if total_steps % 5 == 0:
                    success_rate = success_count / 5.0
                    if success_rate > 0.3:
                        sigma *= 1.2
                    elif success_rate < 0.1:
                        sigma *= 0.7
                    else:
                        sigma *= 0.95
                    success_count = 0
                    sigma = np.clip(sigma, 1e-6, 2.0)
                
                # Early termination if stuck
                if no_improve > max_no_improve:
                    break
            
            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 3: Final polish with (1+1)-ES
        if not check_deadline() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            success_count = 0
            total_steps = 0
            no_improve = 0
            max_no_improve = 50
            
            while not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                total_steps += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Adapt sigma every 5 steps
                if total_steps % 5 == 0:
                    success_rate = success_count / 5.0
                    if success_rate > 0.25:
                        sigma *= 1.15
                    elif success_rate < 0.1:
                        sigma *= 0.75
                    else:
                        sigma *= 0.95
                    success_count = 0
                    sigma = np.clip(sigma, 1e-8, 0.5)
                
                # Early termination if stuck
                if no_improve > max_no_improve:
                    break
            
            # Final update
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
