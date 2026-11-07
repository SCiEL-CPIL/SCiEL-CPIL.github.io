"""
Optimization strategy for BBOB f3 (Rastrigin) on [-5,5]^5:
1. Structured initial sampling: integer and near-integer points to target
   Rastrigin's known basin structure, plus LHS for diverse coverage.
2. Multi-restart Adaptive (1+1)-ES for local refinement of top seeds.
   - Uses success-rate based step size adaptation for robustness.
   - Restarts are budgeted to allow exploration of multiple basins.
3. Final high-precision (1+1)-ES polish around the global best.
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

        # Phase 1: Structured sampling targeting Rastrigin's integer basins
        # Rastrigin's global minima are at integer coordinates. With hidden
        # shift/rotation, the true minimum lies in [-4,4]^5. We sample:
        #   a) integer points in [-4,4]^5
        #   b) near-integer points (integer + small offset)
        #   c) LHS points for diverse coverage
        
        n_integer = min(50, budget // 10)
        n_near_int = min(30, budget // 10)
        n_lhs = min(30, budget // 10)
        
        candidates = []
        
        # Integer points in [-4, 4]^5
        for _ in range(n_integer):
            x_int = rng.integers(-4, 5, size=dim).astype(np.float64)
            candidates.append(x_int)
        
        # Near-integer points
        for _ in range(n_near_int):
            x_int = rng.integers(-4, 5, size=dim).astype(np.float64)
            offset = rng.uniform(-0.3, 0.3, size=dim)
            x_near = np.clip(x_int + offset, -5, 5)
            candidates.append(x_near)
        
        # LHS points for diversity
        if n_lhs > 0:
            lhs_points = np.empty((n_lhs, dim))
            for d in range(dim):
                intervals = np.linspace(0, 1, n_lhs + 1)
                points_in_intervals = np.empty(n_lhs)
                for i in range(n_lhs):
                    points_in_intervals[i] = rng.uniform(intervals[i], intervals[i+1])
                rng.shuffle(points_in_intervals)
                lhs_points[:, d] = points_in_intervals * 10.0 - 5.0
            candidates.extend(lhs_points.tolist())
        
        # Add a few fully random points
        n_random = min(10, budget // 20)
        for _ in range(n_random):
            candidates.append(rng.uniform(-5, 5, size=dim).tolist())
        
        # Evaluate all candidates
        candidate_values = []
        for x_cand in candidates:
            if check_deadline():
                break
            x_arr = np.asarray(x_cand, dtype=np.float64).flatten()
            fx = evaluate(x_arr)
            candidate_values.append((x_arr, fx))
        
        # Sort candidates by fitness
        if candidate_values:
            candidate_values.sort(key=lambda t: t[1])
            # Select up to 10 diverse seeds (skip if too close to an earlier seed)
            seeds = []
            for x_cand, f_cand in candidate_values:
                if len(seeds) >= 10:
                    break
                # Check diversity: skip if within 0.5 of an existing seed
                diverse = True
                for x_seed, _ in seeds:
                    if np.linalg.norm(x_cand - x_seed) < 0.5:
                        diverse = False
                        break
                if diverse:
                    seeds.append((x_cand.copy(), f_cand))
            # If we didn't get enough diverse seeds, fill with top candidates
            if len(seeds) < 3:
                seeds = [(x_cand.copy(), f_cand) for x_cand, f_cand in candidate_values[:min(5, len(candidate_values))]]
        else:
            x0 = rng.uniform(-5, 5, dim)
            f0 = evaluate(x0)
            seeds = [(x0, f0)]

        # Phase 2: Multi-restart Adaptive (1+1)-ES
        remaining_budget = budget - calls_made
        polish_budget = max(30, int(remaining_budget * 0.25))
        refine_budget = remaining_budget - polish_budget
        
        if refine_budget > 0 and seeds:
            n_seeds = len(seeds)
            iters_per_seed = max(1, refine_budget // n_seeds)
            
            for x_init, f_init in seeds:
                if check_deadline():
                    break
                
                x = x_init.copy()
                fx = f_init
                sigma = 0.5
                success_count = 0
                total_steps = 0
                no_improve = 0
                max_no_improve = 25
                
                for _ in range(iters_per_seed):
                    if check_deadline():
                        break
                    
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    total_steps += 1
                    
                    if fy < fx:
                        x, fx = y, fy
                        success_count += 1
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    # Adapt sigma every 8 steps
                    if total_steps % 8 == 0:
                        success_rate = success_count / 8.0
                        target = 0.25
                        if success_rate > target:
                            sigma *= 1.3
                        elif success_rate < target:
                            sigma *= 0.8
                        success_count = 0
                        
                    sigma = np.clip(sigma, 1e-6, 5.0)
                    
                    # Early termination if stuck
                    if no_improve > max_no_improve:
                        break
                
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 3: Final Polish with tighter convergence
        if not check_deadline() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            success_count = 0
            total_steps = 0
            no_improve = 0
            max_no_improve = 40
            
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
                
                # Adapt sigma every 4 steps
                if total_steps % 4 == 0:
                    success_rate = success_count / 4.0
                    if success_rate > 0.2:
                        sigma *= 1.15
                    else:
                        sigma *= 0.85
                    success_count = 0
                    
                sigma = np.clip(sigma, 1e-10, 0.5)
                
                # Stop if sigma is very small and no recent improvement
                if sigma < 1e-8 and no_improve > max_no_improve:
                    break
            
            # Final update
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
