import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return (time.time() - start_time) < deadline

        best_x = np.full(dim, 0.0)
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            if not time_ok() or calls_used >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = func(x)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse grid search exploiting Rastrigin's periodic structure
        # Rastrigin minima are near integer coordinates. Sample at integers and half-integers.
        # With dim=5, we can't do full grid. Use a structured sampling.
        n_grid = 60
        grid_points = []
        # Sample at integer and half-integer positions
        coords = []
        for c in range(-4, 5):
            coords.append(float(c))
            coords.append(float(c) + 0.5)
        coords = np.array(coords)
        
        # Randomly sample from grid to get n_grid points
        # Use a space-filling approach: sample combinations
        for _ in range(n_grid):
            pt = np.array([rng.choice(coords) for _ in range(dim)])
            grid_points.append(pt)
        
        # Also add center and some random points
        grid_points.append(np.zeros(dim))
        for _ in range(10):
            grid_points.append(rng.uniform(-4, 4, dim))
        
        for pt in grid_points:
            if not time_ok() or calls_used >= budget:
                break
            f = evaluate(pt)
            if f is None:
                break

        # Phase 2: Adaptive (1+1)-ES with periodic jumps
        # Start from the best point found
        x = best_x.copy()
        sigma = 1.5  # Initial step size
        
        # 1/5 rule parameters
        sigma_increase = 1.1
        sigma_decrease = 0.9
        jump_interval = 30  # Every N evaluations, consider a jump
        jump_size = 2.0
        stagnation_count = 0
        stagnation_limit = 40
        
        evals_since_jump = 0
        
        while time_ok() and calls_used < budget:
            # Generate one offspring
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5.0, 5.0)
            
            fy = evaluate(y)
            if fy is None:
                break
            
            # 1/5 rule adaptation
            if fy < best_f:
                # Improvement: increase sigma
                sigma *= sigma_increase
                x = y.copy()
                stagnation_count = 0
            else:
                # No improvement: decrease sigma
                sigma *= sigma_decrease
                stagnation_count += 1
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)
            
            evals_since_jump += 1
            
            # Periodic jump to escape local minima
            if evals_since_jump >= jump_interval:
                evals_since_jump = 0
                # Jump from current best with moderate size
                jump = rng.uniform(-jump_size, jump_size, dim)
                jump_x = best_x + jump
                jump_x = np.clip(jump_x, -5.0, 5.0)
                fj = evaluate(jump_x)
                if fj is not None and fj < best_f:
                    x = jump_x.copy()
                    sigma = 1.0  # Reset sigma after successful jump
                    stagnation_count = 0
                
                # If stuck for too long, do a larger random restart
                if stagnation_count >= stagnation_limit:
                    # Large jump or random restart
                    if rng.random() < 0.5:
                        # Large jump from best
                        jump = rng.uniform(-3.0, 3.0, dim)
                        jump_x = best_x + jump
                        jump_x = np.clip(jump_x, -5.0, 5.0)
                    else:
                        # Random point near center
                        jump_x = rng.uniform(-2, 2, dim)
                    
                    fj = evaluate(jump_x)
                    if fj is not None:
                        if fj < best_f:
                            x = jump_x.copy()
                        else:
                            x = best_x.copy()
                        sigma = 1.5
                        stagnation_count = 0

        return best_x, best_f
