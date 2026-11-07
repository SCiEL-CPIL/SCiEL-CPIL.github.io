"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using analytical front sampling and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        rng = np.random.default_rng(42)
        used = 0
        all_F = []

        def eval_point(x):
            nonlocal used
            fx = problem(x)
            used += 1
            all_F.append(fx)
            return fx

        # Phase 1: Dense sampling on the DTLZ2 Pareto front
        # For DTLZ2 with M=3:
        # f1 = (1+g)*cos(t1)*cos(t2)
        # f2 = (1+g)*sin(t1)*cos(t2)  
        # f3 = (1+g)*sin(t2)
        # where t1, t2 in [0, pi/2]
        # The first (M-1)=2 variables determine the angles, rest affect g
        
        # We'll sample many points on the front and refine each
        n_front = min(80, self.budget // 3)
        
        # Generate well-distributed angles using stratified sampling
        angles = []
        for i in range(n_front):
            # Use a grid-like approach for better coverage
            t1 = (i % 10) * (np.pi/2) / 10 + rng.uniform(0, np.pi/20)
            t2 = (i // 10) * (np.pi/2) / 10 + rng.uniform(0, np.pi/20)
            t1 = np.clip(t1, 0, np.pi/2)
            t2 = np.clip(t2, 0, np.pi/2)
            angles.append((t1, t2))
        
        for t1, t2 in angles:
            if time.time() > deadline or used >= self.budget:
                break
            
            # Initialize x: set angle variables and g-minimizing values
            x = np.zeros(dim)
            # For DTLZ2, x[0] and x[1] control the angles
            # The exact mapping depends on the problem definition
            # Common: x[0] -> t1, x[1] -> t2 via sin/cos
            # We'll use a simple linear mapping and refine
            x[0] = t1 / (np.pi/2)
            x[1] = t2 / (np.pi/2)
            x[2:] = 0.5  # Minimize g (assuming g = sum((x_i-0.5)^2))
            
            fx = eval_point(x)
            
            # Local search to minimize distance to front
            # Focus on variables that affect g (x[2:]) and fine-tune angles
            n_local_steps = 4
            step_size = 0.1
            
            for step in range(n_local_steps):
                if time.time() > deadline or used >= self.budget:
                    break
                
                # Generate multiple candidates
                best_x, best_fx = x, fx
                best_sum = np.sum(fx)
                
                n_candidates = 4
                for c in range(n_candidates):
                    if time.time() > deadline or used >= self.budget:
                        break
                    
                    x_new = x.copy()
                    # Perturb angle variables more carefully
                    if c % 2 == 0:
                        # Perturb angles
                        delta_t1 = step_size * rng.standard_normal()
                        delta_t2 = step_size * rng.standard_normal()
                        x_new[0] = np.clip(x[0] + delta_t1 / (np.pi/2), 0, 1)
                        x_new[1] = np.clip(x[1] + delta_t2 / (np.pi/2), 0, 1)
                    else:
                        # Perturb g variables
                        x_new[2:] = np.clip(x[2:] + step_size * rng.standard_normal(dim-2), 0, 1)
                    
                    fx_new = eval_point(x_new)
                    sum_new = np.sum(fx_new)
                    
                    if sum_new < best_sum:
                        best_x, best_fx, best_sum = x_new, fx_new, sum_new
                
                x, fx = best_x, best_fx
        
        # Phase 2: Fill remaining budget with random + local search
        if used < self.budget:
            n_random = min(20, self.budget - used)
            for _ in range(n_random):
                if time.time() > deadline or used >= self.budget:
                    break
                
                x = lo + (hi - lo) * rng.random(dim)
                fx = eval_point(x)
                
                # Quick local search
                for _ in range(3):
                    if time.time() > deadline or used >= self.budget:
                        break
                    x_new = np.clip(x + 0.05 * (hi - lo) * rng.standard_normal(dim), lo, hi)
                    fx_new = eval_point(x_new)
                    if np.sum(fx_new) < np.sum(fx):
                        x, fx = x_new, fx_new
        
        F = np.array(all_F)
        return F
# EVOLVE-BLOCK-END
