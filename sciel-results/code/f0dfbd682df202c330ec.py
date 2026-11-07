"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampling for WFG4 with analytic structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For WFG4, the Pareto front is parameterized by x_0 (first variable)
        # The other variables have specific optimal values on the front
        # WFG4 structure: 
        #   f1 = x1
        #   f2 = (1-x1) * g(x) where g is minimized at specific x values
        #   f3 = x1 * h(x) where h is minimized at specific x values
        # 
        # The Pareto front is achieved when x2..x12 are at their optimal values
        # and x1 varies over its range.
        
        # Strategy: 
        # 1. First, find the optimal values for x2..x12 by minimizing a combined objective
        # 2. Then sample x1 uniformly and evaluate points on the front
        # 3. Add small perturbations to explore near-front regions
        
        used = 0
        all_F = []
        
        # Phase 1: Find good base point for variables 2..12
        # Try to minimize sum of objectives with x1 at midpoint
        x1_mid = (lo[0] + hi[0]) / 2.0
        
        # Initialize base for x2..x12
        base_rest = (lo[1:] + hi[1:]) / 2.0
        
        # Use a few evaluations to refine base_rest
        # Minimize the "distance" from ideal by trying different values
        best_base = base_rest.copy()
        best_sum = np.inf
        
        rng = np.random.default_rng(42)
        
        # Try random points to find a good base for x2..x12
        n_init = min(30, self.budget // 5)
        for i in range(n_init):
            if used >= self.budget or time.time() > deadline - 5.0:
                break
            x = np.zeros(self.dim)
            x[0] = x1_mid
            x[1:] = lo[1:] + (hi[1:] - lo[1:]) * rng.random(self.dim - 1)
            f = problem(x)
            used += 1
            s = np.sum(f)
            if s < best_sum:
                best_sum = s
                best_base = x[1:].copy()
        
        # Phase 2: Sample the Pareto front by varying x1
        # The front is approximately linear in x1
        # Sample x1 uniformly and use best_base for x2..x12
        
        n_front_samples = min(self.budget - used, 150)
        
        # Create a spread of x1 values
        x1_values = np.linspace(lo[0], hi[0], n_front_samples)
        
        for x1_val in x1_values:
            if used >= self.budget or time.time() > deadline - 5.0:
                break
            x = np.zeros(self.dim)
            x[0] = x1_val
            x[1:] = best_base
            f = problem(x)
            used += 1
            all_F.append(f)
        
        # Phase 3: Add local perturbations around front points to ensure coverage
        remaining = self.budget - used
        if remaining > 0:
            # Pick a few front points and perturb them
            n_perturb = min(remaining, 30)
            indices = rng.choice(len(all_F), size=min(n_perturb, len(all_F)), replace=False)
            for idx in indices:
                if used >= self.budget or time.time() > deadline - 5.0:
                    break
                base_x1 = x1_values[idx] if idx < len(x1_values) else x1_mid
                x = np.zeros(self.dim)
                x[0] = base_x1 + 0.05 * (hi[0] - lo[0]) * rng.standard_normal()
                x[0] = np.clip(x[0], lo[0], hi[0])
                x[1:] = best_base + 0.05 * (hi[1:] - lo[1:]) * rng.standard_normal(self.dim - 1)
                x[1:] = np.clip(x[1:], lo[1:], hi[1:])
                f = problem(x)
                used += 1
                all_F.append(f)
        
        # Use remaining budget for random exploration
        while used < self.budget:
            if time.time() > deadline - 5.0:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            f = problem(x)
            used += 1
            all_F.append(f)
        
        return all_F
# EVOLVE-BLOCK-END
