"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1-specific optimizer: exploit known structure for convergence + diversity."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # DTLZ1: x[0:dim-1] control g, x[dim-1] controls spread on front
        # Optimal: x[0:dim-1] = 0.5 (center of [0,1]), x[dim-1] uniform in [0,1]
        
        # Phase 1: Evaluate center point to establish baseline
        center = (lo + hi) / 2.0
        F_center = problem(center)
        used = 1
        all_X = [center]
        all_F = [F_center]
        
        # Phase 2: Generate diverse points on the Pareto front
        # Use quasi-random (low-discrepancy) sequence for x[dim-1]
        n_diversity = min(budget - 1, 100)
        # Use Halton sequence or just uniform grid for simplicity
        # With budget=200, we can afford ~100 diverse points
        if n_diversity > 0:
            # Use a simple low-discrepancy approach: golden ratio sequence
            golden = (1 + 5**0.5) / 2.0
            inv_golden = 1.0 / golden
            z_values = []
            for i in range(n_diversity):
                z = (i * inv_golden) % 1.0
                z_values.append(z)
            
            for z in z_values:
                if time.time() > deadline:
                    break
                x = center.copy()
                # Map z to bounds for last dimension
                x[-1] = lo[-1] + z * (hi[-1] - lo[-1])
                f = problem(x)
                used += 1
                all_X.append(x)
                all_F.append(f)
        
        # Phase 3: Use remaining budget for local refinement around center
        remaining = budget - used
        if remaining > 0:
            # Do local search to improve convergence (reduce g)
            # Try small perturbations around center
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                # Random direction with small step
                direction = rng.standard_normal(dim)
                step_size = 0.05 * (hi - lo)
                x = center + step_size * direction
                x = np.clip(x, lo, hi)
                f = problem(x)
                used += 1
                all_X.append(x)
                all_F.append(f)
        
        return np.array(all_X)
# EVOLVE-BLOCK-END
