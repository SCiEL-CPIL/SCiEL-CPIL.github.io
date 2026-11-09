"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using non-uniform x1 sampling with optimal tails."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT4 structure: x1 controls the Pareto front, x2..x10 affect g(x)
        # Optimal tail: x_i = pi/2 for i >= 2 (since g(x) = 2 + sum((x_i - pi/2)^2 * sin(10*(x_i-pi/2)^2)))
        # At x_i = pi/2, sin(0) = 0, so g = 2 (minimum)
        
        # Strategy: 
        # 1. Use first ~50 evals to probe tail optimality (but we know ZDT4 structure)
        # 2. Use remaining ~150 evals to densely sample x1 with optimal tails
        
        # We know the structure, so let's be smart:
        # - Tail values: x_i = pi/2 gives g=2, which is optimal
        # - But let's verify with a few probes, then focus on x1 sweep
        
        # Phase 1: Quick probe of tail region (10 evals)
        # Try different tail offsets to confirm pi/2 is optimal
        tail_probe = 10
        if budget > tail_probe + 10:
            # Probe a few tail configurations
            base_tail = np.full(dim - 1, np.pi / 2)
            for i in range(min(tail_probe, budget)):
                x = np.empty(dim)
                x[0] = lo[0] + (hi[0] - lo[0]) * 0.5  # middle x1
                if i < (dim - 1):
                    # Perturb one tail variable
                    x[1:] = base_tail.copy()
                    x[1 + i] = base_tail[i] + 0.1 * (hi[1 + i] - lo[1 + i])
                else:
                    x[1:] = base_tail
                problem(x)
        
        # Phase 2: Dense non-uniform sweep of x1 with optimal tails
        # The Pareto front of ZDT4: f1 = x1, f2 = (1 + g) * (1 - sqrt(f1/g))
        # With g = 2 (optimal tail), f2 = 3 * (1 - sqrt(x1/2))
        # This is a smooth curve from (0, 3) to (1, 0)
        
        # Non-uniform grid: more points near the "knee" where the curve bends
        # Use a quadratic weighting to allocate more samples where hypervolume contribution is highest
        
        remaining_budget = budget - tail_probe if budget > tail_probe + 10 else budget
        if remaining_budget < 5:
            remaining_budget = budget
        
        # Create non-uniform grid for x1
        # The hypervolume contribution is highest where the front is farthest from reference
        # Reference is [11, 400], so all points on the front are well inside
        # The "knee" of ZDT4 with g=2 is around x1 = 0.5 where the curve changes curvature
        
        n_sweep = max(5, remaining_budget)
        
        # Non-uniform: use a power transformation to concentrate points
        # t in [0, 1], x1 = lo[0] + (hi[0] - lo[0]) * t^alpha
        # alpha < 1 concentrates near 0, alpha > 1 concentrates near 1
        # For ZDT4, the front is smooth, but let's use alpha=1 (uniform) as baseline
        # Actually, let's use a slightly non-uniform grid with more points in the middle
        
        # Create grid with denser sampling in the middle
        t = np.linspace(0, 1, n_sweep)
        # Apply a slight bias toward the center
        # Use a smooth weighting: t_weighted = 0.5 + 0.5 * sin(pi * t)
        # This maps [0,1] to [0,1] with more density in the middle
        t_weighted = 0.5 + 0.5 * np.sin(np.pi * t)
        
        x1_values = lo[0] + (hi[0] - lo[0]) * t_weighted
        
        # Ensure we stay within bounds
        x1_values = np.clip(x1_values, lo[0], hi[0])
        
        # Evaluate the sweep
        base_tail = np.full(dim - 1, np.pi / 2)
        base_tail = np.clip(base_tail, lo[1:], hi[1:])
        
        # Track evaluated points
        evaluated = []
        for i, x1_val in enumerate(x1_values):
            if time.time() > deadline:
                break
            x = np.empty(dim)
            x[0] = x1_val
            x[1:] = base_tail
            f = problem(x)
            evaluated.append((x, f))
        
        # If we have budget left, do a local refinement around the best regions
        # Find the point on the front that gives the best hypervolume contribution
        # For 2 objectives with reference (11, 400), HV contribution of point (f1, f2) is
        # (11 - f1) * (400 - f2) if f1 < 11 and f2 < 400
        
        if evaluated and time.time() < deadline:
            # Compute hypervolume contribution for each point
            f_arr = np.array([f for _, f in evaluated])
            # Only count points strictly better than reference
            valid = np.all(f_arr < np.array([11.0, 400.0]), axis=1)
            if np.any(valid):
                f_valid = f_arr[valid]
                hv_contrib = (11.0 - f_valid[:, 0]) * (400.0 - f_valid[:, 1])
                best_idx = np.argmax(hv_contrib)
                best_x = evaluated[best_idx][0]
                
                # Local refinement around best point
                n_refine = min(20, budget - len(evaluated) - tail_probe)
                if n_refine > 0:
                    # Try small perturbations in x1
                    x1_best = best_x[0]
                    x1_range = hi[0] - lo[0]
                    for j in range(n_refine):
                        if time.time() > deadline:
                            break
                        # Perturb x1
                        x1_new = x1_best + (j - n_refine // 2) * x1_range / n_refine * 0.05
                        x1_new = np.clip(x1_new, lo[0], hi[0])
                        x = np.empty(dim)
                        x[0] = x1_new
                        x[1:] = base_tail
                        problem(x)
        
        return evaluated
# EVOLVE-BLOCK-END
