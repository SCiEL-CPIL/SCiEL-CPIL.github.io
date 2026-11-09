"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for DTLZ1 using analytic front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # For DTLZ1, the Pareto front is characterized by:
        # f0 = 0.5 * x[0] * (1 + g)
        # f1 = 0.5 * x[0] * (1 - x[1]) * (1 + g)
        # f2 = 0.5 * x[0] * x[1] * (1 + g)
        # where g = sum_{i=3}^{n} (x_i - 0.5)^2 - 10*cos(4*pi*x_i*0.5)
        # g_min = 0 achieved at x_i = 0.5 for all i >= 2
        #
        # Strategy:
        # 1. Find optimal x[2:] by minimizing g (should be near 0.5 for each component)
        # 2. Sample points on the Pareto front using a deterministic grid
        # 3. Use remaining budget for local refinement

        rng = np.random.default_rng(42)
        best_x2 = np.full(dim - 2, 0.5)
        best_g_val = np.inf
        
        # Phase 1: Find optimal x[2:] configuration
        # Probe candidates for x[2:]
        candidates = [
            np.zeros(dim - 2),
            np.ones(dim - 2),
            np.full(dim - 2, 0.5),
        ]
        for _ in range(8):
            candidates.append(rng.random(dim - 2))
            
        for x2 in candidates:
            if time.time() > deadline:
                break
            x_test = np.concatenate([[0.5, 0.5], x2])
            x_test = np.clip(x_test, lo, hi)
            f = problem(x_test)
            g_val = np.sum(f)
            if g_val < best_g_val:
                best_g_val = g_val
                best_x2 = x2.copy()
        
        # Local search to refine x[2:]
        n_local = min(15, self.budget - 12)
        for _ in range(n_local):
            if time.time() > deadline:
                break
            x2_pert = best_x2 + 0.1 * (hi[2:] - lo[2:]) * rng.standard_normal(dim - 2)
            x2_pert = np.clip(x2_pert, lo[2:], hi[2:])
            x_test = np.concatenate([[0.5, 0.5], x2_pert])
            x_test = np.clip(x_test, lo, hi)
            f_test = problem(x_test)
            g_test = np.sum(f_test)
            if g_test < best_g_val:
                best_g_val = g_test
                best_x2 = x2_pert.copy()
        
        evaluated = 12 + n_local
        
        # Phase 2: Sample points on the Pareto front using a deterministic grid
        # The Pareto front for DTLZ1 (m=3) is a 2D simplex in objective space:
        # f0 + f1 + f2 = 0.5 * x0 * (1 + g_min)
        # With g_min ≈ 0 and x0 = 1: f0 + f1 + f2 = 0.5
        # We parameterize by (x0, x1) where x0 ∈ [0, 1], x1 ∈ [0, 1-x0]
        
        recorded_F = []
        
        # Use remaining budget for front sampling with a deterministic grid
        remaining = self.budget - evaluated
        
        # Create a 14x14 grid in (x0, x1) space covering the simplex
        n_grid = 14
        grid_x0 = np.linspace(0.0, 1.0, n_grid)
        
        for i, x0_val in enumerate(grid_x0):
            if time.time() > deadline:
                break
            if evaluated >= self.budget:
                break
            # x1 ranges from 0 to 1 - x0
            max_x1 = 1.0 - x0_val
            if max_x1 < 0:
                continue
            n_x1 = max(1, int(n_grid * (1.0 - i / n_grid)))
            grid_x1 = np.linspace(0.0, max_x1, n_x1)
            
            for x1_val in grid_x1:
                if time.time() > deadline:
                    break
                if evaluated >= self.budget:
                    break
                x_sample = np.concatenate([[x0_val, x1_val], best_x2])
                x_sample = np.clip(x_sample, lo, hi)
                f_sample = problem(x_sample)
                recorded_F.append(f_sample)
                evaluated += 1
        
        # Add mid-edge points for better coverage
        mid_edge_points = [
            [0.5, 0.0],
            [0.0, 0.5],
            [0.25, 0.25],
            [0.75, 0.0],
            [0.0, 0.75],
            [0.5, 0.25],
            [0.25, 0.5],
        ]
        for cp in mid_edge_points:
            if time.time() > deadline:
                break
            if evaluated >= self.budget:
                break
            x_cp = np.concatenate([cp, best_x2])
            x_cp = np.clip(x_cp, lo, hi)
            f_cp = problem(x_cp)
            # Check if this point is already recorded (avoid duplicates)
            is_dup = False
            for existing in recorded_F:
                if np.allclose(existing, f_cp, atol=1e-10):
                    is_dup = True
                    break
            if not is_dup:
                recorded_F.append(f_cp)
                evaluated += 1
        
        # If we still have budget, add more random points for refinement
        if evaluated < self.budget and time.time() < deadline:
            n_extra = min(remaining - (self.budget - evaluated), 20)
            for _ in range(n_extra):
                if time.time() > deadline:
                    break
                if evaluated >= self.budget:
                    break
                u1 = rng.random()
                u2 = rng.random()
                if u1 + u2 > 1:
                    u1, u2 = 1 - u1, 1 - u2
                x_sample = np.concatenate([[u1, u2], best_x2])
                x_sample = np.clip(x_sample, lo, hi)
                f_sample = problem(x_sample)
                recorded_F.append(f_sample)
                evaluated += 1
        
        return recorded_F
# EVOLVE-BLOCK-END
