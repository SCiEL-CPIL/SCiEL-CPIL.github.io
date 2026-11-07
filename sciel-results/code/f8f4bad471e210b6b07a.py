"""Deterministic hierarchical coordinate-wise descent for Katsuura-like f23."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic hierarchical coordinate-wise descent.
    
    Phase 1: Coarse coordinate-wise grid scans to locate the basin.
    Phase 2: Finer coordinate-wise grid scans for refinement.
    Phase 3: Local full-grid search for precise minimization.
    Phase 4: Final coordinate-wise polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            calls += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def coordinate_grid_scan(x, offsets):
            """
            Perform a coordinate-wise scan using a fixed set of offsets.
            Updates x in-place to the best found coordinates.
            """
            for d in range(dim):
                best_val = x[d]
                best_f_d = float('inf')
                for off in offsets:
                    candidate = x.copy()
                    candidate[d] = x[d] + off
                    candidate[d] = np.clip(candidate[d], -5.0, 5.0)
                    if calls >= budget:
                        break
                    fx = evaluate(candidate)
                    if fx < best_f_d:
                        best_f_d = fx
                        best_val = candidate[d]
                x[d] = best_val
            return x

        def local_grid_search(x, step):
            """
            Perform a full grid search in a 3x3x...x3 box around x.
            """
            best_x_local = x.copy()
            best_f_local = float('inf')
            
            # Generate 3 points per dimension: x - step, x, x + step
            points_per_dim = [x[i] - step, x[i], x[i] + step]
            
            # Iterate through all combinations
            # Number of points is 3^dim. For dim=5, 3^5=243.
            # We use itertools.product for clarity, or nested loops.
            # Since dim is small (5), we can use a simple recursion or product.
            import itertools
            
            for combo in itertools.product(*[range(3) for _ in range(dim)]):
                if calls >= budget:
                    break
                candidate = np.zeros(dim)
                for i in range(dim):
                    candidate[i] = points_per_dim[i][combo[i]]
                    candidate[i] = np.clip(candidate[i], -5.0, 5.0)
                fx = evaluate(candidate)
                if fx < best_f_local:
                    best_f_local = fx
                    best_x_local = candidate.copy()
            
            return best_x_local

        # Phase 1: Coarse Coordinate-wise Grid Scan
        # Start at origin. Try large offsets to cover the domain [-5, 5].
        # Offsets are relative to current x[d].
        # Since we start at 0, this effectively tests absolute positions.
        coarse_offsets = [-4.0, -2.0, 0.0, 2.0, 4.0]
        best_x = coordinate_grid_scan(best_x, coarse_offsets)

        # Phase 2: Mid Coordinate-wise Grid Scan
        mid_offsets = [-1.0, -0.5, 0.0, 0.5, 1.0]
        best_x = coordinate_grid_scan(best_x, mid_offsets)

        # Phase 3: Fine Coordinate-wise Grid Scan
        fine_offsets = [-0.2, -0.1, 0.0, 0.1, 0.2]
        best_x = coordinate_grid_scan(best_x, fine_offsets)

        # Phase 4: Very Fine Coordinate-wise Grid Scan
        vfine_offsets = [-0.05, -0.02, 0.0, 0.02, 0.05]
        best_x = coordinate_grid_scan(best_x, vfine_offsets)

        # Phase 5: Local Full Grid Search
        # We have used 4 * 5 * 5 = 100 calls.
        # Budget remaining: 400.
        # Grid search uses 3^5 = 243 calls.
        # Step size 0.01.
        best_x = local_grid_search(best_x, 0.01)

        # Phase 6: Final Polish with Coordinate-wise Descent
        # Calls used: 100 + 243 = 343.
        # Remaining: 157.
        # Use small offsets for coordinate-wise scan.
        # 3 points per dim: 5 * 3 = 15 calls per pass.
        # We can do multiple passes.
        
        polish_offsets = [-0.005, 0.0, 0.005]
        while calls < budget:
            # Check if we can afford a full pass
            cost = dim * len(polish_offsets)
            if calls + cost > budget:
                break
            best_x = coordinate_grid_scan(best_x, polish_offsets)
            
            # Shrink step for next polish
            polish_offsets = [o * 0.5 for o in polish_offsets]
            if abs(polish_offsets[0]) < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
