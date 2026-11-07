"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-resolution adaptive grid search for step-ellipsoidal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Count function calls
        call_count = [0]

        def counted(x):
            call_count[0] += 1
            return func(x)

        # Phase 1: Coarse grid to find promising region
        # Use a 3^dim grid (27 points in 5D) for initial exploration
        n_coarse = 3
        grid_points = []
        for i in range(n_coarse ** dim):
            coords = []
            val = i
            for d in range(dim):
                coords.append(val % n_coarse)
                val //= n_coarse
            x = np.array([(2 * c + 1) / n_coarse * 10 - 5 for c in coords])
            grid_points.append(x)

        # Evaluate coarse grid
        best_x = None
        best_f = np.inf
        for x in grid_points:
            if call_count[0] >= budget:
                break
            fx = counted(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Adaptive octree bisection
        # Maintain a priority queue of regions to explore
        # Each region is defined by its center and half-size
        # We bisect the region with the most "promise" (largest function variation)

        # Collect evaluated points for interpolation
        eval_points = []
        eval_values = []

        # Re-evaluate coarse grid and store
        for x in grid_points:
            if call_count[0] >= budget:
                break
            fx = counted(x)
            eval_points.append(x.copy())
            eval_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Define regions to explore (start with the cell containing best point)
        # Each region: (center, half_size, priority)
        # Priority based on potential for improvement

        # Find the coarse cell containing best_x
        cell_size = 10.0 / n_coarse
        cell_idx = []
        for d in range(dim):
            idx = min(n_coarse - 1, max(0, int((best_x[d] + 5) / cell_size)))
            cell_idx.append(idx)

        # Create initial region around best point
        center = np.array([(2 * idx + 1) / n_coarse * 10 - 5 for idx in cell_idx])
        half_size = cell_size / 2.0

        # Also add neighboring cells for exploration
        regions = [(center, half_size)]

        # Add some neighboring regions
        for d in range(dim):
            for offset in [-1, 1]:
                new_idx = cell_idx[:]
                new_idx[d] = cell_idx[d] + offset
                if 0 <= new_idx[d] < n_coarse:
                    nc = np.array([(2 * idx + 1) / n_coarse * 10 - 5 for idx in new_idx])
                    regions.append((nc, half_size))

        # Adaptive search: repeatedly bisect the most promising region
        # We'll use a simple strategy: always bisect the region containing the current best
        # and explore nearby regions

        while call_count[0] < budget:
            if time.time() > deadline:
                break

            # Get current best region
            if not regions:
                # Fallback: random search
                if call_count[0] >= budget:
                    break
                x = np.clip(best_x + 0.1 * np.random.randn(dim), -5, 5)
                fx = counted(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                continue

            # Pick the region closest to current best
            best_region_idx = 0
            min_dist = np.inf
            for i, (center, hs) in enumerate(regions):
                dist = np.sum((center - best_x) ** 2) / (hs ** 2 + 1e-10)
                if dist < min_dist:
                    min_dist = dist
                    best_region_idx = i

            center, half_size = regions[best_region_idx]

            # Evaluate center if not already done
            center_tuple = tuple(np.round(center, 10))
            already_evaluated = False
            for ep in eval_points:
                if np.allclose(ep, center, atol=1e-10):
                    already_evaluated = True
                    break

            if not already_evaluated:
                if call_count[0] >= budget:
                    break
                fx_center = counted(center)
                eval_points.append(center.copy())
                eval_values.append(fx_center)
                if fx_center < best_f:
                    best_f = fx_center
                    best_x = center.copy()

            # Bisect: create 2^dim sub-regions
            new_regions = []
            for mask in range(2 ** dim):
                new_center = center.copy()
                for d in range(dim):
                    if mask & (1 << d):
                        new_center[d] += half_size / 2.0
                    else:
                        new_center[d] -= half_size / 2.0
                # Clip to bounds
                new_center = np.clip(new_center, -5, 5)
                new_half = half_size / 2.0
                if new_half > 1e-4:  # Don't go too fine
                    new_regions.append((new_center, new_half))

            # Replace the region with its sub-regions
            regions[best_region_idx:best_region_idx + 1] = new_regions

            # Also add some random exploration to avoid getting stuck
            if call_count[0] < budget and len(regions) < 100:
                x = np.clip(best_x + 0.5 * np.random.randn(dim), -5, 5)
                fx = counted(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                # Add a region around this point
                regions.append((x.copy(), 0.3))

            # Prune regions that are too far from best
            regions = [(c, h) for c, h in regions if np.sum((c - best_x) ** 2) < (h * 3) ** 2]

            if not regions:
                # Restart with a broader search
                regions = [(best_x.copy(), 1.0)]

        return best_x, best_f
# EVOLVE-BLOCK-END
