"""Deterministic space-filling search with (1+1)-ES refinement for BBOB f20 (Schwefel)."""
import numpy as np
import time

class Optimizer:
    """Space-filling Sobol/LHS scan followed by (1+1)-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        call_count = 0

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(fx):
                return np.inf
            call_count += 1
            return fx

        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Deterministic space-filling scan using a large number of points
        # We use a combination of structured points and Sobol-like sequence
        # For 5D, we can use a grid + jitter or a proper Sobol sequence
        
        # Generate points using a simple deterministic space-filling approach
        # We'll use a stratified sampling approach (similar to LHS but more structured)
        
        n_scan = 100  # Number of scan points
        
        # Create stratified points: divide each dimension into n_scan^(1/dim) intervals
        # For dim=5, n_scan=100, intervals per dim = 100^(1/5) ≈ 2.5, so use 3 intervals
        # Actually, let's use a proper approach: generate points on a grid with jitter
        
        # Better approach: Use a Halton sequence or simple deterministic sampling
        # Let's use a stratified sampling with multiple levels
        
        # Generate points using a simple deterministic method:
        # For each of n_scan points, use a different deterministic rule
        
        scan_points = []
        
        # Add origin
        scan_points.append(np.zeros(dim))
        
        # Add axis-aligned points
        for axis in range(dim):
            for val in [-4.0, -2.0, 0.0, 2.0, 4.0]:
                pt = np.zeros(dim)
                pt[axis] = val
                scan_points.append(pt)
        
        # Add corner points
        for mask in range(2 ** dim):
            pt = np.zeros(dim)
            for i in range(dim):
                pt[i] = 4.0 if (mask & (1 << i)) else -4.0
            scan_points.append(pt)
        
        # Add more structured points using a stratified approach
        # Use a simple deterministic space-filling: divide space into cells
        n_cells_per_dim = 3  # 3^5 = 243 cells, but we only need ~100 points
        # Actually, let's use a different approach: generate points using a fixed seed
        # but in a stratified manner
        
        rng = np.random.default_rng(42)
        
        # Generate stratified points: for each dimension, divide into n_scan^(1/dim) intervals
        # and place one point in each cell
        n_intervals = int(np.ceil(n_scan ** (1.0 / dim)))
        
        # Generate all combinations of interval indices
        interval_indices = np.array(np.meshgrid(*[np.arange(n_intervals)] * dim)).T.reshape(-1, dim)
        
        # Shuffle to get a random order
        perm = rng.permutation(len(interval_indices))
        interval_indices = interval_indices[perm]
        
        # Take the first n_scan - len(scan_points) points
        n_needed = max(0, n_scan - len(scan_points))
        scan_points_to_add = []
        
        for idx in range(min(n_needed, len(interval_indices))):
            cell_idx = interval_indices[idx]
            # Place point in the cell with random jitter
            pt = np.zeros(dim)
            for j in range(dim):
                low = -5.0 + (cell_idx[j] / n_intervals) * 10.0
                high = -5.0 + ((cell_idx[j] + 1) / n_intervals) * 10.0
                pt[j] = low + rng.uniform(0.0, high - low)
            scan_points_to_add.append(pt)
        
        scan_points.extend(scan_points_to_add)
        
        # Evaluate scan points
        for pt in scan_points:
            if call_count >= budget or time.time() > deadline:
                break
            f = safe_call(pt)
            update_best(pt, f)
        
        # Phase 2: (1+1)-ES refinement from the best point
        if time.time() < deadline and budget - call_count > 10 and best_f < np.inf:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Initial step size
            
            remaining = budget - call_count
            for _ in range(remaining):
                if time.time() > deadline or call_count >= budget:
                    break
                
                # Generate a single mutant
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                
                fy = safe_call(y)
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    # Slightly increase sigma when we make progress
                    sigma = min(sigma * 1.05, 5.0)
                else:
                    # Decrease sigma when we don't make progress
                    sigma = max(sigma * 0.9, 1e-8)
                
                if sigma < 1e-10:
                    break
        
        return best_x, best_f
