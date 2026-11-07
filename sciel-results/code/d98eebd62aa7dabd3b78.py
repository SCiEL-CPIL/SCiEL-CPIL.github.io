"""
Optimization strategy:
1. Latin Hypercube Sampling (LHS) for diverse initial exploration.
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

        # Grid-based coarse-to-fine deterministic scan
        # Phase 1: Coarse grid over [-5,5]^dim
        # Phase 2: Iteratively refine the best cells
        # Phase 3: Final ES polish

        # Coarse grid: 2 points per dimension -> 2^dim cells
        n_coarse = 2
        cell_size = 10.0 / n_coarse  # 5.0
        # Generate grid centers
        grid_coords = np.linspace(-5.0 + cell_size / 2, 5.0 - cell_size / 2, n_coarse)
        
        # Create all grid points
        mesh = np.array(np.meshgrid(*[grid_coords] * dim)).T.reshape(-1, dim)
        
        # Evaluate coarse grid
        coarse_results = []
        for pt in mesh:
            if check_deadline():
                break
            fx = evaluate(pt)
            coarse_results.append((pt, fx))
        
        if not coarse_results:
            # Fallback: random point
            x0 = rng.uniform(-5, 5, dim)
            evaluate(x0)
            return best_x, best_f

        # Sort coarse results by fitness
        coarse_results.sort(key=lambda t: t[1])
        
        # Phase 2: Iterative refinement
        # Keep a list of "active cells" to refine
        # Each cell is defined by its center and half-size
        # Start with the best 2 coarse cells
        active_cells = []
        for pt, fx in coarse_results[:2]:
            active_cells.append((pt.copy(), cell_size / 2.0))  # (center, half_size)
        
        max_refine_levels = 6
        for level in range(max_refine_levels):
            if check_deadline():
                break
            
            new_active = []
            for center, half_size in active_cells:
                if check_deadline():
                    break
                
                # Subdivide this cell into 2^dim sub-cells
                sub_cell_size = half_size / 2.0
                sub_centers = []
                for offset in np.ndindex([2] * dim):
                    # offset is a tuple of 0s and 1s
                    sub_center = center.copy()
                    for d in range(dim):
                        sub_center[d] += (offset[d] - 0.5) * sub_cell_size
                    # Clip to box
                    sub_center = np.clip(sub_center, -5.0, 5.0)
                    sub_centers.append(sub_center)
                
                # Evaluate sub-cells
                sub_results = []
                for sc in sub_centers:
                    if check_deadline():
                        break
                    fx = evaluate(sc)
                    sub_results.append((sc, fx))
                
                if sub_results:
                    sub_results.sort(key=lambda t: t[1])
                    # Keep the best 1-2 sub-cells
                    for pt, fx in sub_results[:2]:
                        new_active.append((pt.copy(), sub_cell_size))
            
            if not new_active:
                break
            active_cells = new_active
            
            # If we've used most of our budget, stop refining
            if calls_made > budget * 0.7:
                break
        
        # Phase 3: Final ES polish around the global best
        if not check_deadline() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            success_count = 0
            total_steps = 0
            no_improve_count = 0
            
            while not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                total_steps += 1
                
                if fy < fx:
                    improvement = fx - fy
                    x, fx = y, fy
                    success_count += 1
                    no_improve_count = 0
                    if improvement < 1e-10:
                        sigma *= 0.7
                else:
                    no_improve_count += 1
                
                # Adapt sigma
                if total_steps % 4 == 0:
                    success_rate = success_count / 4.0
                    if success_rate > 0.2:
                        sigma *= 1.15
                    else:
                        sigma *= 0.85
                    success_count = 0
                    
                sigma = np.clip(sigma, 1e-10, 0.5)
                
                # Stop if sigma is very small and no recent improvement
                if sigma < 1e-8 and no_improve_count > 20:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
