# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic adaptive grid refinement for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count += 1
            return x, f

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def is_non_dominated(f, archive_F):
            """Check if f is non-dominated with respect to archive_F."""
            return not is_dominated(f, archive_F)

        def update_archive(f):
            """Add f to archive if non-dominated, remove dominated points."""
            if not all_F:
                return True
            if is_dominated(f, all_F):
                return False
            # Remove points dominated by f
            new_F = []
            for af in all_F:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_F.append(af)
            all_F.extend(new_F)
            all_F.append(f)
            return True

        # Phase 1: Coarse grid initialization
        # Start with a 2^dim grid
        grid_res = [2] * dim
        grid_cells = []
        for i in range(2 ** dim):
            cell = []
            idx = i
            for d in range(dim):
                cell.append(idx % 2)
                idx //= 2
            grid_cells.append(cell)

        # Evaluate centroids of coarse grid
        for cell in grid_cells:
            if eval_count >= budget or time.time() > deadline:
                break
            centroid = np.array([(lo[d] + hi[d]) * (cell[d] + 0.5) / 2 for d in range(dim)])
            evaluate(centroid)

        # Phase 2: Adaptive refinement
        # Maintain a list of cells to refine, prioritized by potential
        # We'll use a simple strategy: refine cells that contain non-dominated points
        
        # Build a mapping from evaluated points to their grid cells
        # For simplicity, we'll track cells by their corner coordinates
        
        # Start with all coarse cells as candidates for refinement
        # Each cell is represented by its lower corner in normalized [0,1]^dim space
        cells_to_refine = []
        for cell in grid_cells:
            cells_to_refine.append(np.array(cell, dtype=float))

        # Shuffle to add some diversity in refinement order
        np.random.seed(42)
        np.random.shuffle(cells_to_refine)

        # Adaptive refinement loop
        while eval_count < budget and time.time() < deadline:
            if not cells_to_refine:
                break
            
            # Select the next cell to refine
            # Prioritize cells that are near non-dominated points
            # For simplicity, we'll just take the next cell in the list
            cell = cells_to_refine.pop(0)
            
            # Compute the cell boundaries in original space
            # cell[d] is the lower corner index in normalized space
            # The cell size is 1 / (2^level) where level depends on refinement depth
            # For now, we use a fixed refinement: split into 2^dim subcells
            
            subcells = []
            for i in range(2 ** dim):
                subcell = []
                idx = i
                for d in range(dim):
                    subcell.append(idx % 2)
                    idx //= 2
                # Transform to original space
                # The parent cell spans from cell[d] to cell[d]+1 in normalized space
                # But we need to track the actual level...
                # Let's use a different representation: store the actual boundaries
                
                subcell_bounds = []
                for d in range(dim):
                    # Parent cell lower bound in original space
                    lo_d = lo[d] + (hi[d] - lo[d]) * cell[d] / 2
                    hi_d = lo[d] + (hi[d] - lo[d]) * (cell[d] + 1) / 2
                    # Subcell divides this into 2
                    sub_lo = lo_d + (hi_d - lo_d) * subcell[d] / 2
                    sub_hi = lo_d + (hi_d - lo_d) * (subcell[d] + 1) / 2
                    subcell_bounds.append((sub_lo, sub_hi))
                
                # Centroid of subcell
                centroid = np.array([(b[0] + b[1]) / 2 for b in subcell_bounds])
                
                # Check if this centroid is already evaluated (avoid duplicates)
                # For simplicity, we'll just evaluate it
                if eval_count < budget and time.time() < deadline:
                    evaluate(centroid)
                    
                    # Add subcells to refinement list
                    for d in range(dim):
                        new_cell = cell.copy()
                        new_cell[d] = cell[d] * 2 + subcell[d]
                        subcells.append(new_cell)
            
            # Add subcells to the refinement queue
            cells_to_refine.extend(subcells)
            
            # Shuffle to maintain diversity
            np.random.shuffle(cells_to_refine)

        # Phase 3: Nelder-Mead polish on top non-dominated points
        if eval_count < budget and time.time() < deadline:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            # Find non-dominated
            mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                for j in range(len(F_arr)):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            nd_idx = np.where(mask)[0]

            # Pick top points by sum of objectives
            if len(nd_idx) > 0:
                order = np.argsort(np.sum(F_arr[nd_idx], axis=1))
                top_n = min(5, len(nd_idx), (budget - eval_count) // 2)

                for idx in order[:top_n]:
                    if eval_count >= budget or time.time() > deadline:
                        break

                    x0 = X_arr[idx].copy()

                    def objective(x):
                        x = np.clip(x, lo, hi)
                        f = np.array(problem(x))
                        return np.sum(f)

                    try:
                        res = minimize(objective, x0, method='Nelder-Mead',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                        if eval_count < budget:
                            cx = np.clip(res.x, lo, hi)
                            cf = np.array(problem(cx))
                            all_X.append(cx.copy())
                            all_F.append(cf.copy())
                            eval_count += 1
                    except Exception:
                        pass

        return all_X
# EVOLVE-BLOCK-END
