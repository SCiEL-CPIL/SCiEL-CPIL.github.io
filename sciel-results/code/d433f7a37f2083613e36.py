import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, dim, n, offset=0):
        """Generate Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                base = d + 2  # prime bases: 2, 3, 5, 7, 11
                f = 1.0
                result = 0.0
                i_val = i + 1 + offset
                while i_val > 0:
                    f /= base
                    result += f * (i_val % base)
                    i_val //= base
                points[i, d] = result
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin (330s limit, leave 50s buffer)

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        eval_count = 0
        eval_history = []  # (value, x) for tracking top-k

        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_c = np.clip(np.asarray(x, dtype=float), -5, 5)
            try:
                val = float(func(x_c))
            except Exception:
                return np.inf
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x_c.copy()
            eval_history.append((val, x_c.copy()))
            return val

        def still_ok():
            return eval_count < budget and time.time() < deadline

        def grid_refine(lo, hi, n_per_dim, func_eval):
            """Evaluate a grid over [lo, hi]^dim with n_per_dim points per dimension.
            Returns the best point found."""
            if n_per_dim < 2:
                return func_eval((lo + hi) / 2.0)
            
            # Generate grid points
            grid_vals = np.linspace(lo, hi, n_per_dim)
            best_local = np.inf
            best_local_x = None
            
            # Iterate over all grid combinations
            for idx in range(n_per_dim ** dim):
                if not still_ok():
                    break
                # Convert index to multi-dimensional coordinates
                coords = []
                remaining = idx
                for d in range(dim - 1, -1, -1):
                    coords.append(remaining % n_per_dim)
                    remaining //= n_per_dim
                coords.reverse()
                
                x = np.array([grid_vals[c] for c in coords])
                val = func_eval(x)
                if val < best_local:
                    best_local = val
                    best_local_x = x.copy()
            
            return best_local_x if best_local_x is not None else (lo + hi) / 2.0

        # Phase 1: Coarse grid exploration - 3^5 = 243 evaluations
        # This gives good coverage of the domain
        if still_ok():
            n_coarse = 3
            lo = -5.0
            hi = 5.0
            grid_vals = np.linspace(lo, hi, n_coarse)
            
            for idx in range(n_coarse ** dim):
                if not still_ok():
                    break
                coords = []
                remaining = idx
                for d in range(dim - 1, -1, -1):
                    coords.append(remaining % n_coarse)
                    remaining //= n_coarse
                coords.reverse()
                
                x = np.array([grid_vals[c] for c in coords])
                counting_func(x)
            
            # Also evaluate center and a few structured points
            for x_start in [np.zeros(dim), np.ones(dim) * 2.0, -np.ones(dim) * 2.0]:
                if not still_ok():
                    break
                counting_func(x_start)
        
        if best_x is None:
            best_x = np.zeros(dim)
            counting_func(best_x)
        
        # Phase 2: Hierarchical grid refinement
        # Refine the best sub-box found in Phase 1
        remaining = budget - eval_count
        if remaining > 10 and still_ok() and best_x is not None:
            # Determine the best sub-box from the coarse grid
            # The coarse grid had 3 points per dim, so each cell is 10 units wide
            # Find which cell contains best_x
            n_coarse = 3
            cell_size = 10.0 / n_coarse  # ~3.33 units per cell
            
            # Get the cell containing best_x
            cell_idx = np.clip(((best_x + 5.0) / cell_size).astype(int), 0, n_coarse - 1)
            cell_lo = cell_idx * cell_size - 5.0
            cell_hi = (cell_idx + 1) * cell_size - 5.0
            
            # Ensure we stay within bounds
            cell_lo = np.maximum(cell_lo, -5.0)
            cell_hi = np.minimum(cell_hi, 5.0)
            
            # Refine with progressively finer grids
            refinement_grids = [5, 7, 9]
            for n_per_dim in refinement_grids:
                remaining = budget - eval_count
                if remaining < n_per_dim ** dim * 0.5:
                    break
                if not still_ok():
                    break
                
                # Evaluate grid in the current best sub-box
                grid_vals = np.linspace(cell_lo, cell_hi, n_per_dim)
                best_local = np.inf
                best_local_x = None
                
                for idx in range(n_per_dim ** dim):
                    if not still_ok():
                        break
                    coords = []
                    rem = idx
                    for d in range(dim - 1, -1, -1):
                        coords.append(rem % n_per_dim)
                        rem //= n_per_dim
                    coords.reverse()
                    
                    x = np.array([grid_vals[c] for c in coords])
                    val = counting_func(x)
                    if val < best_local:
                        best_local = val
                        best_local_x = x.copy()
                
                if best_local_x is not None:
                    # Update the sub-box to center around the best point
                    new_cell_size = (cell_hi - cell_lo) / n_per_dim
                    cell_lo = np.maximum(best_local_x - new_cell_size, -5.0)
                    cell_hi = np.minimum(best_local_x + new_cell_size, 5.0)
        
        # Phase 3: Local refinement using Nelder-Mead from best point
        remaining = budget - eval_count
        if remaining > 5 and still_ok() and best_x is not None:
            try:
                res = scipy.optimize.minimize(
                    counting_func,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(3, min(remaining, 50)),
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
            except Exception:
                pass
        
        # Phase 4: Final high-precision polish
        remaining = budget - eval_count
        if remaining > 3 and still_ok() and best_x is not None:
            try:
                res = scipy.optimize.minimize(
                    counting_func,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(2, remaining),
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
            except Exception:
                best_f = 0.0

        return best_x, best_f
