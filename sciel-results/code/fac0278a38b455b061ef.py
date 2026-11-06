"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Inverted photonic crystal: dielectric background with 4-fold symmetric air holes.
    Uses a coarse grid search over hole geometry parameters."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    xs = (np.arange(size) + 0.5) / size - 0.5
    
    def make_cell(r1, r2, d):
        """Dielectric background (1) with central air hole (r1) and 4 outer air holes (r2 at distance d)."""
        cell = np.ones((size, size), dtype=np.int8)
        # Central air hole
        dist_c = np.sqrt(xs[:, None]**2 + xs[None, :]**2)
        cell[dist_c < r1] = 0
        # 4 outer air holes
        if r2 > 0 and d > 0:
            for cx, cy in [(d, 0), (-d, 0), (0, d), (0, -d)]:
                dist_h = np.sqrt((xs[:, None] - cx)**2 + (xs[None, :] - cy)**2)
                cell[dist_h < r2] = 0
        return cell
    
    def enforce_symmetry(cell):
        """Ensure 4-fold symmetry by averaging with all symmetry operations."""
        c = cell.astype(np.float64)
        c_rot90 = np.rot90(c, 1)
        c_rot180 = np.rot90(c, 2)
        c_rot270 = np.rot90(c, 3)
        c_ref_h = c[::-1, :]
        c_ref_v = c[:, ::-1]
        c_ref_diag = c.T
        c_ref_anti = np.rot90(c, 1).T
        # Average all 8 symmetry operations
        c_avg = (c + c_rot90 + c_rot180 + c_rot270 + c_ref_h + c_ref_v + c_ref_diag + c_ref_anti) / 8.0
        # Threshold at 0.5
        return (c_avg > 0.5).astype(np.int8)
    
    def validate(cell):
        """Check if cell is valid 0/1 with 4-fold symmetry."""
        c = np.array(cell)
        if not set(np.unique(c)).issubset({0, 1}):
            return False
        if not np.array_equal(c, np.rot90(c, 1)):
            return False
        if not np.array_equal(c, np.rot90(c, 2)):
            return False
        if not np.array_equal(c, c.T):
            return False
        if not np.array_equal(c, c[::-1, :]):
            return False
        if not np.array_equal(c, c[:, ::-1]):
            return False
        return True
    
    # Known good parameter ranges for TM band gap in eps=12 square lattice
    # Central hole radius r1: 0.08-0.20
    # Outer hole radius r2: 0.05-0.15  
    # Distance d: 0.20-0.40
    best_cell = None
    best_air_frac = None
    
    # Coarse grid
    r1_vals = [0.10, 0.12, 0.14, 0.16, 0.18]
    r2_vals = [0.06, 0.08, 0.10, 0.12, 0.14]
    d_vals = [0.22, 0.26, 0.30, 0.34, 0.38]
    
    for r1 in r1_vals:
        for r2 in r2_vals:
            for d in d_vals:
                if time.time() > deadline:
                    break
                cell = make_cell(r1, r2, d)
                cell = enforce_symmetry(cell)
                if validate(cell):
                    air_frac = np.sum(cell == 0) / (size * size)
                    # Prefer air fraction in sweet spot for band gap
                    if best_air_frac is None or (0.30 < air_frac < 0.50 and abs(air_frac - 0.40) < abs(best_air_frac - 0.40)):
                        best_air_frac = air_frac
                        best_cell = cell.tolist()
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if best_cell is None:
        # Fallback: single central air hole
        cell = np.ones((size, size), dtype=np.int8)
        dist_c = np.sqrt(xs[:, None]**2 + xs[None, :]**2)
        cell[dist_c < 0.15] = 0
        best_cell = enforce_symmetry(cell).tolist()
    
    return best_cell
# EVOLVE-BLOCK-END
