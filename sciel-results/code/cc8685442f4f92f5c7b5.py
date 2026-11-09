# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from itertools import combinations

TOL = 1e-7
DEADLINE_MARGIN = 10.0  # seconds before 630s limit

def _is_valid(centers, sides, n=10):
    """Check if a packing is valid."""
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Check inside [-tol, 1+tol]^2
    for i in range(n):
        x0 = centers[i, 0] - sides[i] / 2
        x1 = centers[i, 0] + sides[i] / 2
        y0 = centers[i, 1] - sides[i] / 2
        y1 = centers[i, 1] + sides[i] / 2
        if x0 < -TOL - 1e-10 or x1 > 1 + TOL + 1e-10:
            return False
        if y0 < -TOL - 1e-10 or y1 > 1 + TOL + 1e-10:
            return False
    # Check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _solve_lp_for_separations(n, sep_x, sep_y):
    """
    Solve LP to maximize sum of sides given fixed separation pattern.
    sep_x[i,j] = 1 if square i is to the left of j (x_i + s_i/2 <= x_j - s_j/2),
                = -1 if right, = 0 if not separated in x.
    sep_y similarly for y.
    """
    # Variables: cx_i, cy_i, s_i for i in 0..n-1
    # Total vars: 3n
    # Maximize sum(s_i) -> minimize -sum(s_i)
    
    c = np.zeros(3 * n)
    for i in range(n):
        c[2 * n + i] = -1.0  # maximize s_i
    
    A_ub = []
    b_ub = []
    
    # Inside unit square constraints:
    # cx_i - s_i/2 >= 0  =>  -cx_i + s_i/2 <= 0
    # cx_i + s_i/2 <= 1  =>   cx_i - s_i/2 <= 1
    # cy_i - s_i/2 >= 0  =>  -cy_i + s_i/2 <= 0
    # cy_i + s_i/2 <= 1  =>   cy_i - s_i/2 <= 1
    for i in range(n):
        # -cx_i + s_i/2 <= 0
        row = np.zeros(3 * n)
        row[i] = -1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # cx_i - s_i/2 <= 1
        row = np.zeros(3 * n)
        row[i] = 1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        # -cy_i + s_i/2 <= 0
        row = np.zeros(3 * n)
        row[n + i] = -1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # cy_i - s_i/2 <= 1
        row = np.zeros(3 * n)
        row[n + i] = 1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(1.0)
    
    # Separation constraints
    for i in range(n):
        for j in range(i + 1, n):
            if sep_x[i, j] == 1:
                # cx_i + s_i/2 <= cx_j - s_j/2
                # cx_i - cx_j + s_i/2 + s_j/2 <= 0
                row = np.zeros(3 * n)
                row[i] = 1.0
                row[j] = -1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
            elif sep_x[i, j] == -1:
                # cx_j + s_j/2 <= cx_i - s_i/2
                # -cx_i + cx_j + s_i/2 + s_j/2 <= 0
                row = np.zeros(3 * n)
                row[i] = -1.0
                row[j] = 1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
            
            if sep_y[i, j] == 1:
                # cy_i + s_i/2 <= cy_j - s_j/2
                # cy_i - cy_j + s_i/2 + s_j/2 <= 0
                row = np.zeros(3 * n)
                row[n + i] = 1.0
                row[n + j] = -1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
            elif sep_y[i, j] == -1:
                # cy_j + s_j/2 <= cy_i - s_i/2
                # -cy_i + cy_j + s_i/2 + s_j/2 <= 0
                row = np.zeros(3 * n)
                row[n + i] = -1.0
                row[n + j] = 1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
    
    # Lower bounds: s_i >= 0
    bounds = [(0, None)] * (3 * n)
    # cx, cy can be any real, but effectively bounded by constraints
    for i in range(n):
        bounds[i] = (None, None)
        bounds[n + i] = (None, None)
        bounds[2 * n + i] = (1e-10, None)  # s_i > 0
    
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = np.array([]).reshape(0, 3*n)
        b_ub = np.array([])

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            cx = result.x[:n]
            cy = result.x[n:2*n]
            s = result.x[2*n:]
            centers = np.stack([cx, cy], axis=1)
            return centers, s, -result.fun
    except Exception:
        pass
    return None

def _generate_row_topologies(n, max_rows=4):
    """
    Generate separation patterns for row-based topologies.
    A row topology is defined by a list of row sizes (number of squares in each row).
    Squares in the same row are separated in X.
    Squares in different rows are separated in Y.
    """
    patterns = []
    
    # Generate all integer partitions of n into at most max_rows parts
    def get_partitions(n, max_parts):
        if n == 0 or max_parts == 0:
            yield []
            return
        for i in range(1, n + 1):
            for rest in get_partitions(n - i, max_parts - 1):
                yield [i] + rest

    # We want partitions like [3,3,4], [4,3,3], etc.
    # Order matters for the "rows" (bottom to top).
    for part in get_partitions(n, max_rows):
        # Skip if any part is 0 (handled by recursion)
        # Create separation matrix
        sep_x = np.zeros((n, n), dtype=int)
        sep_y = np.zeros((n, n), dtype=int)
        
        # Assign squares to rows
        square_indices = list(range(n))
        current_idx = 0
        row_squares = []
        for row_size in part:
            row_squares.append(square_indices[current_idx:current_idx+row_size])
            current_idx += row_size
        
        # Within each row, separate in X (left to right)
        for row in row_squares:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    idx_i = row[i]
                    idx_j = row[j]
                    sep_x[idx_i, idx_j] = 1
                    sep_x[idx_j, idx_i] = -1
        
        # Between rows, separate in Y (bottom to top)
        for r_i in range(len(row_squares)):
            for r_j in range(r_i + 1, len(row_squares)):
                for idx_i in row_squares[r_i]:
                    for idx_j in row_squares[r_j]:
                        sep_y[idx_i, idx_j] = 1
                        sep_y[idx_j, idx_i] = -1
        
        patterns.append((part, sep_x, sep_y))
        
    return patterns

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 630 - DEADLINE_MARGIN
    
    # Baseline grid packing
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
    best_sides = np.full(n, side, dtype=np.float64)
    best_val = np.sum(best_sides)
    
    # Try row-based topologies
    patterns = _generate_row_topologies(n, max_rows=4)
    
    # Sort patterns by a heuristic: prefer more balanced rows? 
    # Actually, just try all, they are fast.
    for part, sep_x, sep_y in patterns:
        if time.time() > deadline:
            break
        result = _solve_lp_for_separations(n, sep_x, sep_y)
        if result is not None:
            centers, s, val = result
            if _is_valid(centers, s, n):
                if val > best_val:
                    best_val = val
                    best_centers = centers
                    best_sides = s
    
    # Also try column-based topologies (transpose)
    # We can reuse the patterns by swapping x and y roles in the LP setup,
    # or just generate new patterns where rows are columns.
    # Easier: generate patterns for columns by swapping the logic in _generate_row_topologies
    # But for now, row topologies are sufficient to beat the seed.
    
    # Additional: Try a specific "mixed" topology if time permits
    # e.g. 2x2 block + strips
    # This is harder to generalize, so we stick to row partitions which are robust.
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
