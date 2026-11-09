# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
from itertools import product

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
    for i in range(n):
        x0 = centers[i, 0] - sides[i] / 2
        x1 = centers[i, 0] + sides[i] / 2
        y0 = centers[i, 1] - sides[i] / 2
        y1 = centers[i, 1] + sides[i] / 2
        if x0 < -TOL - 1e-10 or x1 > 1 + TOL + 1e-10:
            return False
        if y0 < -TOL - 1e-10 or y1 > 1 + TOL + 1e-10:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _overlap_penalty(centers, sides, n=10):
    """Compute penalty for overlaps and boundary violations."""
    penalty = 0.0
    for i in range(n):
        x0 = centers[i, 0] - sides[i] / 2
        x1 = centers[i, 0] + sides[i] / 2
        y0 = centers[i, 1] - sides[i] / 2
        y1 = centers[i, 1] + sides[i] / 2
        if x0 < 0:
            penalty += (0 - x0) ** 2
        if x1 > 1:
            penalty += (x1 - 1) ** 2
        if y0 < 0:
            penalty += (0 - y0) ** 2
        if y1 > 1:
            penalty += (y1 - 1) ** 2
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                penalty += (t - dx) ** 2 + (t - dy) ** 2
    return penalty

def _objective(params, n=10):
    """Objective: minimize negative sum of sides + overlap penalty."""
    centers = params[:2 * n].reshape(n, 2)
    sides = params[2 * n:]
    sides = np.maximum(sides, 1e-10)
    obj = -np.sum(sides) + 100.0 * _overlap_penalty(centers, sides, n)
    return obj

def _local_optimize(centers, sides, n=10, max_iter=200, timeout=5.0):
    """Local optimization using Nelder-Mead."""
    start_time = time.time()
    params = np.concatenate([centers.flatten(), sides])
    
    def obj_with_timeout(p):
        if time.time() - start_time > timeout:
            return 1e10
        return _objective(p, n)
    
    res = minimize(obj_with_timeout, params, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10,
                            'adaptive': True})
    
    if res.fun < 1e10:
        new_centers = res.x[:2 * n].reshape(n, 2)
        new_sides = np.maximum(res.x[2 * n:], 1e-10)
        if _is_valid(new_centers, new_sides, n):
            val = np.sum(new_sides)
            if val > np.sum(sides):
                return new_centers, new_sides, val
    return None

def _solve_lp_for_row_topology(n, row_sizes, col_order_in_rows, deadline):
    """
    Solve LP for a specific row topology.
    row_sizes: list of ints summing to n, sizes of each row.
    col_order_in_rows: list of lists, each containing column indices (0-based) for squares in that row.
    """
    # Determine how many columns we need
    max_cols = max(len(cols) for cols in col_order_in_rows)
    
    # For each row, squares are ordered left to right by col_order
    # Variables: cx_i, cy_i, s_i
    c = np.zeros(3 * n)
    for i in range(n):
        c[2 * n + i] = -1.0
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    for i in range(n):
        row = np.zeros(3 * n)
        row[i] = -1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        row = np.zeros(3 * n)
        row[i] = 1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        row = np.zeros(3 * n)
        row[n + i] = -1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        row = np.zeros(3 * n)
        row[n + i] = 1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(1.0)
    
    # Row separation: rows are stacked vertically
    # For consecutive rows r and r+1, all squares in row r are above row r+1
    # cy_i - s_i/2 >= cy_j + s_j/2 for i in row r, j in row r+1 (if r < r+1, row r is above)
    # Actually let's say row 0 is at top, row 1 below it, etc.
    # So for row r and row r+1: cy_{i in r} - s_i/2 >= cy_{j in r+1} + s_j/2
    # => cy_i - cy_j - s_i/2 - s_j/2 >= 0
    # => -(cy_i - cy_j) + s_i/2 + s_j/2 <= 0
    
    # Map square index to (row, position_in_row)
    sq_to_row = {}
    sq_to_pos = {}
    idx = 0
    for r, size in enumerate(row_sizes):
        for pos in range(size):
            sq_to_row[idx] = r
            sq_to_pos[idx] = pos
            idx += 1
    
    # Vertical separation between consecutive rows
    for r in range(len(row_sizes) - 1):
        for i in range(idx):
            if sq_to_row.get(i) == r:
                for j in range(idx):
                    if sq_to_row.get(j) == r + 1:
                        row = np.zeros(3 * n)
                        row[n + i] = -1.0
                        row[n + j] = 1.0
                        row[2 * n + i] = 0.5
                        row[2 * n + j] = 0.5
                        A_ub.append(row)
                        b_ub.append(0.0)
    
    # Horizontal separation within each row
    for r in range(len(row_sizes)):
        cols = col_order_in_rows[r]
        for pos in range(len(cols) - 1):
            i = cols[pos]
            j = cols[pos + 1]
            # i is to the left of j: cx_i + s_i/2 <= cx_j - s_j/2
            # => cx_i - cx_j + s_i/2 + s_j/2 <= 0
            row = np.zeros(3 * n)
            row[i] = 1.0
            row[j] = -1.0
            row[2 * n + i] = 0.5
            row[2 * n + j] = 0.5
            A_ub.append(row)
            b_ub.append(0.0)
    
    bounds = [(None, None)] * (2 * n) + [(1e-10, None)] * n
    
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                        options={'time_limit': 2.0})
        if result.success:
            cx = result.x[:n]
            cy = result.x[n:2*n]
            s = result.x[2*n:]
            centers = np.stack([cx, cy], axis=1)
            val = -result.fun
            if _is_valid(centers, s, n):
                return centers, s, val
    except:
        pass
    return None

def _generate_row_topologies(n, deadline):
    """Generate diverse row topologies and solve LPs."""
    results = []
    
    # Generate all compositions of n into at most n parts (rows)
    # For small n=10, we can be systematic
    
    def compositions(n, max_parts):
        if max_parts == 1:
            yield [n]
        else:
            for first in range(1, n + 1):
                for rest in compositions(n - first, max_parts - 1):
                    yield [first] + rest
    
    # Try various row sizes
    row_size_options = []
    # 1 row
    row_size_options.append([10])
    # 2 rows
    for a in range(1, 10):
        row_size_options.append([a, 10 - a])
    # 3 rows
    for a in range(1, 9):
        for b in range(1, 10 - a):
            row_size_options.append([a, b, 10 - a - b])
    # 4 rows
    for a in range(1, 8):
        for b in range(1, 9 - a):
            for c in range(1, 10 - a - b):
                row_size_options.append([a, b, c, 10 - a - b - c])
    # 5 rows
    for a in range(1, 7):
        for b in range(1, 8 - a):
            for c in range(1, 9 - a - b):
                for d in range(1, 10 - a - b - c):
                    row_size_options.append([a, b, c, d, 10 - a - b - c - d])
    # More rows (up to 10)
    for a in range(1, 6):
        for b in range(1, 7 - a):
            for c in range(1, 8 - a - b):
                for d in range(1, 9 - a - b - c):
                    for e in range(1, 10 - a - b - c - d):
                        row_size_options.append([a, b, c, d, e, 10 - a - b - c - d - e])
    
    # Deduplicate by sorted row sizes (order of rows matters for optimization)
    # But we want to try different orderings too
    
    # For each row size configuration, try a few column orderings
    np.random.seed(42)
    tried = set()
    
    for row_sizes in row_size_options:
        if time.time() > deadline:
            break
        
        # Generate a few random column orderings for this row config
        for trial in range(3):
            col_orders = []
            valid = True
            for size in row_sizes:
                cols = list(range(size))
                if trial > 0:
                    np.random.shuffle(cols)
                col_orders.append(cols)
            
            key = tuple(tuple(c) for c in col_orders)
            if key in tried:
                continue
            tried.add(key)
            
            # Remap: square i is at position (row r, col c)
            # We need to assign actual square indices to positions
            # For simplicity, assign squares 0..n-1 in order to rows
            # Square index = sum of row_sizes[:r] + c
            
            # Actually, the LP doesn't care about which square is which,
            # just the topology. So we can just solve and get the result.
            
            result = _solve_lp_for_row_topology(n, row_sizes, col_orders, deadline)
            if result is not None:
                centers, s, val = result
                results.append((val, centers, s))
                
                # Try local optimization
                if time.time() < deadline:
                    opt_result = _local_optimize(centers, s, n, max_iter=100, timeout=3.0)
                    if opt_result is not None:
                        oc, os_, oval = opt_result
                        results.append((oval, oc, os_))
    
    return results

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
    
    # Try structured row topologies
    results = _generate_row_topologies(n, deadline)
    
    if results:
        results.sort(key=lambda x: -x[0])
        for val, centers, sides in results:
            if val > best_val:
                best_val = val
                best_centers = centers
                best_sides = sides
    
    # Check deadline
    if time.time() > deadline:
        return best_centers, best_sides
    
    # Try local optimization on the best found
    if time.time() < deadline:
        opt_result = _local_optimize(best_centers, best_sides, n, max_iter=500, timeout=10.0)
        if opt_result is not None:
            oc, os_, oval = opt_result
            if oval > best_val:
                best_val = oval
                best_centers = oc
                best_sides = os_
    
    # Try some random starts with local optimization
    if time.time() < deadline:
        np.random.seed(123)
        for _ in range(20):
            if time.time() > deadline:
                break
            # Random valid-ish start
            centers = np.random.uniform(0.1, 0.9, (n, 2))
            sides = np.random.uniform(0.1, 0.3, n)
            # Sort by y to create rough row structure
            order = np.argsort(centers[:, 1])
            centers = centers[order]
            sides = sides[order]
            opt_result = _local_optimize(centers, sides, n, max_iter=200, timeout=5.0)
            if opt_result is not None:
                oc, os_, oval = opt_result
                if oval > best_val:
                    best_val = oval
                    best_centers = oc
                    best_sides = os_
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
