# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import linprog, minimize

TOL = 1e-7
DEADLINE_MARGIN = 30.0

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

def _solve_lp_for_pattern(n, pattern):
    """
    Given a separation pattern (for each pair (i,j), which axis separates them),
    solve the LP to maximize sum of sides.
    
    pattern: dict mapping (i,j) -> 'x' or 'y' for i < j
    Variables: s_0, ..., s_{n-1}, x_0, ..., x_{n-1}, y_0, ..., y_{n-1}
    where x_i = left edge of square i, y_i = bottom edge of square i.
    So center_i = (x_i + s_i/2, y_i + s_i/2)
    
    Maximize: sum(s_i)
    Subject to:
      0 <= x_i, x_i + s_i <= 1
      0 <= y_i, y_i + s_i <= 1
      s_i >= 0
      For each pair (i,j) with pattern 'x': x_i + s_i <= x_j OR x_j + s_j <= x_i
        We linearize by fixing the direction: assume x_i + s_i <= x_j (i is left of j)
      For each pair (i,j) with pattern 'y': y_i + s_i <= y_j OR y_j + s_j <= y_i
        Assume y_i + s_i <= y_j (i is below j)
    """
    # Variables: s[0..n-1], x[0..n-1], y[0..n-1]
    # Index: s_i at i, x_i at n+i, y_i at 2*n+i
    def s_idx(i): return i
    def x_idx(i): return n + i
    def y_idx(i): return 2 * n + i
    
    # Objective: maximize sum(s_i) -> minimize -sum(s_i)
    c = np.zeros(3 * n)
    for i in range(n):
        c[s_idx(i)] = -1.0
    
    # Inequality constraints: A_ub @ vars <= b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # Boundary: x_i + s_i <= 1  ->  s_i + x_i <= 1
    for i in range(n):
        row = np.zeros(3 * n)
        row[s_idx(i)] = 1.0
        row[x_idx(i)] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)
    
    # Boundary: y_i + s_i <= 1
    for i in range(n):
        row = np.zeros(3 * n)
        row[s_idx(i)] = 1.0
        row[y_idx(i)] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)
    
    # Separation constraints
    for (i, j), axis in pattern.items():
        if axis == 'x':
            # i is to the left of j: x_i + s_i <= x_j
            # -> s_i + x_i - x_j <= 0
            row = np.zeros(3 * n)
            row[s_idx(i)] = 1.0
            row[x_idx(i)] = 1.0
            row[x_idx(j)] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
        else:  # 'y'
            # i is below j: y_i + s_i <= y_j
            # -> s_i + y_i - y_j <= 0
            row = np.zeros(3 * n)
            row[s_idx(i)] = 1.0
            row[y_idx(i)] = 1.0
            row[y_idx(j)] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    
    # Bounds: s_i >= 0, x_i >= 0, y_i >= 0, x_i <= 1, y_i <= 1, s_i <= 1
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # s_i
    for i in range(n):
        bounds.append((0.0, 1.0))  # x_i
    for i in range(n):
        bounds.append((0.0, 1.0))  # y_i
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            s = result.x[:n]
            x = result.x[n:2*n]
            y = result.x[2*n:3*n]
            centers = np.column_stack([x + s / 2, y + s / 2])
            return centers, s, np.sum(s)
    except:
        pass
    return None, None, -np.inf

def _generate_patterns(n, max_patterns=None, seed=42):
    """
    Generate separation patterns. For n squares, there are C(n,2) pairs,
    each with 2 choices (x or y separation). Total 2^(n*(n-1)/2) patterns.
    For n=10, that's 2^45 which is too many. We use structured patterns.
    """
    rng = np.random.RandomState(seed)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_pairs = len(pairs)
    
    patterns = []
    
    # Pattern 1: all x-separation (squares in a row)
    patterns.append({(i,j): 'x' for i, j in pairs})
    
    # Pattern 2: all y-separation (squares in a column)
    patterns.append({(i,j): 'y' for i, j in pairs})
    
    # Pattern 3: grid-like (row-major: same row -> x, different row -> y)
    # Try different grid configurations
    for rows in range(1, n+1):
        for cols in range(1, n+1):
            if rows * cols < n:
                continue
            pat = {}
            for i in range(n):
                ri, ci = divmod(i, cols)
            for (i, j) in pairs:
                ri_i, ci_i = divmod(i, cols)
                ri_j, ci_j = divmod(j, cols)
                if ri_i == ri_j:
                    pat[(i,j)] = 'x'
                else:
                    pat[(i,j)] = 'y'
            patterns.append(pat)
    
    # Random patterns
    if max_patterns is not None:
        remaining = max_patterns - len(patterns)
    else:
        remaining = 200
    
    for _ in range(max(0, remaining)):
        pat = {}
        for (i, j) in pairs:
            pat[(i,j)] = 'x' if rng.random() < 0.5 else 'y'
        patterns.append(pat)
    
    return patterns

def _local_polish(centers, sides, n, max_iter=100, deadline=None):
    """
    Local improvement: try to increase each side slightly while maintaining validity.
    """
    best_val = np.sum(sides)
    centers = centers.copy()
    sides = sides.copy()
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            # Try to increase side i
            for delta in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                new_sides = sides.copy()
                new_sides[i] += delta
                if new_sides[i] <= 0 or new_sides[i] > 1:
                    continue
                # Check if still valid
                if _is_valid(centers, new_sides, n):
                    sides = new_sides
                    best_val = np.sum(sides)
                    improved = True
                    break
        if not improved:
            break
    
    return centers, sides, best_val

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
    
    # Enumerate separation patterns and solve LP for each
    patterns = _generate_patterns(n, max_patterns=300, seed=42)
    
    for pat in patterns:
        if time.time() > deadline:
            break
        centers_p, sides_p, val_p = _solve_lp_for_pattern(n, pat)
        if centers_p is not None and _is_valid(centers_p, sides_p, n):
            if val_p > best_val:
                best_val = val_p
                best_centers = centers_p.copy()
                best_sides = sides_p.copy()
    
    # Local polish from the best solution
    if time.time() < deadline - 10:
        centers_p, sides_p, val_p = _local_polish(best_centers, best_sides, n, max_iter=200, deadline=deadline)
        if val_p > best_val:
            best_val = val_p
            best_centers = centers_p.copy()
            best_sides = sides_p.copy()
    
    # Try local polish from a few other good LP solutions
    if time.time() < deadline - 10:
        # Re-solve a few top patterns and polish
        top_patterns = patterns[:20]
        for pat in top_patterns:
            if time.time() > deadline:
                break
            centers_p, sides_p, val_p = _solve_lp_for_pattern(n, pat)
            if centers_p is not None and _is_valid(centers_p, sides_p, n):
                if val_p >= best_val - 0.01:  # Close to best, try polishing
                    centers_p, sides_p, val_p = _local_polish(centers_p, sides_p, n, max_iter=100, deadline=deadline)
                    if val_p > best_val:
                        best_val = val_p
                        best_centers = centers_p.copy()
                        best_sides = sides_p.copy()
    
    # Final validation
    if not _is_valid(best_centers, best_sides, n):
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
