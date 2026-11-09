# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from itertools import product

TOL = 1e-7
DEADLINE_MARGIN = 15.0  # seconds before 630s limit

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
    sep_x[i,j] = 1 if square i is to the left of j, -1 if right, 0 if not separated in x.
    sep_y similarly for y.
    """
    # Variables: cx_i (0..n-1), cy_i (n..2n-1), s_i (2n..3n-1)
    c = np.zeros(3 * n)
    for i in range(n):
        c[2 * n + i] = -1.0  # maximize s_i
    
    A_ub = []
    b_ub = []
    
    # Inside unit square constraints
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
                # cx_i + s_i/2 <= cx_j - s_j/2 => cx_i - cx_j + s_i/2 + s_j/2 <= 0
                row = np.zeros(3 * n)
                row[i] = 1.0
                row[j] = -1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
            elif sep_x[i, j] == -1:
                # cx_j + s_j/2 <= cx_i - s_i/2 => -cx_i + cx_j + s_i/2 + s_j/2 <= 0
                row = np.zeros(3 * n)
                row[i] = -1.0
                row[j] = 1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
            
            if sep_y[i, j] == 1:
                # cy_i + s_i/2 <= cy_j - s_j/2 => cy_i - cy_j + s_i/2 + s_j/2 <= 0
                row = np.zeros(3 * n)
                row[n + i] = 1.0
                row[n + j] = -1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
            elif sep_y[i, j] == -1:
                # cy_j + s_j/2 <= cy_i - s_i/2 => -cy_i + cy_j + s_i/2 + s_j/2 <= 0
                row = np.zeros(3 * n)
                row[n + i] = -1.0
                row[n + j] = 1.0
                row[2 * n + i] = 0.5
                row[2 * n + j] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
    
    bounds = [(0, None)] * (3 * n)
    for i in range(n):
        bounds[i] = (None, None)
        bounds[n + i] = (None, None)
        bounds[2 * n + i] = (1e-10, None)
    
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
    """
    patterns = []
    
    def get_partitions(n, max_parts):
        if n == 0:
            yield []
            return
        if max_parts == 0:
            return
        for i in range(1, n + 1):
            for rest in get_partitions(n - i, max_parts - 1):
                yield [i] + rest

    for part in get_partitions(n, max_rows):
        sep_x = np.zeros((n, n), dtype=int)
        sep_y = np.zeros((n, n), dtype=int)
        
        square_indices = list(range(n))
        current_idx = 0
        row_squares = []
        for row_size in part:
            row_squares.append(square_indices[current_idx:current_idx+row_size])
            current_idx += row_size
        
        # Within each row, separate in X
        for row in row_squares:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    idx_i = row[i]
                    idx_j = row[j]
                    sep_x[idx_i, idx_j] = 1
                    sep_x[idx_j, idx_i] = -1
        
        # Between rows, separate in Y
        for r_i in range(len(row_squares)):
            for r_j in range(r_i + 1, len(row_squares)):
                for idx_i in row_squares[r_i]:
                    for idx_j in row_squares[r_j]:
                        sep_y[idx_i, idx_j] = 1
                        sep_y[idx_j, idx_i] = -1
        
        patterns.append((part, sep_x, sep_y))
        
    return patterns

def _generate_col_topologies(n, max_cols=4):
    """
    Generate separation patterns for column-based topologies.
    """
    patterns = []
    
    def get_partitions(n, max_parts):
        if n == 0:
            yield []
            return
        if max_parts == 0:
            return
        for i in range(1, n + 1):
            for rest in get_partitions(n - i, max_parts - 1):
                yield [i] + rest

    for part in get_partitions(n, max_cols):
        sep_x = np.zeros((n, n), dtype=int)
        sep_y = np.zeros((n, n), dtype=int)
        
        square_indices = list(range(n))
        current_idx = 0
        col_squares = []
        for col_size in part:
            col_squares.append(square_indices[current_idx:current_idx+col_size])
            current_idx += col_size
        
        # Within each column, separate in Y
        for col in col_squares:
            for i in range(len(col)):
                for j in range(i + 1, len(col)):
                    idx_i = col[i]
                    idx_j = col[j]
                    sep_y[idx_i, idx_j] = 1
                    sep_y[idx_j, idx_i] = -1
        
        # Between columns, separate in X
        for c_i in range(len(col_squares)):
            for c_j in range(c_i + 1, len(col_squares)):
                for idx_i in col_squares[c_i]:
                    for idx_j in col_squares[c_j]:
                        sep_x[idx_i, idx_j] = 1
                        sep_x[idx_j, idx_i] = -1
        
        patterns.append((part, sep_x, sep_y))
        
    return patterns

def _solve_milp(n):
    """
    Solve the full MILP for the packing problem.
    Variables:
    - x_i, y_i, s_i for i in 0..n-1 (continuous)
    - d_x[i,j], d_y[i,j] for i < j (binary)
    
    Constraints:
    - Boundary: 0 <= x_i - s_i/2, x_i + s_i/2 <= 1, etc.
    - Non-overlap: |x_i - x_j| + |y_i - y_j| >= (s_i + s_j)/2
      This is linearized using big-M constraints with binary variables.
      For each pair (i, j), we need one of 4 conditions:
      1. x_i + s_i/2 <= x_j - s_j/2  (i left of j)
      2. x_j + s_j/2 <= x_i - s_i/2  (j left of i)
      3. y_i + s_i/2 <= y_j - s_j/2  (i below j)
      4. y_j + s_j/2 <= y_i - s_i/2  (j below i)
      
      We introduce 4 binary variables per pair: b1, b2, b3, b4
      b1 + b2 + b3 + b4 >= 1
      Constraint 1: x_i - x_j + s_i/2 + s_j/2 <= M * (1 - b1)
      Constraint 2: -x_i + x_j + s_i/2 + s_j/2 <= M * (1 - b2)
      Constraint 3: y_i - y_j + s_i/2 + s_j/2 <= M * (1 - b3)
      Constraint 4: -y_i + y_j + s_i/2 + s_j/2 <= M * (1 - b4)
    """
    n_pairs = n * (n - 1) // 2
    pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Variable indices
    # x_i: 0..n-1
    # y_i: n..2n-1
    # s_i: 2n..3n-1
    # b1[i,j]: 3n + 4*k
    # b2[i,j]: 3n + 4*k + 1
    # b3[i,j]: 3n + 4*k + 2
    # b4[i,j]: 3n + 4*k + 3
    # where k is the index of the pair (i, j)
    
    num_vars = 3 * n + 4 * n_pairs
    M = 1.0  # Big-M, since all variables are in [0, 1]
    
    # Objective: maximize sum(s_i) -> minimize -sum(s_i)
    c = np.zeros(num_vars)
    for i in range(n):
        c[2 * n + i] = -1.0
    
    # Constraints
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    # Boundary constraints
    for i in range(n):
        # -x_i + s_i/2 <= 0
        row = np.zeros(num_vars)
        row[i] = -1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # x_i - s_i/2 <= 1
        row = np.zeros(num_vars)
        row[i] = 1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        # -y_i + s_i/2 <= 0
        row = np.zeros(num_vars)
        row[n + i] = -1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # y_i - s_i/2 <= 1
        row = np.zeros(num_vars)
        row[n + i] = 1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(1.0)
    
    # Non-overlap constraints
    for k, (i, j) in enumerate(pair_indices):
        base_idx = 3 * n + 4 * k
        b1_idx = base_idx
        b2_idx = base_idx + 1
        b3_idx = base_idx + 2
        b4_idx = base_idx + 3
        
        # b1 + b2 + b3 + b4 >= 1 => -(b1 + b2 + b3 + b4) <= -1
        row = np.zeros(num_vars)
        row[b1_idx] = -1.0
        row[b2_idx] = -1.0
        row[b3_idx] = -1.0
        row[b4_idx] = -1.0
        A_ub.append(row)
        b_ub.append(-1.0)
        
        # x_i - x_j + s_i/2 + s_j/2 <= M * (1 - b1)
        # x_i - x_j + s_i/2 + s_j/2 + M * b1 <= M
        row = np.zeros(num_vars)
        row[i] = 1.0
        row[j] = -1.0
        row[2 * n + i] = 0.5
        row[2 * n + j] = 0.5
        row[b1_idx] = M
        A_ub.append(row)
        b_ub.append(M)
        
        # -x_i + x_j + s_i/2 + s_j/2 <= M * (1 - b2)
        # -x_i + x_j + s_i/2 + s_j/2 + M * b2 <= M
        row = np.zeros(num_vars)
        row[i] = -1.0
        row[j] = 1.0
        row[2 * n + i] = 0.5
        row[2 * n + j] = 0.5
        row[b2_idx] = M
        A_ub.append(row)
        b_ub.append(M)
        
        # y_i - y_j + s_i/2 + s_j/2 <= M * (1 - b3)
        # y_i - y_j + s_i/2 + s_j/2 + M * b3 <= M
        row = np.zeros(num_vars)
        row[n + i] = 1.0
        row[n + j] = -1.0
        row[2 * n + i] = 0.5
        row[2 * n + j] = 0.5
        row[b3_idx] = M
        A_ub.append(row)
        b_ub.append(M)
        
        # -y_i + y_j + s_i/2 + s_j/2 <= M * (1 - b4)
        # -y_i + y_j + s_i/2 + s_j/2 + M * b4 <= M
        row = np.zeros(num_vars)
        row[n + i] = -1.0
        row[n + j] = 1.0
        row[2 * n + i] = 0.5
        row[2 * n + j] = 0.5
        row[b4_idx] = M
        A_ub.append(row)
        b_ub.append(M)
    
    # Bounds
    integrality = np.zeros(num_vars)
    for k in range(n_pairs):
        base_idx = 3 * n + 4 * k
        integrality[base_idx:base_idx + 4] = 1
    
    bounds = Bounds(
        lb=np.zeros(num_vars),
        ub=np.ones(num_vars)
    )
    # x, y, s are in [0, 1]
    # But s can be up to 1, x and y can be in [0, 1]
    # Actually, centers can be in [0, 1], sides in [0, 1]
    # Wait, x_i is the center, so x_i in [0, 1] is correct for the center.
    # But the constraint x_i - s_i/2 >= 0 implies x_i >= s_i/2.
    # And x_i + s_i/2 <= 1 implies x_i <= 1 - s_i/2.
    # So x_i is effectively bounded.
    
    # Let's use LinearConstraint for clarity if available, otherwise stick to A_ub/b_ub
    # scipy.optimize.milp requires LinearConstraint
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        constraint = LinearConstraint(A_ub, -np.inf, b_ub)
    else:
        constraint = LinearConstraint(np.array([]).reshape(0, num_vars), -np.inf, np.array([]))
    
    try:
        res = milp(
            c=c,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 30}
        )
        if res.success:
            x = res.x[:n]
            y = res.x[n:2*n]
            s = res.x[2*n:3*n]
            centers = np.stack([x, y], axis=1)
            return centers, s, -res.fun
    except Exception:
        pass
    return None

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
    
    # 1. Try row-based topologies
    patterns = _generate_row_topologies(n, max_rows=4)
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
    
    # 2. Try column-based topologies
    patterns = _generate_col_topologies(n, max_cols=4)
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
    
    # 3. Try MILP if time permits
    if time.time() < deadline - 30:
        result = _solve_milp(n)
        if result is not None:
            centers, s, val = result
            if _is_valid(centers, s, n):
                if val > best_val:
                    best_val = val
                    best_centers = centers
                    best_sides = s
    
    # 4. Refine with local search if time permits
    if time.time() < deadline - 5:
        # Simple local search: try to increase sides slightly and adjust positions
        # This is a placeholder for a more sophisticated refinement
        pass
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
