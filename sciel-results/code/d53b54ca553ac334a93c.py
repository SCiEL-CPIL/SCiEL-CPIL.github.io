# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time
import itertools
import random

def squares_in_square10() -> tuple:
    n = 10
    deadline = time.time() + 600.0
    
    # Variables per square: cx, cy, s -> 3n variables total
    # We'll index as: var_i = [cx_i, cy_i, s_i]
    # Total variables: 30
    
    def build_lp(sep_pattern):
        """
        sep_pattern: list of length n*(n-1)//2, each entry is 0,1,2,3
          0: i left of j  -> cx_i + s_i/2 <= cx_j - s_j/2
          1: j left of i  -> cx_j + s_j/2 <= cx_i - s_i/2
          2: i below j    -> cy_i + s_i/2 <= cy_j - s_j/2
          3: j below i    -> cy_j + s_j/2 <= cy_i - s_i/2
        """
        # Objective: maximize sum of s_i -> minimize -sum(s_i)
        # c vector: for each square i, c[3*i+0]=0, c[3*i+1]=0, c[3*i+2]=-1
        c = np.zeros(3 * n)
        for i in range(n):
            c[3 * i + 2] = -1.0
        
        # Inequality constraints A_ub @ x <= b_ub
        A_rows = []
        b_rows = []
        
        # Boundary constraints:
        # cx_i - s_i/2 >= 0  ->  -cx_i + s_i/2 <= 0
        # cx_i + s_i/2 <= 1  ->   cx_i - s_i/2 <= 1
        # cy_i - s_i/2 >= 0  ->  -cy_i + s_i/2 <= 0
        # cy_i + s_i/2 <= 1  ->   cy_i - s_i/2 <= 1
        # s_i > 0 handled by lower bounds
        
        for i in range(n):
            # -cx_i + s_i/2 <= 0
            row = np.zeros(3 * n)
            row[3 * i + 0] = -1.0
            row[3 * i + 2] = 0.5
            A_rows.append(row)
            b_rows.append(0.0)
            
            # cx_i - s_i/2 <= 1
            row = np.zeros(3 * n)
            row[3 * i + 0] = 1.0
            row[3 * i + 2] = -0.5
            A_rows.append(row)
            b_rows.append(1.0)
            
            # -cy_i + s_i/2 <= 0
            row = np.zeros(3 * n)
            row[3 * i + 1] = -1.0
            row[3 * i + 2] = 0.5
            A_rows.append(row)
            b_rows.append(0.0)
            
            # cy_i - s_i/2 <= 1
            row = np.zeros(3 * n)
            row[3 * i + 1] = 1.0
            row[3 * i + 2] = -0.5
            A_rows.append(row)
            b_rows.append(1.0)
        
        # Separation constraints
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                sep = sep_pattern[pair_idx]
                row = np.zeros(3 * n)
                if sep == 0:  # i left of j: cx_i + s_i/2 <= cx_j - s_j/2
                    # cx_i - cx_j + s_i/2 + s_j/2 <= 0
                    row[3 * i + 0] = 1.0
                    row[3 * j + 0] = -1.0
                    row[3 * i + 2] = 0.5
                    row[3 * j + 2] = 0.5
                    b_val = 0.0
                elif sep == 1:  # j left of i: cx_j + s_j/2 <= cx_i - s_i/2
                    # -cx_i + cx_j + s_i/2 + s_j/2 <= 0
                    row[3 * i + 0] = -1.0
                    row[3 * j + 0] = 1.0
                    row[3 * i + 2] = 0.5
                    row[3 * j + 2] = 0.5
                    b_val = 0.0
                elif sep == 2:  # i below j: cy_i + s_i/2 <= cy_j - s_j/2
                    # cy_i - cy_j + s_i/2 + s_j/2 <= 0
                    row[3 * i + 1] = 1.0
                    row[3 * j + 1] = -1.0
                    row[3 * i + 2] = 0.5
                    row[3 * j + 2] = 0.5
                    b_val = 0.0
                else:  # sep == 3: j below i: cy_j + s_j/2 <= cy_i - s_i/2
                    # -cy_i + cy_j + s_i/2 + s_j/2 <= 0
                    row[3 * i + 1] = -1.0
                    row[3 * j + 1] = 1.0
                    row[3 * i + 2] = 0.5
                    row[3 * j + 2] = 0.5
                    b_val = 0.0
                A_rows.append(row)
                b_rows.append(b_val)
                pair_idx += 1
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        # Bounds: s_i > 0, we use small epsilon
        eps = 1e-12
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # cx_i
            bounds.append((0.0, 1.0))  # cy_i
            bounds.append((eps, 1.0))  # s_i
        
        return c, A_ub, b_ub, bounds
    
    def solve_lp(sep_pattern):
        """Solve LP for given separation pattern. Returns (objective_value, x) or (None, None)."""
        c, A_ub, b_ub, bounds = build_lp(sep_pattern)
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                return res.fun * -1, res.x  # fun is -sum(s_i), so negating gives sum
            else:
                return None, None
        except Exception:
            return None, None
    
    def extract_solution(x):
        """Extract centers and sides from LP solution vector."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        for i in range(n):
            centers[i, 0] = x[3 * i + 0]
            centers[i, 1] = x[3 * i + 1]
            sides[i] = x[3 * i + 2]
        return centers, sides
    
    def validate(centers, sides, tol=1e-7):
        """Check if solution is valid."""
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # Boundary check
        for i in range(n):
            if centers[i, 0] - sides[i] / 2 < -tol or centers[i, 0] + sides[i] / 2 > 1 + tol:
                return False
            if centers[i, 1] - sides[i] / 2 < -tol or centers[i, 1] + sides[i] / 2 > 1 + tol:
                return False
        # Overlap check
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True
    
    def get_sep_pattern_from_solution(centers, sides):
        """Infer separation pattern from a valid solution."""
        sep_pattern = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                half_i = sides[i] / 2
                half_j = sides[j] / 2
                # Determine which separation is active
                # i left of j: dx >= half_i + half_j
                # j left of i: -dx >= half_i + half_j
                # i below j: dy >= half_i + half_j
                # j below i: -dy >= half_i + half_j
                if dx >= half_i + half_j - 1e-9:
                    sep_pattern.append(0)
                elif -dx >= half_i + half_j - 1e-9:
                    sep_pattern.append(1)
                elif dy >= half_i + half_j - 1e-9:
                    sep_pattern.append(2)
                else:
                    sep_pattern.append(3)
        return sep_pattern
    
    # Baseline: grid solution
    k = int(np.ceil(np.sqrt(n)))
    side_grid = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_grid = (cols + 0.5) * side_grid
    cy_grid = (rows + 0.5) * side_grid
    centers_grid = np.stack([cx_grid, cy_grid], axis=1).astype(np.float64)
    sides_grid = np.full(n, side_grid, dtype=np.float64)
    
    best_value = np.sum(sides_grid)
    best_centers = centers_grid.copy()
    best_sides = sides_grid.copy()
    
    # Generate initial separation patterns
    sep_patterns = []
    
    # Pattern 1: from grid solution
    sep_patterns.append(get_sep_pattern_from_solution(centers_grid, sides_grid))
    
    # Pattern 2: all x-separations (sorted by x)
    sep_all_x = []
    for i in range(n):
        for j in range(i + 1, n):
            sep_all_x.append(0)  # i left of j
    sep_patterns.append(sep_all_x)
    
    # Pattern 3: all y-separations (sorted by y)
    sep_all_y = []
    for i in range(n):
        for j in range(i + 1, n):
            sep_all_y.append(2)  # i below j
    sep_patterns.append(sep_all_y)
    
    # Random patterns
    random.seed(42)
    for _ in range(20):
        sep_patterns.append([random.randint(0, 3) for _ in range(n * (n - 1) // 2)])
    
    # Solve LPs for initial patterns
    for sep_pat in sep_patterns:
        if time.time() > deadline - 30:
            break
        val, x = solve_lp(sep_pat)
        if val is not None:
            centers, sides = extract_solution(x)
            if validate(centers, sides):
                if val > best_value:
                    best_value = val
                    best_centers = centers.copy()
                    best_sides = sides.copy()
    
    # Iterative improvement: use best solution to generate new patterns
    max_iterations = 100
    for it in range(max_iterations):
        if time.time() > deadline - 30:
            break
        
        # Get separation pattern from best solution
        sep_pat = get_sep_pattern_from_solution(best_centers, best_sides)
        
        # Generate variations by flipping some separation directions
        for _ in range(10):
            if time.time() > deadline - 30:
                break
            new_sep = sep_pat.copy()
            # Flip 1-3 random separations
            num_flips = random.randint(1, 3)
            for _ in range(num_flips):
                idx_flip = random.randint(0, len(new_sep) - 1)
                new_sep[idx_flip] = (new_sep[idx_flip] + 1) % 4  # cycle through options
                # Actually, let's just pick a random different option
                new_sep[idx_flip] = random.choice([0, 1, 2, 3])
            
            val, x = solve_lp(new_sep)
            if val is not None:
                centers, sides = extract_solution(x)
                if validate(centers, sides):
                    if val > best_value:
                        best_value = val
                        best_centers = centers.copy()
                        best_sides = sides.copy()
    
    # Final polish: try more random patterns if time remains
    while time.time() < deadline - 60:
        sep_pat = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        val, x = solve_lp(sep_pat)
        if val is not None:
            centers, sides = extract_solution(x)
            if validate(centers, sides):
                if val > best_value:
                    best_value = val
                    best_centers = centers.copy()
                    best_sides = sides.copy()
    
    # Ensure final solution is valid
    if not validate(best_centers, best_sides):
        # Fall back to grid
        best_centers = centers_grid
        best_sides = sides_grid
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
