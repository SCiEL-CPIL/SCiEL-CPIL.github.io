import numpy as np
import time
from scipy.optimize import linprog

def _is_feasible(centers, sides, tol=1e-7):
    """Check if the configuration is feasible."""
    n = len(sides)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True

def _solve_lp_topology(sep_matrix):
    """
    Solve LP for a given separation matrix.
    sep_matrix[i][j] is 0 if i is left of j, 1 if i is right of j, 
    2 if i is below j, 3 if i is above j.
    Only i < j entries are used.
    """
    n = 10
    # Variables: x_0..x_9, y_0..y_9, s_0..s_9 (30 vars)
    # Objective: max sum(s_i) -> min -sum(s_i)
    c = np.zeros(30)
    for i in range(n):
        c[20 + i] = -1.0
        
    A_ub = []
    b_ub = []
    
    # Bounds: x_i - s_i/2 >= 0 => -x_i + 0.5 s_i <= 0
    # x_i + s_i/2 <= 1 => x_i + 0.5 s_i <= 1
    # y_i - s_i/2 >= 0 => -y_i + 0.5 s_i <= 0
    # y_i + s_i/2 <= 1 => y_i + 0.5 s_i <= 1
    # s_i >= 0 (handled by bounds)
    
    for i in range(n):
        # -x_i + 0.5 s_i <= 0
        row = np.zeros(30)
        row[i] = -1.0
        row[20 + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # x_i + 0.5 s_i <= 1
        row = np.zeros(30)
        row[i] = 1.0
        row[20 + i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        # -y_i + 0.5 s_i <= 0
        row = np.zeros(30)
        row[n + i] = -1.0
        row[20 + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # y_i + 0.5 s_i <= 1
        row = np.zeros(30)
        row[n + i] = 1.0
        row[20 + i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
    # Separation constraints
    for i in range(n):
        for j in range(i+1, n):
            sep = sep_matrix[i][j]
            # x_j - x_i >= (s_i + s_j)/2  => -x_j + x_i + 0.5 s_i + 0.5 s_j <= 0
            # x_i - x_j >= (s_i + s_j)/2  => x_j - x_i + 0.5 s_i + 0.5 s_j <= 0
            # y_j - y_i >= (s_i + s_j)/2  => -y_j + y_i + 0.5 s_i + 0.5 s_j <= 0
            # y_i - y_j >= (s_i + s_j)/2  => y_j - y_i + 0.5 s_i + 0.5 s_j <= 0
            
            row = np.zeros(30)
            row[20 + i] = 0.5
            row[20 + j] = 0.5
            
            if sep == 0: # i left of j: x_j - x_i >= ...
                row[i] = -1.0
                row[j] = 1.0
            elif sep == 1: # i right of j: x_i - x_j >= ...
                row[i] = 1.0
                row[j] = -1.0
            elif sep == 2: # i below j: y_j - y_i >= ...
                row[n + i] = -1.0
                row[n + j] = 1.0
            elif sep == 3: # i above j: y_i - y_j >= ...
                row[n + i] = 1.0
                row[n + j] = -1.0
                
            A_ub.append(row)
            b_ub.append(0.0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, 1) for _ in range(20)] + [(0, 1) for _ in range(10)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.status == 0:
        x = res.x[:n]
        y = res.x[n:2*n]
        s = res.x[2*n:]
        centers = np.column_stack((x, y))
        return centers, s, -res.fun
    else:
        return None, None, -np.inf

def _make_topology_2big_8small():
    n = 10
    sep = np.zeros((n, n), dtype=int)
    # 0,1 are big. 2-9 are small.
    # 0 left of 1
    sep[0][1] = 0
    # 0,1 above 2-9
    for i in [0, 1]:
        for j in range(2, 10):
            sep[i][j] = 3 # i above j
    
    # Small squares: 2-5 in row 1 (bottom), 6-9 in row 2 (top of bottom section)
    # Actually, let's say 2,3,4,5 are bottom row. 6,7,8,9 are top row.
    # 2 left of 3, 3 left of 4, 4 left of 5
    sep[2][3] = 0
    sep[3][4] = 0
    sep[4][5] = 0
    # 6 left of 7, 7 left of 8, 8 left of 9
    sep[6][7] = 0
    sep[7][8] = 0
    sep[8][9] = 0
    # 2 below 6, 3 below 7, 4 below 8, 5 below 9
    sep[2][6] = 2
    sep[3][7] = 2
    sep[4][8] = 2
    sep[5][9] = 2
    # Also need to ensure 2 doesn't overlap 7 etc?
    # If 2 is left of 3 and 6 is above 2, and 7 is above 3...
    # We need to define relative order for all pairs to be safe in LP.
    # If not defined, LP might assume they can overlap in x or y if not constrained?
    # No, LP requires explicit constraints for non-overlap.
    # If we don't add a constraint, they can overlap.
    # So we MUST define sep for all pairs.
    
    # Let's fill in the rest with a consistent grid logic.
    # Indices:
    # 6 7 8 9
    # 2 3 4 5
    
    # Horizontal relations in row 2 (6-9) and row 1 (2-5) are set.
    # Vertical relations:
    # 2 below 6, 3 below 7, 4 below 8, 5 below 9.
    # What about 2 and 7? 2 is left of 3, 3 is below 7. So 2 is left-below 7.
    # We can set 2 left of 7? Or 2 below 7?
    # If we set 2 left of 7, it adds a constraint x_7 - x_2 >= ...
    # If we set 2 below 7, it adds y_7 - y_2 >= ...
    # Both are true in the grid. Adding both is redundant but safe.
    # To keep LP small, we can just add the "transitive" ones or just the direct grid neighbors?
    # No, non-neighbors can overlap if not constrained.
    # Example: 2 and 8. 2 is (0,0), 8 is (2,1).
    # If we don't constrain 2 and 8, LP might put 2 at (0.4, 0) and 8 at (0.5, 0.5)?
    # If 2 is wide and 8 is wide, they might overlap.
    # So we must constrain ALL pairs.
    
    # Let's define a helper to fill sep based on grid positions.
    # Grid coords (col, row) for 2-9:
    # 2: (0,0), 3: (1,0), 4: (2,0), 5: (3,0)
    # 6: (0,1), 7: (1,1), 8: (2,1), 9: (3,1)
    
    pos = {
        2: (0,0), 3: (1,0), 4: (2,0), 5: (3,0),
        6: (0,1), 7: (1,1), 8: (2,1), 9: (3,1)
    }
    
    for i in range(2, 10):
        for j in range(i+1, 10):
            if sep[i][j] != 0 and sep[i][j] != 1 and sep[i][j] != 2 and sep[i][j] != 3:
                # Default to 0, we will correct below
                pass
            
    # Re-evaluate all pairs 2-9
    for i in range(2, 10):
        for j in range(i+1, 10):
            ci, ri = pos[i]
            cj, rj = pos[j]
            if ci < cj:
                sep[i][j] = 0 # i left of j
            elif ci > cj:
                sep[i][j] = 1 # i right of j
            elif ri < rj:
                sep[i][j] = 2 # i below j
            else:
                sep[i][j] = 3 # i above j
                
    # Now handle 0,1 vs 2-9
    # 0 is top-left, 1 is top-right.
    # 0: (-1, 2), 1: (0, 2) roughly?
    # Let's say 0 is col 0-1, 1 is col 2-3.
    # Actually 0 is [0, 0.5], 1 is [0.5, 1].
    # Small squares are 0.25 wide.
    # 0 is above 2,3. 1 is above 4,5?
    # 2,3 are cols 0,1. 4,5 are cols 2,3.
    # So 0 is above 2,3. 1 is above 4,5.
    # And 0 is left of 1.
    
    # Update sep for 0,1 vs 2-9
    for i in [0, 1]:
        for j in range(2, 10):
            # i is above j
            sep[i][j] = 3
            
    # And 0 vs 1
    sep[0][1] = 0
    
    return sep

def squares_in_square10():
    start_time = time.time()
    deadline = start_time + 580
    
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    # Topology 1: 2 Big + 8 Small
    sep1 = _make_topology_2big_8small()
    c1, s1, val1 = _solve_lp_topology(sep1)
    if c1 is not None and val1 > best_val:
        if _is_feasible(c1, s1):
            best_centers = c1
            best_sides = s1
            best_val = val1
            
    # Fallback to hardcoded if LP fails or is worse
    if best_val < 3.0:
        centers = np.array([
            [0.25, 0.75], [0.75, 0.75],
            [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
            [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375]
        ])
        sides = np.array([0.5, 0.5] + [0.25]*8)
        if _is_feasible(centers, sides):
            best_centers = centers
            best_sides = sides
            best_val = np.sum(sides)
            
    return best_centers, best_sides
