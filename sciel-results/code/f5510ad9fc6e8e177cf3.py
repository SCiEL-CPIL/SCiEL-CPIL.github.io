# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize, linprog

def _is_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t > 0:
                if dx < t and dy < t:
                    return False
    return True

def _objective(centers, sides, penalty_weight=1e6):
    """
    Objective: sum(sides) - penalty. Higher is better.
    Uses smooth penalty for boundary and overlap violations.
    """
    n = len(sides)
    penalty = 0.0
    
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # Boundary penalties (smooth)
        if cx - s/2 < 0:
            penalty += (cx - s/2)**2
        if cx + s/2 > 1:
            penalty += (cx + s/2 - 1)**2
        if cy - s/2 < 0:
            penalty += (cy - s/2)**2
        if cy + s/2 > 1:
            penalty += (cy + s/2 - 1)**2
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0
            if dx < t and dy < t:
                overlap_x = t - dx
                overlap_y = t - dy
                # Quadratic penalty for overlap area
                penalty += overlap_x * overlap_y * 10
    
    return np.sum(sides) - penalty_weight * penalty

def _lp_pack_fixed_sides(sides, deadline):
    """
    For fixed side lengths, solve LP to find valid positions.
    Uses binary ordering variables for each pair in each dimension.
    Returns centers if feasible, else None.
    """
    n = len(sides)
    
    # Variables: cx[0..n-1], cy[0..n-1], then binary vars for ordering
    # For each pair (i,j) with i<j: x_order[i,j] = 1 if cx_i < cx_j, else 0
    # Similarly y_order[i,j] = 1 if cy_i < cy_j, else 0
    
    pairs = list(itertools.combinations(range(n), 2))
    m = len(pairs)  # number of pairs
    
    # Total variables: 2*n (positions) + 2*m (binary orderings)
    total_vars = 2 * n + 2 * m
    
    # Objective: minimize 0 (just feasibility)
    c_obj = np.zeros(total_vars)
    
    # A_ub, b_ub for inequality constraints
    A_ub_rows = []
    b_ub_vals = []
    
    # Boundary constraints: s_i/2 <= cx_i <= 1 - s_i/2
    # -cx_i <= -s_i/2  =>  cx_i >= s_i/2
    # cx_i <= 1 - s_i/2
    
    # We'll use A_ub * x <= b_ub form
    # For cx_i >= s_i/2: -cx_i <= -s_i/2
    for i in range(n):
        row = np.zeros(total_vars)
        row[i] = -1.0  # -cx_i
        A_ub_rows.append(row)
        b_ub_vals.append(-sides[i] / 2.0)
        
        row2 = np.zeros(total_vars)
        row2[i] = 1.0  # cx_i
        A_ub_rows.append(row2)
        b_ub_vals.append(1.0 - sides[i] / 2.0)
        
        # cy_i >= s_i/2: -cy_i <= -s_i/2
        row3 = np.zeros(total_vars)
        row3[n + i] = -1.0  # -cy_i
        A_ub_rows.append(row3)
        b_ub_vals.append(-sides[i] / 2.0)
        
        # cy_i <= 1 - s_i/2
        row4 = np.zeros(total_vars)
        row4[n + i] = 1.0  # cy_i
        A_ub_rows.append(row4)
        b_ub_vals.append(1.0 - sides[i] / 2.0)
    
    # Non-overlap constraints for each pair (i, j)
    # Either: cx_i + s_i/2 <= cx_j - s_j/2 (i is left of j)
    #   OR: cx_j + s_j/2 <= cx_i - s_i/2 (j is left of i)
    #   OR: cy_i + s_i/2 <= cy_j - s_j/2 (i is below j)
    #   OR: cy_j + s_j/2 <= cy_i - s_i/2 (j is below i)
    
    # Using big-M formulation with binary variables:
    # x_order[i,j] = 1 means cx_i < cx_j (i is left)
    # If x_order=1: cx_i + s_i/2 <= cx_j - s_j/2 + M*(1-x_order)
    # If x_order=0: cx_j + s_j/2 <= cx_i - s_i/2 + M*x_order
    
    # Actually, let's use a different approach. For each pair, we need:
    # (cx_j - cx_i >= (s_i+s_j)/2) OR (cx_i - cx_j >= (s_i+s_j)/2) OR
    # (cy_j - cy_i >= (s_i+s_j)/2) OR (cy_i - cy_j >= (s_i+s_j)/2)
    
    # This is a disjunctive constraint. We can use binary variables.
    # Let a_ij = 1 if i is left of j, b_ij = 1 if i is right of j
    # Let c_ij = 1 if i is below j, d_ij = 1 if i is above j
    # Constraint: a_ij + b_ij + c_ij + d_ij >= 1
    # And: cx_j - cx_i >= (s_i+s_j)/2 - M*(1-a_ij)
    #      cx_i - cx_j >= (s_i+s_j)/2 - M*(1-b_ij)
    #      cy_j - cy_i >= (s_i+s_j)/2 - M*(1-c_ij)
    #      cy_i - cy_j >= (s_i+s_j)/2 - M*(1-d_ij)
    
    # This gives 4 binary vars per pair. Let's simplify by using 2 binary vars per pair
    # and noting that we need at least one separation.
    
    # Alternative: use the fact that for non-overlap, at least one of the four
    # separations must hold. We can enumerate or use a smarter formulation.
    
    # For now, let's use a simpler approach: try all 2^(2*m) orderings is too many.
    # Instead, let's use a heuristic: fix the ordering based on initial positions,
    # then solve LP with those fixed orderings.
    
    # Actually, for the LP approach to work well, we need to fix the relative orderings.
    # Let's return None and use a different strategy in the main function.
    
    return None

def _scipy_optimize(centers, sides, deadline):
    """
    Uses scipy.optimize.minimize to refine the configuration.
    """
    n = len(sides)
    # Flatten to vector: [cx0, cy0, s0, cx1, cy1, s1, ...]
    x0 = np.concatenate([centers.flatten(), sides])
    
    def fun(x):
        # Check time
        if time.time() > deadline:
            return 1e10 # Large penalty to stop
        
        c = x[:2*n].reshape(n, 2)
        s = x[2*n:]
        
        # Clip to valid ranges to help optimizer
        s_clipped = np.clip(s, 0.01, 0.5)
        
        c_clipped = np.zeros_like(c)
        for i in range(n):
            c_clipped[i, 0] = np.clip(c[i, 0], s_clipped[i]/2, 1 - s_clipped[i]/2)
            c_clipped[i, 1] = np.clip(c[i, 1], s_clipped[i]/2, 1 - s_clipped[i]/2)
            
        # We want to maximize sum(s) - penalty
        # So minimize - (sum(s) - penalty)
        obj = _objective(c_clipped, s_clipped)
        return -obj

    # Use Nelder-Mead with more iterations and tighter tolerance
    try:
        res = minimize(fun, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
    except Exception:
        return centers, sides

    x_opt = res.x
    c_opt = x_opt[:2*n].reshape(n, 2)
    s_opt = x_opt[2*n:]
    
    # Final clipping
    s_opt = np.clip(s_opt, 0.01, 0.5)
    for i in range(n):
        c_opt[i, 0] = np.clip(c_opt[i, 0], s_opt[i]/2, 1 - s_opt[i]/2)
        c_opt[i, 1] = np.clip(c_opt[i, 1], s_opt[i]/2, 1 - s_opt[i]/2)
        
    return c_opt, s_opt

def _local_search(centers, sides, max_iters=5000, deadline=None, rng_seed=42):
    """
    Aggressive local search with grow-bias and multi-square perturbations.
    """
    n = len(sides)
    best_score = np.sum(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    rng = np.random.RandomState(rng_seed)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Strategy 1: Perturb a single square with grow bias
        i = rng.randint(n)
        for attempt in range(12):
            # Strong grow bias: 80% chance to grow
            grow = rng.random() < 0.8
            if grow:
                s_new = best_s[i] * (1 + rng.uniform(0.02, 0.12))
            else:
                s_new = best_s[i] * (1 + rng.uniform(-0.05, 0.05))
            s_new = max(0.01, min(0.5, s_new))
            
            cx_new = best_c[i, 0] + rng.uniform(-0.08, 0.08)
            cy_new = best_c[i, 1] + rng.uniform(-0.08, 0.08)
            
            cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
            cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
            
            c_new = best_c.copy()
            s_new_arr = best_s.copy()
            c_new[i] = [cx_new, cy_new]
            s_new_arr[i] = s_new
            
            if _is_valid(c_new, s_new_arr):
                score = np.sum(s_new_arr)
                if score > best_score + 1e-12:
                    best_score = score
                    best_c = c_new
                    best_s = s_new_arr
                    improved = True
                    break
        
        # Strategy 2: Perturb two squares simultaneously
        if not improved and rng.random() < 0.5:
            i1 = rng.randint(n)
            i2 = rng.randint(n)
            if i1 != i2:
                for attempt in range(6):
                    c_new = best_c.copy()
                    s_new_arr = best_s.copy()
                    for idx in [i1, i2]:
                        grow = rng.random() < 0.7
                        if grow:
                            s_new = s_new_arr[idx] * (1 + rng.uniform(0.02, 0.08))
                        else:
                            s_new = s_new_arr[idx] * (1 + rng.uniform(-0.04, 0.04))
                        s_new = max(0.01, min(0.5, s_new))
                        cx_new = c_new[idx, 0] + rng.uniform(-0.05, 0.05)
                        cy_new = c_new[idx, 1] + rng.uniform(-0.05, 0.05)
                        cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
                        cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
                        c_new[idx] = [cx_new, cy_new]
                        s_new_arr[idx] = s_new
                    
                    if _is_valid(c_new, s_new_arr):
                        score = np.sum(s_new_arr)
                        if score > best_score + 1e-12:
                            best_score = score
                            best_c = c_new
                            best_s = s_new_arr
                            improved = True
                            break
        
        # Strategy 3: Perturb three squares (less frequent)
        if not improved and rng.random() < 0.2:
            indices = rng.choice(n, size=3, replace=False)
            for attempt in range(4):
                c_new = best_c.copy()
                s_new_arr = best_s.copy()
                for idx in indices:
                    grow = rng.random() < 0.6
                    if grow:
                        s_new = s_new_arr[idx] * (1 + rng.uniform(0.01, 0.06))
                    else:
                        s_new = s_new_arr[idx] * (1 + rng.uniform(-0.03, 0.03))
                    s_new = max(0.01, min(0.5, s_new))
                    cx_new = c_new[idx, 0] + rng.uniform(-0.04, 0.04)
                    cy_new = c_new[idx, 1] + rng.uniform(-0.04, 0.04)
                    cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
                    cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
                    c_new[idx] = [cx_new, cy_new]
                    s_new_arr[idx] = s_new
                
                if _is_valid(c_new, s_new_arr):
                    score = np.sum(s_new_arr)
                    if score > best_score + 1e-12:
                        best_score = score
                        best_c = c_new
                        best_s = s_new_arr
                        improved = True
                        break
        
        # Strategy 4: Try to push all squares to corners/edges (structural move)
        if not improved and rng.random() < 0.05:
            c_new = best_c.copy()
            s_new_arr = best_s.copy()
            # Try to snap one square to a corner
            i = rng.randint(n)
            s = s_new_arr[i]
            corners = [(s/2, s/2), (1-s/2, s/2), (s/2, 1-s/2), (1-s/2, 1-s/2)]
            for cx_try, cy_try in corners:
                c_new[i] = [cx_try, cy_try]
                if _is_valid(c_new, s_new_arr):
                    score = np.sum(s_new_arr)
                    if score > best_score + 1e-12:
                        best_score = score
                        best_c = c_new
                        best_s = s_new_arr
                        improved = True
                        break
                if improved:
                    break
    
    return best_c, best_s

def _coordinate_descent(centers, sides, max_iters=300, deadline=None):
    """
    Coordinate descent: for each square, optimize its position and size.
    """
    n = len(sides)
    best_score = np.sum(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(n):
            best_local_score = np.sum(best_s)
            best_local_c = best_c.copy()
            best_local_s = best_s.copy()
            
            # Try to grow the square and find a valid position
            for growth in [1.05, 1.1, 1.15, 1.2]:
                s_try = best_s[i] * growth
                if s_try > 0.5:
                    continue
                # Try multiple positions
                for _ in range(50):
                    cx = np.random.uniform(s_try/2, 1 - s_try/2)
                    cy = np.random.uniform(s_try/2, 1 - s_try/2)
                    
                    c_try = best_local_c.copy()
                    s_try_arr = best_local_s.copy()
                    c_try[i] = [cx, cy]
                    s_try_arr[i] = s_try
                    
                    if _is_valid(c_try, s_try_arr):
                        score = np.sum(s_try_arr)
                        if score > best_local_score:
                            best_local_score = score
                            best_local_c = c_try
                            best_local_s = s_try_arr
                            break
            
            # Now optimize position with fixed size (try to move to free space)
            s_fixed = best_local_s[i]
            for _ in range(30):
                cx = np.random.uniform(s_fixed/2, 1 - s_fixed/2)
                cy = np.random.uniform(s_fixed/2, 1 - s_fixed/2)
                
                c_try = best_local_c.copy()
                c_try[i] = [cx, cy]
                
                if _is_valid(c_try, best_local_s):
                    score = np.sum(best_local_s)
                    if score > best_local_score:
                        best_local_score = score
                        best_local_c = c_try
            
            best_c = best_local_c
            best_s = best_local_s
            best_score = best_local_score
    
    return best_c, best_s

def _fixed_sides_lp(centers, sides, deadline):
    """
    For fixed side lengths, use LP to find valid positions.
    We fix the relative ordering based on initial centers.
    """
    n = len(sides)
    
    # Determine ordering from initial centers
    # For x: sort by cx
    x_order = np.argsort(centers[:, 0])
    # For y: sort by cy
    y_order = np.argsort(centers[:, 1])
    
    # Build LP constraints
    # Variables: cx[0..n-1], cy[0..n-1]
    # Objective: minimize 0 (feasibility)
    
    c_obj = np.zeros(2 * n)
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    for i in range(n):
        # cx_i >= s_i/2: -cx_i <= -s_i/2
        row = np.zeros(2 * n)
        row[i] = -1.0
        A_ub.append(row)
        b_ub.append(-sides[i] / 2.0)
        
        # cx_i <= 1 - s_i/2
        row2 = np.zeros(2 * n)
        row2[i] = 1.0
        A_ub.append(row2)
        b_ub.append(1.0 - sides[i] / 2.0)
        
        # cy_i >= s_i/2
        row3 = np.zeros(2 * n)
        row3[n + i] = -1.0
        A_ub.append(row3)
        b_ub.append(-sides[i] / 2.0)
        
        # cy_i <= 1 - s_i/2
        row4 = np.zeros(2 * n)
        row4[n + i] = 1.0
        A_ub.append(row4)
        b_ub.append(1.0 - sides[i] / 2.0)
    
    # Non-overlap based on fixed ordering
    # For x: if i comes before j in x_order, then cx_i + s_i/2 <= cx_j - s_j/2
    # i.e., cx_j - cx_i >= (s_i + s_j) / 2
    # i.e., -cx_i + cx_j >= (s_i + s_j) / 2
    # i.e., cx_i - cx_j <= -(s_i + s_j) / 2
    
    for pos in range(n - 1):
        i = x_order[pos]
        j = x_order[pos + 1]
        # cx_i - cx_j <= -(s_i + s_j) / 2
        row = np.zeros(2 * n)
        row[i] = 1.0
        row[j] = -1.0
        A_ub.append(row)
        b_ub.append(-(sides[i] + sides[j]) / 2.0)
    
    # Similarly for y
    for pos in range(n - 1):
        i = y_order[pos]
        j = y_order[pos + 1]
        # cy_i - cy_j <= -(s_i + s_j) / 2
        row = np.zeros(2 * n)
        row[n + i] = 1.0
        row[n + j] = -1.0
        A_ub.append(row)
        b_ub.append(-(sides[i] + sides[j]) / 2.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((sides[i] / 2.0, 1.0 - sides[i] / 2.0))
        bounds.append((sides[i] / 2.0, 1.0 - sides[i] / 2.0))
    
    try:
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return None
    
    if result.success:
        cx = result.x[:n]
        cy = result.x[n:]
        centers_new = np.column_stack([cx, cy])
        return centers_new
    else:
        return None

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Fallback: 2x5 grid
    fallback_c = np.zeros((n, 2))
    fallback_s = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        fallback_c[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        fallback_s[i] = 0.2
    update_best(fallback_c, fallback_s)
    
    # Generate diverse initial configurations
    configs = []
    
    # 2x5 grid
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c1[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s1[i] = 0.2
    configs.append((c1, s1))
    
    # 1x10 strip
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    for i in range(n):
        c2[i] = [(i + 0.5) * 0.1, 0.5]
        s2[i] = 0.1
    configs.append((c2, s2))
    
    # 5x2 grid
    c3 = np.zeros((n, 2))
    s3 = np.zeros(n)
    for i in range(n):
        r = i % 2
        c = i // 2
        c3[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s3[i] = 0.2
    configs.append((c3, s3))
    
    # 3x3 grid + 1
    c4 = np.zeros((n, 2))
    s4 = np.zeros(n)
    for i in range(9):
        r = i // 3
        c = i % 3
        c4[i] = [(c + 0.5) / 3.0, (r + 0.5) / 3.0]
        s4[i] = 1.0/3.0
    c4[9] = [0.5, 0.5]
    s4[9] = 0.1
    configs.append((c4, s4))
    
    # 4x4 grid subset (4 squares per row, 2.5 rows)
    c5 = np.zeros((n, 2))
    s5 = np.zeros(n)
    for i in range(8):
        r = i // 4
        c = i % 4
        c5[i] = [(c + 0.5) * 0.25, (r + 0.5) * 0.4]
        s5[i] = 0.25
    # Last 2 squares in bottom row
    c5[8] = [0.125, 0.8]
    s5[8] = 0.25
    c5[9] = [0.375, 0.8]
    s5[9] = 0.25
    configs.append((c5, s5))
    
    # Asymmetric: large squares in corners, smaller in center
    c6 = np.zeros((n, 2))
    s6 = np.zeros(n)
    # Four large corner squares
    s_large = 0.3
    c6[0] = [s_large/2, s_large/2]
    s6[0] = s_large
    c6[1] = [1-s_large/2, s_large/2]
    s6[1] = s_large
    c6[2] = [s_large/2, 1-s_large/2]
    s6[2] = s_large
    c6[3] = [1-s_large/2, 1-s_large/2]
    s6[3] = s_large
    # Six smaller squares filling gaps
    s_small = 0.2
    positions = [
        [0.5, 0.15], [0.5, 0.35], [0.5, 0.65], [0.5, 0.85],
        [0.15, 0.5], [0.85, 0.5]
    ]
    for i in range(6):
        c6[4+i] = positions[i]
        s6[4+i] = s_small
    configs.append((c6, s6))
    
    # Mixed sizes: 2 large, 8 medium
    c7 = np.zeros((n, 2))
    s7 = np.zeros(n)
    # Two large squares
    s_lg = 0.35
    c7[0] = [s_lg/2, s_lg/2]
    s7[0] = s_lg
    c7[1] = [1-s_lg/2, 1-s_lg/2]
    s7[1] = s_lg
    # Eight medium squares
    s_md = 0.18
    remaining_positions = [
        [1-s_lg/2, s_lg/2], [s_lg/2, 1-s_lg/2],
        [0.5, 0.2], [0.5, 0.8], [0.2, 0.5], [0.8, 0.5],
        [0.5, 0.5], [0.3, 0.3]
    ]
    for i in range(8):
        c7[2+i] = remaining_positions[i]
        s7[2+i] = s_md
    configs.append((c7, s7))
    
    # Random initializations with greedy placement (more trials, wider size range)
    rng = np.random.RandomState(123)
    for trial in range(60):
        if time.time() > deadline - 30:
            break
        
        # Mix of size distributions
        if trial % 3 == 0:
            s_rand = rng.uniform(0.1, 0.38, n)
        elif trial % 3 == 1:
            s_rand = rng.uniform(0.08, 0.3, n)
        else:
            s_rand = np.concatenate([rng.uniform(0.25, 0.4, 3), rng.uniform(0.1, 0.25, 7)])
        
        c_rand = np.zeros((n, 2))
        placed = np.zeros(n, dtype=bool)
        
        order = np.argsort(-s_rand)
        
        for idx in order:
            s = s_rand[idx]
            found = False
            for _ in range(500):
                cx = rng.uniform(s/2, 1 - s/2)
                cy = rng.uniform(s/2, 1 - s/2)
                c_rand[idx] = [cx, cy]
                
                ok = True
                for j in range(n):
                    if placed[j] and j != idx:
                        dx = abs(c_rand[idx, 0] - c_rand[j, 0])
                        dy = abs(c_rand[idx, 1] - c_rand[j, 1])
                        t = (s_rand[idx] + s_rand[j]) / 2.0
                        if dx < t and dy < t:
                            ok = False
                            break
                if ok:
                    placed[idx] = True
                    found = True
                    break
            
            if not found:
                # Try smaller size
                s_rand[idx] = max(0.05, s_rand[idx] * 0.75)
                s = s_rand[idx]
                for _ in range(200):
                    cx = rng.uniform(s/2, 1 - s/2)
                    cy = rng.uniform(s/2, 1 - s/2)
                    c_rand[idx] = [cx, cy]
                    
                    ok = True
                    for j in range(n):
                        if placed[j] and j != idx:
                            dx = abs(c_rand[idx, 0] - c_rand[j, 0])
                            dy = abs(c_rand[idx, 1] - c_rand[j, 1])
                            t = (s_rand[idx] + s_rand[j]) / 2.0
                            if dx < t and dy < t:
                                ok = False
                                break
                    if ok:
                        placed[idx] = True
                        found = True
                        break
        
        if _is_valid(c_rand, s_rand):
            configs.append((c_rand, s_rand))
    
    # For each config, run optimization
    for c_init, s_init in configs:
        if time.time() > deadline - 15:
            break
        
        update_best(c_init, s_init)
        
        # 1. Try LP with fixed sides to find better positions
        if time.time() < deadline - 15:
            c_lp = _fixed_sides_lp(c_init, s_init, deadline)
            if c_lp is not None:
                update_best(c_lp, s_init)
        
        # 2. Run SciPy Optimizer
        if time.time() < deadline - 15:
            c_scipy, s_scipy = _scipy_optimize(c_init, s_init, deadline)
            update_best(c_scipy, s_scipy)
            
            # 3. Try LP on the result
            if time.time() < deadline - 10:
                c_lp2 = _fixed_sides_lp(c_scipy, s_scipy, deadline)
                if c_lp2 is not None:
                    update_best(c_lp2, s_scipy)
            
            # 4. Run Local Search with more iterations
            if time.time() < deadline - 10:
                c_ls, s_ls = _local_search(c_scipy, s_scipy, max_iters=800, deadline=deadline, rng_seed=42)
                update_best(c_ls, s_ls)
                
                # 5. Second round of local search with different seed
                if time.time() < deadline - 8:
                    c_ls2, s_ls2 = _local_search(c_ls, s_ls, max_iters=500, deadline=deadline, rng_seed=999)
                    update_best(c_ls2, s_ls2)
        
    # Final polish: try to squeeze more out of the best configuration
    if best_centers is not None and time.time() < deadline - 30:
        # Try LP on best
        c_final_lp = _fixed_sides_lp(best_centers, best_sides, deadline)
        if c_final_lp is not None:
            update_best(c_final_lp, best_sides)
        
        c_final, s_final = _scipy_optimize(best_centers, best_sides, deadline)
        update_best(c_final, s_final)
        
        # Multiple local search passes with different seeds
        if time.time() < deadline - 15:
            c_final2, s_final2 = _local_search(c_final, s_final, max_iters=1000, deadline=deadline, rng_seed=777)
            update_best(c_final2, s_final2)
        
        if time.time() < deadline - 10:
            c_final3, s_final3 = _local_search(c_final2, s_final2, max_iters=800, deadline=deadline, rng_seed=1234)
            update_best(c_final3, s_final3)
        
        # Final coordinate descent pass
        if time.time() < deadline - 5:
            c_cd, s_cd = _coordinate_descent(best_centers, best_sides, max_iters=200, deadline=deadline)
            update_best(c_cd, s_cd)
    
    if best_centers is None:
        best_centers = fallback_c
        best_sides = fallback_s
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
