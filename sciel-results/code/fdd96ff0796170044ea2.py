import numpy as np
import time
from scipy.optimize import linprog, minimize

def _is_feasible(centers, sides, tol=1e-7):
    """Check if the configuration is feasible."""
    n = len(sides)
    if n != 10:
        return False
    # Boundary check
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
    # Overlap check
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True

def _sum_sides(sides):
    """Objective: sum of side lengths."""
    return np.sum(sides)

def _flatten(centers, sides):
    """Flatten centers and sides into a single vector."""
    n = len(sides)
    return np.concatenate([centers.flatten(), sides])

def _unflatten(x):
    """Unflatten a vector back into centers and sides."""
    n = 10
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    return centers, sides

def _penalty(x, tol=1e-7, weight=1e5):
    """Penalty for constraint violations on flattened vector."""
    centers, sides = _unflatten(x)
    n = len(sides)
    pen = 0.0
    # Boundary penalties
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            pen += weight * (1.0 - s)
        if cx - s/2 < -tol:
            pen += weight * (-tol - (cx - s/2))
        if cx + s/2 > 1 + tol:
            pen += weight * ((cx + s/2) - (1 + tol))
        if cy - s/2 < -tol:
            pen += weight * (-tol - (cy - s/2))
        if cy + s/2 > 1 + tol:
            pen += weight * ((cy + s/2) - (1 + tol))
    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            if abs(dx) < t and abs(dy) < t:
                pen += weight * ((t - abs(dx)) + (t - abs(dy)))
    return pen

def _objective(x, tol=1e-7, weight=1e5):
    """Combined objective: maximize sum of sides minus penalty (negated for minimization)."""
    centers, sides = _unflatten(x)
    return -np.sum(sides) + _penalty(x, tol, weight)

def _make_grid_fallback():
    """Fallback grid configuration: 4x3 grid roughly."""
    n = 10
    # 4 cols, 3 rows (12 slots, use 10)
    cols = 4
    rows = 3
    side = 1.0 / cols
    idx = np.arange(n)
    c = idx % cols
    r = idx // cols
    cx = (c + 0.5) * side
    cy = (r + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides

def _make_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Grid fallback
    configs.append(_make_grid_fallback())
    
    # 2. 2x5 Grid
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    s = 0.2
    for i in range(5):
        centers[i] = [(i + 0.5) * s, 0.1]
        sides[i] = s
        centers[i+5] = [(i + 0.5) * s, 0.9]
        sides[i+5] = s
    configs.append((centers, sides))
    
    # 3. 5x2 Grid
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    s = 0.2
    for i in range(5):
        centers[i] = [0.1, (i + 0.5) * s]
        sides[i] = s
        centers[i+5] = [0.9, (i + 0.5) * s]
        sides[i+5] = s
    configs.append((centers, sides))
    
    # 4. Random uniform
    np.random.seed(123)
    centers = np.random.rand(10, 2)
    sides = np.full(10, 0.1)
    configs.append((centers, sides))
    
    # 5. Clustered
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    # 5 big, 5 small
    s_big = 0.3
    s_small = 0.15
    for i in range(5):
        centers[i] = [(i % 2) * 0.5 + 0.15, (i // 2) * 0.5 + 0.15]
        sides[i] = s_big
    for i in range(5):
        centers[i+5] = [(i % 2) * 0.5 + 0.65, (i // 2) * 0.5 + 0.65]
        sides[i+5] = s_small
    configs.append((centers, sides))
    
    return configs

def _solve_lp_topology(pattern, tol=1e-7):
    """
    Solve LP for a given separation pattern.
    pattern: (10, 10) matrix, pattern[i, j] = 0 for horizontal sep, 1 for vertical sep.
    Assumes i < j. 
    Direction is fixed: if horizontal, x_j >= x_i + (s_i+s_j)/2.
    If vertical, y_j >= y_i + (s_i+s_j)/2.
    """
    n = 10
    nvar = 3 * n # x, y, s for each
    
    c = np.zeros(nvar)
    for i in range(n):
        c[3*i + 2] = -1.0 # Maximize sum(s) -> Minimize -sum(s)
        
    A_rows = []
    b_vals = []
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((1e-6, 1.0)) # s
        
    # Constraints
    # 1. Boundary
    for i in range(n):
        # x_i - s_i/2 >= 0  => -x_i + 0.5*s_i <= 0
        row = np.zeros(nvar)
        row[3*i] = -1.0
        row[3*i+2] = 0.5
        A_rows.append(row)
        b_vals.append(0.0)
        
        # x_i + s_i/2 <= 1  => x_i + 0.5*s_i <= 1
        row = np.zeros(nvar)
        row[3*i] = 1.0
        row[3*i+2] = 0.5
        A_rows.append(row)
        b_vals.append(1.0)
        
        # y_i - s_i/2 >= 0
        row = np.zeros(nvar)
        row[3*i+1] = -1.0
        row[3*i+2] = 0.5
        A_rows.append(row)
        b_vals.append(0.0)
        
        # y_i + s_i/2 <= 1
        row = np.zeros(nvar)
        row[3*i+1] = 1.0
        row[3*i+2] = 0.5
        A_rows.append(row)
        b_vals.append(1.0)
        
    # 2. Separation
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(nvar)
            if pattern[i, j] == 0: # Horizontal: x_j - x_i >= (s_i+s_j)/2
                # -x_i + x_j - 0.5*s_i - 0.5*s_j >= 0
                # x_i - x_j + 0.5*s_i + 0.5*s_j <= 0
                row[3*i] = 1.0
                row[3*j] = -1.0
                row[3*i+2] = 0.5
                row[3*j+2] = 0.5
            else: # Vertical: y_j - y_i >= (s_i+s_j)/2
                # y_i - y_j + 0.5*s_i + 0.5*s_j <= 0
                row[3*i+1] = 1.0
                row[3*j+1] = -1.0
                row[3*i+2] = 0.5
                row[3*j+2] = 0.5
            A_rows.append(row)
            b_vals.append(0.0)
            
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x = res.x
            centers = np.zeros((n, 2))
            sides = np.zeros(n)
            for i in range(n):
                centers[i, 0] = x[3*i]
                centers[i, 1] = x[3*i+1]
                sides[i] = x[3*i+2]
            return centers, sides
        else:
            return None, None
    except Exception:
        return None, None

def _generate_patterns(n=10):
    """Generate a diverse set of separation patterns."""
    patterns = []
    
    # 1. All Horizontal
    p = np.zeros((n, n))
    patterns.append(p)
    
    # 2. All Vertical
    p = np.ones((n, n))
    patterns.append(p)
    
    # 3. Row-based: Horizontal within rows, Vertical between rows
    # Assuming 5 cols, 2 rows
    p = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            row_i = i // 5
            row_j = j // 5
            if row_i == row_j:
                p[i, j] = 0 # Horizontal
            else:
                p[i, j] = 1 # Vertical
    patterns.append(p)
    
    # 4. Column-based: Vertical within cols, Horizontal between cols
    # Assuming 2 cols, 5 rows
    p = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            col_i = i % 2
            col_j = j % 2
            if col_i == col_j:
                p[i, j] = 1 # Vertical
            else:
                p[i, j] = 0 # Horizontal
    patterns.append(p)
    
    # 5. Random patterns
    rng = np.random.RandomState(42)
    for _ in range(20):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                p[i, j] = rng.randint(0, 2)
        patterns.append(p)
        
    return patterns

def _local_search_pattern(pattern, current_best_val, time_limit):
    """
    Try to improve the pattern by flipping bits.
    """
    n = 10
    current_pattern = pattern.copy()
    current_val = current_best_val
    start_time = time.time()
    
    # Get current solution to check tightness
    centers, sides = _solve_lp_topology(current_pattern)
    if centers is None:
        return current_pattern, current_val, None, None
        
    val = np.sum(sides)
    if val < current_val:
        return current_pattern, current_val, None, None
        
    # Try flipping each bit
    for i in range(n):
        for j in range(i+1, n):
            if time.time() - start_time > time_limit:
                return current_pattern, current_val, centers, sides
                
            new_pattern = current_pattern.copy()
            new_pattern[i, j] = 1 - new_pattern[i, j]
            
            c_new, s_new = _solve_lp_topology(new_pattern)
            if c_new is not None and s_new is not None:
                v_new = np.sum(s_new)
                if v_new > current_val + 1e-9:
                    current_val = v_new
                    current_pattern = new_pattern
                    centers, sides = c_new, s_new
                    # Restart local search from this new best
                    # To keep it simple, we just continue scanning from this state
                    # but we don't restart the i,j loops to avoid infinite loops
                    # A more robust approach would be a proper local search, 
                    # but this greedy hill climb is fast.
                    
    return current_pattern, current_val, centers, sides

def _polish_with_nm(centers, sides, time_limit=10):
    """
    Polish the solution using Nelder-Mead on the penalty objective.
    This can help if the LP solution is on the boundary and small moves improve it.
    """
    x0 = _flatten(centers, sides)
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        return _objective(x)
        
    try:
        res = minimize(obj, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < -np.sum(sides) + 1e-6: # Improved
            x_opt = res.x
            c_opt, s_opt = _unflatten(x_opt)
            if _is_feasible(c_opt, s_opt):
                return c_opt, s_opt
    except Exception:
        pass
    return centers, sides

def squares_in_square10() -> tuple:
    start_time = time.time()
    deadline = start_time + 590 # 10s safety margin
    
    best_centers, best_sides = None, None
    best_val = 0.0
    
    # 1. Generate initial patterns
    patterns = _generate_patterns()
    
    # 2. Evaluate each pattern with LP
    for p in patterns:
        if time.time() > deadline - 30:
            break
            
        centers, sides = _solve_lp_topology(p)
        if centers is not None and sides is not None:
            if _is_feasible(centers, sides):
                val = np.sum(sides)
                if val > best_val:
                    best_val = val
                    best_centers = centers
                    best_sides = sides
                    
    # 3. If we have a good candidate, try local search on the pattern
    if best_centers is not None and time.time() < deadline - 20:
        # Find the pattern that produced the best result
        # We need to re-evaluate or store it. For now, let's just run local search on the best pattern found.
        # To do this properly, we should have stored the best pattern.
        # Let's re-run LP for the best pattern found in the loop above? 
        # Actually, we can just take the best pattern from the list that gave best_val.
        best_pattern = None
        for p in patterns:
            c, s = _solve_lp_topology(p)
            if c is not None and s is not None:
                if abs(np.sum(s) - best_val) < 1e-6:
                    best_pattern = p
                    break
        if best_pattern is not None:
            # Local search on pattern
            improved_pattern, improved_val, imp_centers, imp_sides = _local_search_pattern(
                best_pattern, best_val, time_limit=20
            )
            if improved_val > best_val and imp_centers is not None:
                if _is_feasible(imp_centers, imp_sides):
                    best_val = improved_val
                    best_centers = imp_centers
                    best_sides = imp_sides
                    
    # 4. Polish with Nelder-Mead
    if best_centers is not None and time.time() < deadline - 5:
        polished_centers, polished_sides = _polish_with_nm(best_centers, best_sides, time_limit=5)
        if _is_feasible(polished_centers, polished_sides):
            val = np.sum(polished_sides)
            if val > best_val:
                best_val = val
                best_centers = polished_centers
                best_sides = polished_sides
                
    # 5. Fallback
    if best_centers is None:
        best_centers, best_sides = _make_grid_fallback()
        
    # Final check
    if not _is_feasible(best_centers, best_sides):
        best_centers, best_sides = _make_grid_fallback()
        
    return best_centers, best_sides
