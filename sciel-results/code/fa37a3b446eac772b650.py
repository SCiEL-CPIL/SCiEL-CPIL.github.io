# EVOLVE-BLOCK-START
import numpy as np
import time
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
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if dx < t and dy < t:
                return False
    return True

def _smooth_penalty(params, n=10, tol=1e-7, beta=50.0):
    """Smooth penalty for non-linear optimization."""
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    pen = 0.0
    
    # Boundary penalties (smooth)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            pen += 1e6 * (1.0 - s)
            continue
        # Left
        v = -tol - (cx - s/2)
        if v > 0:
            pen += 1e6 * v
        # Right
        v = (cx + s/2) - (1 + tol)
        if v > 0:
            pen += 1e6 * v
        # Bottom
        v = -tol - (cy - s/2)
        if v > 0:
            pen += 1e6 * v
        # Top
        v = (cy + s/2) - (1 + tol)
        if v > 0:
            pen += 1e6 * v
            
    # Overlap penalties (smooth soft-max based)
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides[i] + sides[j]) / 2.0 - tol
            
            # For separation, we need |dx| >= t OR |dy| >= t
            # Violation if |dx| < t AND |dy| < t
            # Smooth approximation of max(0, t - |dx|) * max(0, t - |dy|)
            # Use soft-plus for smoothness
            vx = t - abs(dx)
            vy = t - abs(dy)
            
            # Smooth max(0, x) approx: x + (1/beta)*log(1 + exp(-beta*x))
            # But we want product, so use log-sum-exp trick or just direct smooth max
            # Simple approach: use max(0, x)^2 smoothed
            sx = np.maximum(0, vx)
            sy = np.maximum(0, vy)
            pen += 1e5 * (sx * sy)
            
    return pen

def _objective(params, n=10, tol=1e-7):
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    pen = _smooth_penalty(params, n, tol)
    
    # Minimize negative sum
    obj = -np.sum(sides) + pen
    return obj

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580
    
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

    # --- Initial Configurations ---
    
    # Config 1: Seed (2 large 0.5, 8 small 0.25)
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    c1[0] = [0.25, 0.75]; s1[0] = 0.5
    c1[1] = [0.75, 0.75]; s1[1] = 0.5
    # Bottom 2x4 grid of 0.25
    for i in range(4):
        r = i // 4
        c = i % 4
        c1[2+i] = [(c + 0.5)*0.25, 0.25 + r*0.25]
        s1[2+i] = 0.25
    # Wait, 2+i goes up to 5. Need 8 small.
    # Indices 2..9
    idx = 2
    for r in range(2):
        for c in range(4):
            c1[idx] = [(c + 0.5)*0.25, 0.125 + r*0.25]
            s1[idx] = 0.25
            idx += 1
    update_best(c1, s1)
    
    # Config 2: 10 squares of 0.25 (2x5 grid)
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c2[i] = [(c + 0.5)*0.2, (r + 0.5)*0.5]
        s2[i] = 0.2
    update_best(c2, s2)
        
    # Config 3: 4 corners 0.4, 6 center 0.2
    c3 = np.zeros((n, 2))
    s3 = np.zeros(n)
    a = 0.4
    b = 0.2
    c3[0] = [a/2, a/2]; s3[0] = a
    c3[1] = [1-a/2, a/2]; s3[1] = a
    c3[2] = [a/2, 1-a/2]; s3[2] = a
    c3[3] = [1-a/2, 1-a/2]; s3[3] = a
    # Center 0.2x0.2
    c3[4] = [0.5, 0.5]; s3[4] = b
    # Arms
    # Top arm: 2 squares
    c3[5] = [0.4, 0.8]; s3[5] = b
    c3[6] = [0.6, 0.8]; s3[6] = b
    # Bottom arm: 2 squares
    c3[7] = [0.4, 0.2]; s3[7] = b
    c3[8] = [0.6, 0.2]; s3[8] = b
    # Left arm: 1 square?
    c3[9] = [0.2, 0.5]; s3[9] = b
    update_best(c3, s3)

    # Config 4: LP-based initialization with fixed ordering
    def _lp_init(fixed_orderings=None):
        """
        Solve LP for square packing with given ordering constraints.
        Variables: x_i, y_i (left edge), s_i (side)
        Constraints:
        - 0 <= x_i, x_i + s_i <= 1
        - 0 <= y_i, y_i + s_i <= 1
        - For each pair (i,j): either x_i + s_i <= x_j OR x_j + s_j <= x_i OR y_i + s_i <= y_j OR y_j + s_j <= y_i
        """
        if fixed_orderings is None:
            # Try a few common orderings
            fixed_orderings = []
        
        best_lp = None
        best_lp_val = -np.inf
        
        # Variable indices: 0..9 for x_i, 10..19 for y_i, 20..29 for s_i
        # Total 30 variables
        
        # Objective: maximize sum(s_i) -> minimize -sum(s_i)
        c_obj = np.zeros(30)
        for i in range(n):
            c_obj[20+i] = -1.0
        
        # Inequality constraints A_ub @ x <= b_ub
        A_ub_rows = []
        b_ub_vals = []
        
        # Boundary: x_i >= 0 -> -x_i <= 0
        for i in range(n):
            row = np.zeros(30)
            row[i] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            
        # x_i + s_i <= 1
        for i in range(n):
            row = np.zeros(30)
            row[i] = 1.0
            row[20+i] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)
            
        # y_i >= 0 -> -y_i <= 0
        for i in range(n):
            row = np.zeros(30)
            row[10+i] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            
        # y_i + s_i <= 1
        for i in range(n):
            row = np.zeros(30)
            row[10+i] = 1.0
            row[20+i] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)
            
        # s_i >= 0.1 (minimum size) -> -s_i <= -0.1
        for i in range(n):
            row = np.zeros(30)
            row[20+i] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(-0.1)
        
        # Pairwise non-overlap: for each pair, we need one of 4 orderings
        # We'll try different orderings
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Try a few ordering patterns
        order_patterns = [
            # Pattern 0: default grid-like
            {('i', 'j'): 0 for i, j in pairs},  # 0: i left of j, 1: j left of i, 2: i below j, 3: j below i
        ]
        
        # Generate some heuristic orderings based on config 1
        # Config 1: 2 large on top, 8 small in 2x4 grid below
        # Large: 0,1. Small: 2-9
        # For large-large: 0 left of 1
        # For large-small: large above small
        # For small-small: grid ordering
        
        ordering = {}
        # Large-large
        ordering[(0, 1)] = 0  # 0 left of 1
        # Large-small
        for i in [0, 1]:
            for j in range(2, 10):
                ordering[(i, j)] = 2  # i below j? No, i is above j means i.y + i.s <= j.y
                # Wait, if i is above j, then y_i + s_i <= y_j
                # So constraint: y_i + s_i - y_j <= 0
                # That's ordering 2 if we define 2 as "i below j" (y_i + s_i <= y_j)
                # But I want i above j, so y_j + s_j <= y_i, which is ordering 3
                ordering[(i, j)] = 3  # j below i (y_j + s_j <= y_i)
        
        # Small-small: 2x4 grid
        # Row 0: 2,3,4,5. Row 1: 6,7,8,9
        # In each row, left to right
        for r in range(2):
            for c in range(4):
                i = 2 + r*4 + c
                for c2 in range(c+1, 4):
                    j = 2 + r*4 + c2
                    ordering[(i, j)] = 0  # i left of j (x_i + s_i <= x_j)
        
        # Between rows, row 0 above row 1
        for i in range(2, 6):
            for j in range(6, 10):
                ordering[(i, j)] = 3  # j below i (y_j + s_j <= y_i)
        
        # Add ordering constraints to LP
        for (i, j), ord_val in ordering.items():
            row = np.zeros(30)
            if ord_val == 0:  # i left of j: x_i + s_i <= x_j
                row[i] = 1.0
                row[20+i] = 1.0
                row[j] = -1.0
            elif ord_val == 1:  # j left of i: x_j + s_j <= x_i
                row[j] = 1.0
                row[20+j] = 1.0
                row[i] = -1.0
            elif ord_val == 2:  # i below j: y_i + s_i <= y_j
                row[10+i] = 1.0
                row[20+i] = 1.0
                row[10+j] = -1.0
            elif ord_val == 3:  # j below i: y_j + s_j <= y_i
                row[10+j] = 1.0
                row[20+j] = 1.0
                row[10+i] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
        
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*30, method='highs')
            if res.success:
                x_vals = res.x[:n]
                y_vals = res.x[10:20]
                s_vals = res.x[20:30]
                
                # Convert to centers
                centers_lp = np.column_stack([x_vals + s_vals/2, y_vals + s_vals/2])
                sides_lp = s_vals
                
                if _is_valid(centers_lp, sides_lp):
                    val = np.sum(sides_lp)
                    if val > best_lp_val:
                        best_lp_val = val
                        best_lp = (centers_lp, sides_lp)
        except:
            pass
        
        return best_lp
    
    lp_result = _lp_init()
    if lp_result is not None:
        update_best(lp_result[0], lp_result[1])

    # Optimization
    np.random.seed(42)
    starts = [(c1.copy(), s1.copy()), (c2.copy(), s2.copy()), (c3.copy(), s3.copy())]
    
    # Add LP result if available
    if lp_result is not None:
        starts.append((lp_result[0].copy(), lp_result[1].copy()))
    
    # Add random perturbations from best config
    for _ in range(15):
        c_r = c1.copy() + np.random.normal(0, 0.05, (n, 2))
        s_r = s1.copy() + np.random.normal(0, 0.02, n)
        s_r = np.clip(s_r, 0.1, 0.6)
        starts.append((c_r, s_r))
    
    # Add perturbations from LP result
    if lp_result is not None:
        for _ in range(10):
            c_r = lp_result[0].copy() + np.random.normal(0, 0.03, (n, 2))
            s_r = lp_result[1].copy() + np.random.normal(0, 0.02, n)
            s_r = np.clip(s_r, 0.1, 0.6)
            starts.append((c_r, s_r))

    for c_init, s_init in starts:
        if time.time() > deadline:
            break
        params_init = np.concatenate([c_init.flatten(), s_init])
        
        try:
            # Use Powell for better handling of non-smooth objectives
            res = minimize(_objective, params_init, args=(n,), method='Powell',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            params_opt = res.x
            c_opt = params_opt[:2*n].reshape(n, 2)
            s_opt = np.maximum(params_opt[2*n:], 1e-6)
            update_best(c_opt, s_opt)
            
            # Polish with COBYLA if valid
            if _is_valid(c_opt, s_opt) and time.time() < deadline - 30:
                res_polish = minimize(_objective, params_opt, args=(n,), method='COBYLA',
                                      options={'maxiter': 300, 'rhobeg': 1e-6})
                params_pol = res_polish.x
                c_pol = params_pol[:2*n].reshape(n, 2)
                s_pol = np.maximum(params_pol[2*n:], 1e-6)
                update_best(c_pol, s_pol)
                
        except Exception:
            pass

    if best_centers is None:
        best_centers = c1
        best_sides = s1

    # Final high-precision polish on the best result
    if time.time() < deadline - 60:
        try:
            params_best = np.concatenate([best_centers.flatten(), best_sides])
            res_final = minimize(_objective, params_best, args=(n,), method='Powell',
                                 options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            params_final = res_final.x
            c_final = params_final[:2*n].reshape(n, 2)
            s_final = np.maximum(params_final[2*n:], 1e-6)
            update_best(c_final, s_final)
        except:
            pass

    return best_centers, best_sides

# EVOLVE-BLOCK-END
