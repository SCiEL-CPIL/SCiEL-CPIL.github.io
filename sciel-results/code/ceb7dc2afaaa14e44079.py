# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Pack 10 axis-aligned squares in [0,1]^2 maximizing sum of side lengths.
    
    Strategy:
    1. Direct continuous optimization using SLSQP.
    2. Objective: -sum(sides) + penalty * (sum of overlap penalties + boundary penalties).
    3. Overlap penalty uses a smooth approximation (soft-max) of the maximum violation.
    4. Multiple random restarts to escape local minima.
    """
    t0 = time.time()
    deadline = t0 + 580.0  # safety margin before 630s kill

    n = 10
    tol = 1e-7
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    n_pairs = len(pairs)

    def is_valid(centers, sides):
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # boundary
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        # overlap
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def compute_penalty(x, beta=100.0):
        """
        x: [s_0, ..., s_9, cx_0, ..., cx_9, cy_0, ..., cy_9]
        Returns a scalar penalty value.
        """
        s = x[:n]
        cx = x[n:2*n]
        cy = x[2*n:3*n]
        
        penalty = 0.0
        
        # Boundary penalties:
        # We want s/2 <= cx <= 1 - s/2
        # Violation if cx < s/2 or cx > 1 - s/2
        # Smooth penalty: max(0, s/2 - cx)^2 + max(0, cx - (1-s/2))^2
        # Using soft max for differentiability:
        # Actually, simple hinge is fine for SLSQP if we use it in objective, 
        # but SLSQP prefers smoothness. Let's use quadratic penalty for boundary violations.
        
        # x boundaries
        x_low_viol = np.maximum(0, s/2 - cx)
        x_high_viol = np.maximum(0, cx - (1 - s/2))
        penalty += np.sum(x_low_viol**2) + np.sum(x_high_viol**2)
        
        # y boundaries
        y_low_viol = np.maximum(0, s/2 - cy)
        y_high_viol = np.maximum(0, cy - (1 - s/2))
        penalty += np.sum(y_low_viol**2) + np.sum(y_high_viol**2)
        
        # Overlap penalties
        # For each pair (i, j):
        # dx = |cx_i - cx_j|, dy = |cy_i - cy_j|
        # required_dx = (s_i + s_j)/2
        # required_dy = (s_i + s_j)/2
        # Violation if dx < required_dx AND dy < required_dy
        # We can penalize the "depth" of overlap.
        # A common trick: penalty = max(0, required_dx - dx)^2 + max(0, required_dy - dy)^2
        # But this penalizes even if they are separated in one dimension.
        # Correct logic: If they overlap, it's because BOTH dx < req_dx AND dy < req_dy.
        # If dx >= req_dx, no penalty. If dy >= req_dy, no penalty.
        # So penalty is non-zero only when both are violated.
        # We can use: penalty = max(0, req_dx - dx) * max(0, req_dy - dy) * 2 ?
        # Or sum of squares of violations only if both are violated.
        # Let's use a smooth approximation of the indicator function.
        # Indicator(dx < req_dx and dy < req_dy) ~ 1 / (1 + exp(beta * (dx - req_dx))) * 1 / (1 + exp(beta * (dy - req_dy)))
        # This is expensive.
        
        # Alternative: Penalize the "overlap area" or "minimum separation".
        # Let's stick to the standard approach for SLSQP: 
        # Instead of a penalty in the objective, we can use constraints.
        # But the prompt asks for a DIFFERENT strategy. The current one uses constraints.
        # Let's try a penalty method with a strong weight.
        
        # Better penalty:
        # For each pair, calculate the gap in x and y.
        # gap_x = dx - req_dx
        # gap_y = dy - req_dy
        # If gap_x >= 0 or gap_y >= 0, they are separated. Penalty 0.
        # If gap_x < 0 and gap_y < 0, they overlap.
        # Penalty should be proportional to how much they overlap.
        # Let's use: P = max(0, -gap_x)^2 + max(0, -gap_y)^2 is wrong because it penalizes separation in one axis if the other is overlapping? No, max(0, -gap) is 0 if gap >= 0.
        # So if gap_x >= 0, first term is 0. If gap_y < 0, second term is > 0.
        # This would penalize a configuration where they are separated in X but overlapping in Y (which is valid!).
        # So we need a term that is non-zero ONLY if BOTH are negative.
        
        # Let u = max(0, -gap_x) and v = max(0, -gap_y).
        # We want penalty ~ u * v.
        # This is not smooth at 0, but SLSQP can handle it often.
        # To make it smooth: u * v is fine.
        
        for i, j in pairs:
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            req = (s[i] + s[j]) / 2
            gap_x = dx - req
            gap_y = dy - req
            
            # u = max(0, -gap_x), v = max(0, -gap_y)
            u = max(0.0, -gap_x)
            v = max(0.0, -gap_y)
            
            # Penalty is u * v. This is 0 if either gap is positive.
            # It is positive if both are negative (overlap).
            # Magnitude: if they are deeply overlapping, u and v are large, product is large.
            penalty += u * v
            
        return penalty

    def objective(x):
        s = x[:n]
        # Ensure sides are positive in objective? Bounds handle it.
        return -np.sum(s) + 100.0 * compute_penalty(x)

    def objective_grad(x):
        # Numerical gradient or analytical?
        # SLSQP can use numerical if not provided, but analytical is faster.
        # Let's just let SLSQP use numerical to save coding effort, 
        # or implement a simple finite difference if needed.
        # Actually, providing gradient is crucial for SLSQP speed.
        # But implementing the gradient of the max(0, ...) terms is tricky.
        # Let's rely on SLSQP's numerical differentiation first. If it's too slow, we'll adjust.
        return None # Force SLSQP to use numerical

    def solve_instance(x0, maxiter=200):
        bounds = [(0, 1)] * (3*n)
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            if res.success:
                val = -res.x[:n].sum() # Raw value, ignoring penalty in return
                s = res.x[:n]
                cx = res.x[n:2*n]
                cy = res.x[2*n:3*n]
                centers = np.column_stack([cx, cy])
                if is_valid(centers, s):
                    return val, res.x
            # Even if not "success", check if it's valid and good
            s = res.x[:n]
            cx = res.x[n:2*n]
            cy = res.x[2*n:3*n]
            centers = np.column_stack([cx, cy])
            if is_valid(centers, s):
                val = np.sum(s)
                return val, res.x
        except Exception:
            pass
        return None, None

    best_val = 0.0
    best_x = None

    # Initial guesses
    # 1. Grid 3x3 + 1
    # 2. Random
    # 3. Structured: 2 rows of 5?
    
    def make_grid_x(n, cols):
        # n items in 'cols' columns
        rows = int(np.ceil(n / cols))
        x0 = np.zeros(3*n)
        for i in range(n):
            r = i // cols
            c = i % cols
            s = 1.0 / cols
            x0[i] = s
            x0[n+i] = (c + 0.5) * s
            x0[2*n+i] = (r + 0.5) * (1.0/rows)
        return x0

    candidates = []
    
    # Candidate 1: 3 cols, 4 rows (approx)
    candidates.append(make_grid_x(n, 3))
    
    # Candidate 2: 4 cols, 3 rows
    candidates.append(make_grid_x(n, 4))
    
    # Candidate 3: Random
    np.random.seed(42)
    for _ in range(5):
        s = np.random.uniform(0.1, 0.4, n)
        cx = np.random.uniform(s/2, 1-s/2, n)
        cy = np.random.uniform(s/2, 1-s/2, n)
        x0 = np.concatenate([s, cx, cy])
        candidates.append(x0)
        
    # Candidate 4: All small in center
    s = np.full(n, 0.1)
    cx = np.full(n, 0.5)
    cy = np.full(n, 0.5)
    candidates.append(np.concatenate([s, cx, cy]))

    for x0 in candidates:
        if time.time() > deadline - 30:
            break
        val, x = solve_instance(x0, maxiter=100)
        if val is not None and val > best_val:
            best_val = val
            best_x = x

    # Refine best
    if best_x is not None:
        if time.time() < deadline - 20:
            val, x = solve_instance(best_x, maxiter=500)
            if val is not None and val > best_val:
                best_val = val
                best_x = x

    # Fallback
    if best_x is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
    else:
        s = best_x[:n]
        cx = best_x[n:2*n]
        cy = best_x[2*n:3*n]
        centers = np.column_stack([cx, cy]).astype(np.float64)
        sides = s.astype(np.float64)
        
        if not is_valid(centers, sides):
             k = 4
             side = 1.0 / k
             idx = np.arange(n)
             cols = idx % k
             rows = idx // k
             cx = (cols + 0.5) * side
             cy = (rows + 0.5) * side
             centers = np.stack([cx, cy], axis=1).astype(np.float64)
             sides = np.full(n, side, dtype=np.float64)

    return centers, sides

# EVOLVE-BLOCK-END
