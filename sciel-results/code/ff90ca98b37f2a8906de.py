import numpy as np
import time
from scipy.optimize import minimize


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


def _make_2big_8small():
    """Construct 2 large (0.5) + 8 small (0.25) tiling. Sum = 3.0."""
    centers = np.array([
        # 2 large squares at top
        [0.25, 0.75],  # large 1: [0,0.5]x[0.5,1]
        [0.75, 0.75],  # large 2: [0.5,1]x[0.5,1]
        # 8 small squares in 2x4 grid below
        [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
        [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375]
    ])
    sides = np.array([0.5, 0.5] + [0.25]*8)
    return centers, sides


def _local_refine(centers, sides, tol=1e-7, max_iter=2000):
    """
    Local refinement: try to grow squares while maintaining feasibility.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    deltas = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            for delta in deltas:
                new_sides = best_sides.copy()
                new_sides[i] = best_sides[i] + delta
                new_centers = best_centers.copy()
                
                s = new_sides[i]
                cx = np.clip(best_centers[i, 0], s/2, 1 - s/2)
                cy = np.clip(best_centers[i, 1], s/2, 1 - s/2)
                new_centers[i] = [cx, cy]
                
                if _is_feasible(new_centers, new_sides, tol):
                    best_centers = new_centers
                    best_sides = new_sides
                    best_val = np.sum(best_sides)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_centers, best_sides


def _smooth_refine(centers, sides, tol=1e-7, deadline=None):
    """
    Smooth SLSQP-based refinement using a soft penalty surrogate.
    Variables: [cx_0..cx_9, cy_0..cy_9, s_0..s_9] (30 vars)
    Objective: maximize sum(s_i)
    Constraints (smooth):
      - boundary: s_i/2 <= cx_i <= 1 - s_i/2, same for cy
      - non-overlap (soft): for each pair (i,j),
        |cx_i - cx_j| + |cy_i - cy_j| >= (s_i + s_j)/2 - tol
        approximated with soft-abs via log-sum-exp
    """
    n = len(sides)
    x0 = np.concatenate([centers.flatten(), sides])
    
    # Soft-abs parameter for smoothness
    k = 50.0  # larger = sharper, more accurate
    
    def soft_abs(a, k=k):
        """Smooth approximation of |a| using log-sum-exp."""
        return (a + np.logaddexp(0, -k*a)) / k
    
    def objective(x):
        """Minimize negative sum of sides."""
        s = x[2*n:]
        return -np.sum(s)
    
    def constraint_overlap(i, j):
        """Return constraint function for pair (i,j): |dx|+|dy| - (s_i+s_j)/2 + tol >= 0"""
        def c(x):
            cx_i, cy_i = x[i], x[n+i]
            cx_j, cy_j = x[j], x[n+j]
            s_i, s_j = x[2*n+i], x[2*n+j]
            dx = cx_i - cx_j
            dy = cy_i - cy_j
            # Use soft-abs for smoothness
            sadx = soft_abs(dx)
            sady = soft_abs(dy)
            return sadx + sady - (s_i + s_j)/2.0 + tol
        return c
    
    def constraint_boundary_x(i):
        def c_lower(x):
            s_i = x[2*n+i]
            return x[i] - s_i/2.0
        def c_upper(x):
            s_i = x[2*n+i]
            return 1.0 - x[i] - s_i/2.0
        return c_lower, c_upper
    
    def constraint_boundary_y(i):
        def c_lower(x):
            s_i = x[2*n+i]
            return x[n+i] - s_i/2.0
        def c_upper(x):
            s_i = x[2*n+i]
            return 1.0 - x[n+i] - s_i/2.0
        return c_lower, c_upper
    
    # Build constraints
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        cx_lo, cx_hi = constraint_boundary_x(i)
        cy_lo, cy_hi = constraint_boundary_y(i)
        constraints.append({'type': 'ineq', 'fun': cx_lo})
        constraints.append({'type': 'ineq', 'fun': cx_hi})
        constraints.append({'type': 'ineq', 'fun': cy_lo})
        constraints.append({'type': 'ineq', 'fun': cy_hi})
    
    # Overlap constraints (soft)
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': constraint_overlap(i, j)})
    
    # Side positivity: s_i > 0, enforced via lower bound
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1.0))  # cx_i
        bounds.append((1e-6, 1.0))  # cy_i
        bounds.append((1e-6, 1.0))  # s_i
    
    # Run SLSQP
    result = minimize(
        objective, x0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
    )
    
    if result.success and np.all(np.isfinite(result.x)):
        new_centers = result.x[:2*n].reshape(n, 2)
        new_sides = result.x[2*n:]
        # Clip to valid range
        for i in range(n):
            new_sides[i] = max(1e-6, new_sides[i])
            new_centers[i, 0] = np.clip(new_centers[i, 0], new_sides[i]/2, 1 - new_sides[i]/2)
            new_centers[i, 1] = np.clip(new_centers[i, 1], new_sides[i]/2, 1 - new_sides[i]/2)
        
        if _is_feasible(new_centers, new_sides, tol):
            return new_centers, new_sides
    
    return centers, sides


def _high_precision_polish(centers, sides, tol=1e-7, deadline=None):
    """
    Final high-precision polish: run SLSQP with tighter settings from the best found.
    """
    try:
        c, s = _smooth_refine(centers, sides, tol=tol, deadline=deadline)
        return c, s
    except Exception:
        return centers, sides


def squares_in_square10():
    start_time = time.time()
    deadline = start_time + 580
    
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    # Primary packing: 2 large (0.5) + 8 small (0.25), sum = 3
    c1, s1 = _make_2big_8small()
    if _is_feasible(c1, s1):
        best_centers = c1.copy()
        best_sides = s1.copy()
        best_val = np.sum(s1)
    
    # Local refinement (discrete)
    if best_centers is not None and time.time() < deadline - 60:
        try:
            refined_centers, refined_sides = _local_refine(
                best_centers, best_sides, max_iter=2000
            )
            if _is_feasible(refined_centers, refined_sides):
                refined_val = np.sum(refined_sides)
                if refined_val > best_val:
                    best_val = refined_val
                    best_centers = refined_centers
                    best_sides = refined_sides
        except Exception:
            pass
    
    # Smooth SLSQP refinement
    if best_centers is not None and time.time() < deadline - 30:
        try:
            smooth_centers, smooth_sides = _smooth_refine(
                best_centers, best_sides, tol=1e-7, deadline=deadline
            )
            if _is_feasible(smooth_centers, smooth_sides):
                smooth_val = np.sum(smooth_sides)
                if smooth_val > best_val:
                    best_val = smooth_val
                    best_centers = smooth_centers
                    best_sides = smooth_sides
        except Exception:
            pass
    
    # High-precision polish from the best found
    if best_centers is not None and time.time() < deadline - 15:
        try:
            polish_centers, polish_sides = _high_precision_polish(
                best_centers, best_sides, tol=1e-7, deadline=deadline
            )
            if _is_feasible(polish_centers, polish_sides):
                polish_val = np.sum(polish_sides)
                if polish_val > best_val:
                    best_val = polish_val
                    best_centers = polish_centers
                    best_sides = polish_sides
        except Exception:
            pass
    
    if best_centers is None:
        # Fallback: should not happen
        best_centers, best_sides = _make_2big_8small()
    
    return best_centers, best_sides
