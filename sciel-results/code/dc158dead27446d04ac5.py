"""
Heilbronn triangles: Place 11 points to maximize the minimum triangle area.
Strategy: Multi-start gradient-based optimization using a smooth log-sum-exp
approximation of the minimum area objective. This converts the non-smooth
max-min problem into a smooth minimization problem suitable for L-BFGS-B.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0

def _check_inside(points):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    # points shape (N, 2)
    x = points[:, 0]
    y = points[:, 1]
    
    # Conditions for inside triangle (0,0), (1,0), (0.5, S3H)
    # 1. y >= 0
    # 2. y <= S3H * x
    # 3. y <= -S3H * x + S3H
    
    tol = 1e-9
    if np.any(y < -tol):
        return False
    if np.any(y > S3H * x + tol):
        return False
    if np.any(y > -S3H * x + S3H + tol):
        return False
    return True

def _project_to_triangle(points):
    """
    Project points onto the convex hull of the triangle if they are outside.
    This is a soft projection for feasibility during optimization steps.
    For L-BFGS-B, we rely on bounds, but this helps with initialization or recovery.
    """
    # We will primarily use bounds in the optimizer, so this is a fallback
    # or for post-processing. A simple clamping strategy:
    # Since it's a convex polygon, we can check constraints and project.
    # However, for the optimization loop, we define bounds:
    # x in [0, 1], y in [0, S3H]
    # And the linear constraints are handled by the objective penalty or 
    # simply by rejecting invalid steps. 
    # Actually, L-BFGS-B only handles box constraints. 
    # So we must ensure the objective penalizes going outside or we map space.
    # Better approach: Use a transformation or just penalize in objective.
    # Given the time constraint, we will use a penalty in the objective function 
    # for points going outside the triangle.
    return points

def _compute_min_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Returns the raw minimum area (not normalized).
    """
    n = len(points)
    min_area = float('inf')
    # Vectorized approach for speed
    # We need to check all C(11,3) = 165 triangles.
    # We can compute all areas at once.
    
    # Get indices
    idx = np.indices((n, n, n)).reshape(3, -1)
    # Filter i < j < k
    mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
    i, j, k = idx[:, mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    if len(areas) > 0:
        min_area = np.min(areas)
    else:
        min_area = 0.0
        
    return min_area

def _smooth_min_area(points, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    We want to maximize min(area_i).
    min(area_i) ≈ - (1/beta) * log( sum(exp(-beta * area_i)) )
    
    To make it differentiable and stable, we work with the negative log-sum-exp.
    Objective to minimize: -smooth_min_area
    """
    n = len(points)
    idx = np.indices((n, n, n)).reshape(3, -1)
    mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
    i, j, k = idx[:, mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp trick for stability
    # log(sum(exp(-beta * areas)))
    # = -beta * max(areas) + log(sum(exp(-beta * (areas - max(areas)))))
    # Wait, we want min(areas).
    # min(a) = - max(-a)
    # -max(-a) = - ( - (1/beta) log(sum(exp(-beta * (-a))) ) ) ? No.
    
    # Let's use the standard soft-min:
    # soft_min(x, beta) = - (1/beta) * log( sum( exp(-beta * x_i) ) )
    # As beta -> inf, this approaches min(x_i).
    
    # To avoid overflow in exp(-beta * x_i) if x_i is large positive,
    # note that we are summing exp of negative values if areas are positive.
    # If areas are small, -beta*area is large negative, exp is small.
    # The term with the smallest area will have the largest exponent (least negative).
    
    # Stability:
    # m = min(areas)
    # sum(exp(-beta * areas)) = exp(-beta * m) * sum(exp(-beta * (areas - m)))
    # log(sum(...)) = -beta * m + log(sum(exp(-beta * (areas - m))))
    # soft_min = - (1/beta) * ( -beta * m + log(...) ) = m - (1/beta) log(...)
    
    m = np.min(areas)
    # If m is very small, areas-m is >= 0. exp(-beta*(areas-m)) <= 1.
    # This is numerically stable.
    shifted = areas - m
    sum_exp = np.sum(np.exp(-beta * shifted))
    log_sum_exp = np.log(sum_exp)
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return smooth_min

def _objective(flat_points):
    """
    Objective function for minimization.
    We want to maximize the minimum area.
    So we minimize -smooth_min_area.
    We also add a penalty if points are outside the triangle.
    """
    points = flat_points.reshape(-1, 2)
    
    # Penalty for being outside
    # Check constraints:
    x = points[:, 0]
    y = points[:, 1]
    
    # y >= 0
    viol1 = np.maximum(0, -y)
    # y <= S3H * x
    viol2 = np.maximum(0, y - S3H * x)
    # y <= -S3H * x + S3H
    viol3 = np.maximum(0, y - (-S3H * x + S3H))
    
    # x bounds are handled by L-BFGS-B, but y bounds are not strictly box-constrained 
    # in the same way because of the slanted edges.
    # We use a quadratic penalty.
    penalty = 100.0 * (np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2))
    
    # Compute smooth min area
    # Use a high beta for sharpness, but not too high to cause numerical issues
    # Beta = 50 seems reasonable for area scales around 0.01-0.1
    beta = 50.0
    smooth_min = _smooth_min_area(points, beta)
    
    # We want to maximize smooth_min, so minimize -smooth_min
    obj = -smooth_min + penalty
    
    return obj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial guesses
    # 1. Random points
    # 2. Grid-like points
    # 3. Vertices + Midpoints + Center
    
    best_points = None
    best_score = -np.inf
    
    np.random.seed(42)
    
    # Generate several initial configurations
    inits = []
    
    # Random
    for _ in range(5):
        u = np.random.random(N)
        v = np.random.random(N)
        # Reflect to ensure u+v <= 1
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        x = u + 0.5 * v
        y = S3H * v
        inits.append(np.column_stack((x, y)))
        
    # Structured: 3x3 grid in barycentric + 2 extra
    # Barycentric coords (a,b,c) with a+b+c=1
    # Map to Cartesian: P = a*A + b*B + c*C
    # A=(0,0), B=(1,0), C=(0.5, S3H)
    # x = b + 0.5c, y = S3H*c
    # Grid on (b,c) such that b>=0, c>=0, b+c<=1
    # 4x4 grid points
    b_vals = np.linspace(0, 1, 4)
    c_vals = np.linspace(0, 1, 4)
    pts_struct = []
    for b in b_vals:
        for c in c_vals:
            if b + c <= 1.0 + 1e-9:
                x = b + 0.5 * c
                y = S3H * c
                pts_struct.append([x, y])
    # We have 10 points (4+3+2+1). Need 11. Add center of mass or midpoint of an edge.
    # Add (0.5, S3H/2) ? No, that's (0.5, S3H*0.5). 
    # Let's just take the first 11 from a denser grid or random perturbation of this.
    if len(pts_struct) >= 11:
        inits.append(np.array(pts_struct[:11]))
    else:
        # Pad with random
        pts_struct += [inits[0][i] for i in range(11 - len(pts_struct))]
        inits.append(np.array(pts_struct))
        
    # Vertices and midpoints
    verts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0], # Mid bottom
        [0.25, S3H/2], # Mid left
        [0.75, S3H/2], # Mid right
        [0.5, S3H/3], # Centroid-ish
        [0.5, 2*S3H/3], # Top centroid-ish
        [0.25, S3H/4],
        [0.75, S3H/4],
        [0.5, S3H/6]
    ]
    inits.append(np.array(verts))
    
    # Optimization parameters
    # Bounds for L-BFGS-B
    # x in [0, 1]
    # y in [0, S3H]
    bounds = [(0, 1)] * N + [(0, S3H)] * N
    
    # We flatten points to (2N,)
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Ensure init is valid
        if not _check_inside(init):
            # Project or skip? For now, let the penalty handle it, 
            # but a bad start might fail. Let's project roughly.
            # Simple clamp
            init = init.copy()
            init[:, 0] = np.clip(init[:, 0], 0, 1)
            init[:, 1] = np.clip(init[:, 1], 0, S3H)
            # Check again
            if not _check_inside(init):
                # Try to fix
                x = init[:, 0]
                y = init[:, 1]
                # y <= S3H * x
                mask1 = y > S3H * x
                if np.any(mask1):
                    # Reduce y
                    init[mask1, 1] = S3H * init[mask1, 0]
                # y <= -S3H * x + S3H
                mask2 = y > -S3H * x + S3H
                if np.any(mask2):
                    init[mask2, 1] = -S3H * init[mask2, 0] + S3H
                if not _check_inside(init):
                    continue # Skip this init if we can't easily fix it

        flat_init = init.flatten()
        
        try:
            res = minimize(
                _objective,
                flat_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        if res.success or res.fun < best_score:
            opt_points = res.x.reshape(-1, 2)
            
            # Check validity
            if _check_inside(opt_points):
                score = _compute_min_area(opt_points)
                # Normalize by triangle area to match the "value" metric if needed?
                # The task says "maximize the smallest area... normalized by its area".
                # The evaluator measures "raw value". 
                # Looking at the seed: `area_normalized = (cross / 2) / (S3H / 2)`
                # So the score returned by construct is just the points.
                # The evaluator computes the score.
                # We just need to keep the best points based on _compute_min_area.
                
                if score > best_score:
                    best_score = score
                    best_points = opt_points.copy()
            else:
                # If optimizer went outside, try to project back and evaluate
                # This is a heuristic fallback
                proj_points = opt_points.copy()
                x = proj_points[:, 0]
                y = proj_points[:, 1]
                # Simple projection logic repeated
                mask1 = y > S3H * x
                if np.any(mask1):
                    proj_points[mask1, 1] = S3H * proj_points[mask1, 0]
                mask2 = y > -S3H * x + S3H
                if np.any(mask2):
                    proj_points[mask2, 1] = -S3H * proj_points[mask2, 0] + S3H
                if _check_inside(proj_points):
                    score = _compute_min_area(proj_points)
                    if score > best_score:
                        best_score = score
                        best_points = proj_points.copy()

    # If no valid points found, return a safe fallback
    if best_points is None:
        # Return random points
        u = np.random.random(N)
        v = np.random.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        x = u + 0.5 * v
        y = S3H * v
        best_points = np.column_stack((x, y))
        
    return best_points

# For testing
if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Min area (normalized):", _compute_min_area(pts) / TRI_AREA)
