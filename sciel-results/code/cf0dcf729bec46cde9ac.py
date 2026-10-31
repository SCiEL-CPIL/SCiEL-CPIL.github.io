"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = 3 ** 0.5

# Vertices of the unit equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])


def bary_to_cart(bary):
    """Convert barycentric coordinates (n, 3) to Cartesian (n, 2)."""
    # bary columns: (lambda0, lambda1, lambda2) with sum=1
    # cart = lambda0*V0 + lambda1*V1 + lambda2*V2
    # Since V0 is origin, cart = lambda1 * V1 + lambda2 * V2
    # V1 = (1,0), V2 = (0.5, sqrt(3)/2)
    # x = lambda1 + 0.5*lambda2
    # y = (sqrt(3)/2)*lambda2
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2.0) * bary[:, 2]
    return np.column_stack([x, y])


def min_triangle_area(pts):
    """
    Compute the minimum area of any triangle formed by 3 of the n points.
    pts: (n, 2) array of Cartesian coordinates.
    Returns the minimum area (not doubled area).
    Vectorized computation.
    """
    n = len(pts)
    # Compute all pairwise differences
    # For each pair (i, j), the area with a third point k is 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
    # We need min over all triples (i, j, k) with i < j < k
    
    # Efficient approach: for each pair (i,j), compute cross product with all other points
    # Actually, let's use a fully vectorized approach for small n
    
    # Generate all pairs of indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # For each pair (i, j), compute the vector from i to j
    v_ij = pts[j_idx] - pts[i_idx]  # (m, 2) where m = n*(n-1)/2
    
    # For each pair (i, j) and each point k, compute cross product
    # cross(v_ij, pts[k] - pts[i])
    # = v_ij[:,0] * (pts[k,1] - pts[i,1]) - v_ij[:,1] * (pts[k,0] - pts[i,0])
    
    # pts[i_idx] is (m, 2), pts[j_idx] is (m, 2)
    # We need for each pair index p and each point k:
    # cross = v_ij[p,0] * (pts[k,1] - pts[i_idx[p],1]) - v_ij[p,1] * (pts[k,0] - pts[i_idx[p],0])
    
    # This is: v_ij[:,0] * (pts[1] - pts[i_idx,1]) - v_ij[:,1] * (pts[0] - pts[i_idx,0])
    # where pts[1] and pts[0] are broadcast over the first axis
    
    # Let's compute:
    # diff_y = pts[:, 1:2] - pts[i_idx, 1:2]  # (m, n)
    # diff_x = pts[:, 0:1] - pts[i_idx, 0:1]  # (m, n)
    # cross = v_ij[:, 0:1] * diff_y - v_ij[:, 1:2] * diff_x  # (m, n)
    
    diff_y = pts[1:2, :] - pts[i_idx, 1:2]  # (m, n)
    diff_x = pts[0:1, :] - pts[i_idx, 0:1]  # (m, n)
    
    cross = v_ij[:, 0:1] * diff_y - v_ij[:, 1:2] * diff_x  # (m, n)
    
    # Area = 0.5 * |cross|
    # We want the minimum over all valid triples (i, j, k) where i < j < k
    # But our pairs are i < j, and k can be any point. We need k != i and k != j, and to avoid double counting
    # and ensure i < j < k, we can just take the min over all k != i and k != j, but that includes k < i and k > j.
    # Actually, for the minimum, it doesn't matter if we consider all k, because the area is symmetric.
    # We just need to exclude k = i and k = j.
    
    # Set cross to inf where k == i or k == j
    # k == i means column index equals i_idx[p]
    # k == j means column index equals j_idx[p]
    
    # Create a mask
    k_indices = np.arange(n)
    # For each pair p, k_indices == i_idx[p] or k_indices == j_idx[p]
    # We can do:
    # mask_i = (k_indices[None, :] == i_idx[:, None])  # (m, n)
    # mask_j = (k_indices[None, :] == j_idx[:, None])  # (m, n)
    # mask = mask_i | mask_j
    
    # But this is memory intensive for large n. For n=11, it's fine.
    
    mask_i = (np.arange(n)[None, :] == i_idx[:, None])  # (m, n)
    mask_j = (np.arange(n)[None, :] == j_idx[:, None])  # (m, n)
    mask = mask_i | mask_j
    
    cross_abs = np.abs(cross)
    cross_abs[mask] = np.inf
    
    # The minimum area is 0.5 * min(cross_abs)
    min_cross = np.min(cross_abs)
    return 0.5 * min_cross


def objective(params):
    """
    Objective function for optimization: minimize the negative of the min triangle area,
    so we maximize the min triangle area.
    params: (n, 3) barycentric coordinates (each row sums to 1).
    """
    # Convert to Cartesian
    pts = bary_to_cart(params)
    
    # Check if all points are inside the triangle
    # Barycentric coords should all be >= 0 and sum to 1
    # Since we're using optimization, they might go slightly negative
    # Clip to valid range
    if np.any(params < -1e-10):
        return 1e10  # Large penalty
    
    # Compute min triangle area
    area = min_triangle_area(pts)
    return -area


def random_bary_init(rng, n=N):
    """Generate random barycentric coordinates for n points."""
    # Generate random points in the triangle using rejection sampling in barycentric coords
    bary = np.zeros((n, 3))
    for _ in range(n):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                bary[_] = [1.0 - u - v, u, v]
                break
    return bary


def coordinate_descent(pts, steps_per_dim=100, n_steps=50):
    """
    Simple coordinate descent to fine-tune the points.
    pts: (n, 2) array
    Returns improved pts.
    """
    current_area = min_triangle_area(pts)
    
    for step in range(n_steps):
        # Decrease step size
        step_size = 0.1 / (step + 1)
        
        improved = False
        for i in range(len(pts)):
            for d in range(2):
                # Try moving point i in dimension d
                for sign in [1, -1]:
                    new_pts = pts.copy()
                    new_pts[i, d] += sign * step_size
                    
                    # Check if new point is inside the triangle
                    # Convert to barycentric: 
                    # x = b1 + 0.5*b2, y = (sqrt(3)/2)*b2
                    # b2 = 2*y/sqrt(3)
                    # b1 = x - 0.5*b2
                    # b0 = 1 - b1 - b2
                    x, y = new_pts[i]
                    b2 = 2.0 * y / SQRT3
                    b1 = x - 0.5 * b2
                    b0 = 1.0 - b1 - b2
                    
                    if b0 >= -1e-10 and b1 >= -1e-10 and b2 >= -1e-10:
                        new_area = min_triangle_area(new_pts)
                        if new_area > current_area:
                            pts = new_pts
                            current_area = new_area
                            improved = True
        
        if not improved:
            break
    
    return pts, current_area


def construct():
    """
    Place 11 points on or inside a unit-area triangle and maximize the smallest area
    of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = 140.0  # 10 second safety margin
    
    import random
    rng = random.Random(42)
    
    best_pts = None
    best_area = -np.inf
    
    # Multiple restarts with different random inits
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial points
        bary = random_bary_init(rng, N)
        
        # Optimize using L-BFGS-B
        # We need to handle the constraint that barycentric coords sum to 1
        # and are non-negative.
        # We can use 2 free parameters per point (the third is 1 - sum of the other two)
        # Or we can use the fact that scipy's L-BFGS-B can handle bounds
        
        # Let's use a different parameterization: use 2D Cartesian coordinates
        # and check if they're inside the triangle
        
        pts_init = bary_to_cart(bary)
        
        def obj_cart(params):
            """Objective for Cartesian coordinates."""
            pts = params.reshape(-1, 2)
            
            # Check if all points are inside the triangle
            # For each point (x, y):
            # b2 = 2*y/sqrt(3)
            # b1 = x - 0.5*b2
            # b0 = 1 - b1 - b2
            # All must be >= 0
            y = pts[:, 1]
            x = pts[:, 0]
            b2 = 2.0 * y / SQRT3
            b1 = x - 0.5 * b2
            b0 = 1.0 - b1 - b2
            
            if np.any(b0 < -1e-8) or np.any(b1 < -1e-8) or np.any(b2 < -1e-8):
                return 1e10
            
            area = min_triangle_area(pts)
            return -area
        
        # Bounds for Cartesian: x in [0, 1], y in [0, sqrt(3)/2]
        # But this is a rectangle, not the triangle. We need tighter bounds.
        # For a point (x, y) to be in the triangle:
        # y >= 0
        # y <= sqrt(3)*x (left edge from (0,0) to (0.5, sqrt(3)/2))
        # y <= sqrt(3)*(1-x) (right edge from (1,0) to (0.5, sqrt(3)/2))
        
        # Let's use a different approach: parameterize with barycentric coords
        # using 2 free variables per point
        
        def obj_bary(params):
            """Objective for barycentric coordinates with 2 free vars per point."""
            bary = np.zeros((N, 3))
            bary[:, 0] = 1.0
            bary[:, 1] = params[0:N]
            bary[:, 2] = params[N:2*N]
            
            # Check if valid
            if np.any(bary < -1e-10):
                return 1e10
            
            pts = bary_to_cart(bary)
            area = min_triangle_area(pts)
            return -area
        
        # Bounds: b1 >= 0, b2 >= 0, b1 + b2 <= 1
        # So b1 in [0, 1], b2 in [0, 1]
        bounds = [(0, 1)] * (2 * N)
        
        x0 = np.concatenate([bary[:, 1], bary[:, 2]])
        
        try:
            result = minimize(obj_bary, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-12})
            
            if result.success or result.fun < 1e9:
                # Convert back to Cartesian
                bary_opt = np.zeros((N, 3))
                bary_opt[:, 0] = 1.0
                bary_opt[:, 1] = result.x[0:N]
                bary_opt[:, 2] = result.x[N:2*N]
                
                pts_opt = bary_to_cart(bary_opt)
                
                # Verify all points are inside
                y = pts_opt[:, 1]
                x = pts_opt[:, 0]
                b2 = 2.0 * y / SQRT3
                b1 = x - 0.5 * b2
                b0 = 1.0 - b1 - b2
                
                if np.all(b0 >= -1e-8) and np.all(b1 >= -1e-8) and np.all(b2 >= -1e-8):
                    area = min_triangle_area(pts_opt)
                    
                    # Coordinate descent for fine-tuning
                    pts_cd, area_cd = coordinate_descent(pts_opt, n_steps=30)
                    
                    if area_cd > area:
                        pts_opt = pts_cd
                        area = area_cd
                    
                    if area > best_area:
                        best_area = area
                        best_pts = pts_opt
        
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback to random
        bary = random_bary_init(rng, N)
        best_pts = bary_to_cart(bary)
    
    return best_pts
# EVOLVE-BLOCK-END
