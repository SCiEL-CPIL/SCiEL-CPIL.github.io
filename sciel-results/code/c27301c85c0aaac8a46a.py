import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, linprog

N = 12

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side length, and rotation."""
    angle = np.radians(angle_deg)
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _min_enclosing_hexagon(vertices_all):
    """
    Find the minimum side length of a regular hexagon containing all vertices.
    Uses LP for each rotation to find optimal center and radius.
    """
    points = np.vstack(vertices_all)
    n_pts = len(points)
    
    if n_pts == 0:
        return (0.0, 0.0, 0.0, 0.0)
        
    # Pre-compute point coordinates
    px = points[:, 0]
    py = points[:, 1]
    
    def get_side_and_center(theta):
        """
        For a fixed rotation theta, find the minimal enclosing hexagon side length
        and center using Linear Programming.
        """
        normals = np.array([[np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)] for k in range(6)])
        
        # Projections of points onto normals
        # proj[i, k] = dot(point_i, normal_k)
        proj = points @ normals.T
        
        # We want to find c = (cx, cy) and R such that:
        # For all i, k: |proj[i, k] - (cx * n_kx + cy * n_ky)| <= R
        # This is equivalent to:
        # proj[i, k] - cx * n_kx - cy * n_ky <= R
        # -proj[i, k] + cx * n_kx + cy * n_ky <= R
        
        # Variables: x = [cx, cy, R]
        # Minimize R -> c = [0, 0, 1]
        
        # Constraints: A_ub @ x <= b_ub
        # For each i, k:
        # [ -n_kx, -n_ky, 1 ] @ [cx, cy, R] <= proj[i, k]
        # [  n_kx,  n_ky, 1 ] @ [cx, cy, R] <= -proj[i, k]
        
        A_ub = []
        b_ub = []
        
        for k in range(6):
            nkx, nky = normals[k]
            # Constraint 1: proj - dot(c, n) <= R  => -dot(c, n) - R <= -proj? 
            # Wait: proj - cx*nkx - cy*nky <= R
            # -cx*nkx - cy*nky - R <= -proj
            # Coeffs: [-nkx, -nky, -1] ... wait, we want to minimize R.
            # Standard form: A_ub @ x <= b_ub
            # Inequality: cx*nkx + cy*nky + R >= proj  => -cx*nkx - cy*nky - R <= -proj
            # Inequality: cx*nkx + cy*nky - R <= -proj  => cx*nkx + cy*nky - R <= -proj
            
            # Let's re-verify:
            # |proj - dot(c,n)| <= R
            # proj - dot(c,n) <= R  => -dot(c,n) - R <= -proj
            # dot(c,n) - proj <= R  => dot(c,n) - R <= proj
            
            # Row 1: [-nkx, -nky, -1] @ x <= -proj[i,k]
            # Row 2: [ nkx,  nky, -1] @ x <=  proj[i,k]
            
            for i in range(n_pts):
                A_ub.append([-nkx, -nky, -1])
                b_ub.append(-proj[i, k])
                A_ub.append([nkx, nky, -1])
                b_ub.append(proj[i, k])
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: R >= 0
        bounds = [(None, None), (None, None), (0, None)]
        
        c_obj = [0, 0, 1]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                cx, cy, R = res.x
                side = 2 * R / np.sqrt(3)
                return side, cx, cy
            else:
                # Fallback to midpoint method if LP fails
                midpoints = (proj.max(axis=0) + proj.min(axis=0)) / 2
                c, _, _, _ = np.linalg.lstsq(normals, midpoints, rcond=None)
                widths = proj.max(axis=0) - proj.min(axis=0)
                side = widths.max() / np.sqrt(3)
                return side, c[0], c[1]
        except Exception:
            # Fallback
            midpoints = (proj.max(axis=0) + proj.min(axis=0)) / 2
            c, _, _, _ = np.linalg.lstsq(normals, midpoints, rcond=None)
            widths = proj.max(axis=0) - proj.min(axis=0)
            side = widths.max() / np.sqrt(3)
            return side, c[0], c[1]

    def objective(theta):
        side, _, _ = get_side_and_center(theta)
        return side

    # Optimize theta
    # The function is periodic with period pi/3.
    # We search in [0, pi/3].
    
    # Coarse grid search to find good starting points
    thetas = np.linspace(0, np.pi/3, 20)
    best_side = float('inf')
    best_theta = 0
    best_cx = 0
    best_cy = 0
    
    for t in thetas:
        s, cx, cy = get_side_and_center(t)
        if s < best_side:
            best_side = s
            best_theta = t
            best_cx = cx
            best_cy = cy
            
    # Local refinement around the best coarse point
    # Nelder-Mead
    res = minimize(objective, best_theta, method='Nelder-Mead', 
                   options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
    
    if res.success and res.fun < best_side:
        best_side = res.fun
        best_theta = res.x[0]
        _, best_cx, best_cy = get_side_and_center(best_theta)
        
    # Add a tiny safety margin to side length to ensure containment
    # due to floating point precision in the evaluator
    best_side *= 1.0001
    
    return (best_cx, best_cy, best_side, np.degrees(best_theta))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    sqrt3 = np.sqrt(3)
    
    inner_positions = []
    for row in range(4):
        for col in range(3):
            x = col * sqrt3 + (row % 2) * (sqrt3 / 2)
            y = row * 1.5
            inner_positions.append([x, y, 0.0])
    
    # Compute vertices for all inner hexagons
    all_vertices = []
    for x, y, ang in inner_positions:
        verts = _hexagon_vertices(x, y, 1.0, ang)
        all_vertices.append(verts)
    
    # Find the minimal enclosing hexagon
    cx, cy, side, theta = _min_enclosing_hexagon(all_vertices)
    
    # Check time
    if time.time() > deadline:
        # Return best found so far (which is the only one we computed)
        pass
        
    return inner_positions, (cx, cy), side, theta
# EVOLVE-BLOCK-END
