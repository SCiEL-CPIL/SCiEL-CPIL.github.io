import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Constants
    N = 11
    sqrt3 = np.sqrt(3)
    
    # Vertices of a unit hexagon centered at origin, angle 0
    # (1, 0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1, 0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
    base_vertices = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2.0],
        [-0.5, sqrt3/2.0],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2.0],
        [0.5, -sqrt3/2.0]
    ])
    
    # Normals of the outer hexagon (directions of edges)
    # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees?
    # Vertices of outer hexagon (side S) are at angles 0, 60, ...
    # Edge between vertex 0 and 1 is perpendicular to angle 30.
    # So normals are at 30, 90, 150, 210, 270, 330.
    # Let's use 0, 60, 120, 180, 240, 300 for simplicity?
    # If outer hexagon is rotated by alpha, normals are at alpha + 30, alpha + 90, ...
    # Let's define normals relative to alpha.
    # n_k(alpha) = (cos(alpha + pi/6 + k*pi/3), sin(alpha + pi/6 + k*pi/3))
    
    def get_outer_normals(alpha):
        angles = alpha + np.pi/6 + np.arange(6) * np.pi/3
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    def calculate_required_side(centers, alpha):
        """
        Calculate the minimum side length S of an outer hexagon centered at origin, 
        rotated by alpha, that contains all hexagons with given centers (aligned).
        """
        normals = get_outer_normals(alpha)
        max_proj = 0.0
        for c in centers:
            # Vertices of this hexagon
            verts = base_vertices + c
            # Projections onto normals
            projs = verts @ normals.T
            max_proj = max(max_proj, np.max(projs))
        # S * sqrt(3)/2 >= max_proj
        S = 2.0 * max_proj / sqrt3
        return S
    
    def objective(x):
        """
        x: [x0, y0, x1, y1, ..., x10, y10, alpha]
        """
        if time.time() > deadline - 10:
            return 1e10
        centers = x[:-1].reshape(11, 2)
        alpha = x[-1]
        return calculate_required_side(centers, alpha)
    
    # Constraints: Non-overlap
    # For aligned unit hexagons, min distance is sqrt(3)
    # ||ci - cj|| >= sqrt(3)
    # ||ci - cj||^2 >= 3
    
    def constraint_func(x):
        centers = x[:-1].reshape(11, 2)
        cons = []
        for i in range(11):
            for j in range(i+1, 11):
                diff = centers[i] - centers[j]
                dist_sq = np.dot(diff, diff)
                cons.append(dist_sq - 3.0)
        return cons
    
    # Initial guess: 7-flower + 4 extras
    # 0: (0,0)
    # 1-6: Ring
    # 7-10: Attached to 1, 3, 5, 7? Let's attach to 1, 2, 4, 5 to spread them out.
    
    def init_positions():
        pos = np.zeros((11, 2))
        # Center
        pos[0] = [0, 0]
        # Ring
        for i in range(6):
            angle = i * np.pi / 3
            pos[i+1] = [sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)]
        
        # Extras
        # Attach 7 to 1 (angle 0) -> (2*sqrt3, 0)
        pos[7] = [2*sqrt3, 0]
        # Attach 8 to 3 (angle 120) -> pos[3] + sqrt3 * (cos120, sin120)
        pos[8] = pos[3] + sqrt3 * np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3)])
        # Attach 9 to 5 (angle 240) -> pos[5] + sqrt3 * (cos240, sin240)
        pos[9] = pos[5] + sqrt3 * np.array([np.cos(4*np.pi/3), np.sin(4*np.pi/3)])
        # Attach 10 to 2 (angle 60) -> pos[2] + sqrt3 * (cos60, sin60)
        pos[10] = pos[2] + sqrt3 * np.array([np.cos(np.pi/3), np.sin(np.pi/3)])
        
        return pos.flatten()
    
    x0 = init_positions()
    x0 = np.append(x0, 0.0) # alpha = 0
    
    # Define constraints for SLSQP
    # Inequality constraints: f(x) >= 0
    constraints = [
        {'type': 'ineq', 'fun': constraint_func}
    ]
    
    # Bounds: Keep centers within a reasonable range
    # Max side likely < 6. Centers should be within [-6, 6]
    bounds = [(-6, 6)] * 22 + [(0, np.pi/3)]
    
    best_score = 0
    best_x = x0
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res.success:
            best_x = res.x
        else:
            # If failed, maybe try a different method or check if it's still better than init
            # Calculate objective for res.x and x0
            obj_res = calculate_required_side(res.x[:-1].reshape(11,2), res.x[-1])
            obj_x0 = calculate_required_side(x0[:-1].reshape(11,2), x0[-1])
            if obj_res < obj_x0:
                best_x = res.x
            else:
                best_x = x0
                
    except Exception as e:
        print(f"Optimization error: {e}")
        best_x = x0

    # Extract results
    centers = best_x[:-1].reshape(11, 2)
    alpha = best_x[-1]
    
    # Calculate final side
    side = calculate_required_side(centers, alpha)
    
    # Ensure validity: check for overlaps numerically
    # If overlaps exist, the constraint might have been violated slightly or optimizer failed
    # We can try to fix by pushing them apart? 
    # For now, if invalid, we might return a known valid packing.
    
    # Check validity
    valid = True
    for i in range(11):
        for j in range(i+1, 11):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < sqrt3 - 1e-6:
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Fallback to a known valid packing (e.g. the seed's, or a slightly larger valid one)
        # Let's use the seed's packing which is valid but large
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        side = 8.0
    else:
        inner_hex_data = np.column_stack((centers, np.zeros(11)))
        outer_hex_data = np.array([0, 0, np.degrees(alpha)])
    
    return inner_hex_data, outer_hex_data, side

hexagon_packing_11()
