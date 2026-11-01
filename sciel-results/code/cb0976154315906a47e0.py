"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import random
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Vertices of a unit regular hexagon (flat-top, side 1)
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Pre-define the rigid body cluster of 12 hexagons in hexagonal lattice coordinates
    # Center at (0,0), ring 1 (6 hexagons), ring 2 (5 hexagons)
    # Lattice basis: e1 = (sqrt(3), 0), e2 = (sqrt(3)/2, 1.5)
    sqrt3 = np.sqrt(3)
    lattice_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -2)
    ]
    
    # Convert lattice coords to Cartesian positions for the rigid body
    rigid_body_positions = []
    for q, r in lattice_coords:
        x = sqrt3 * q + sqrt3/2 * r
        y = 1.5 * r
        rigid_body_positions.append([x, y])
    
    rigid_body_positions = np.array(rigid_body_positions)
    
    # Function to compute the minimal enclosing hexagon side length for a set of points
    # assuming the outer hexagon is flat-top and centered at (cx, cy) with rotation theta
    def min_enclosing_hex_side(points, cx, cy, theta):
        # Rotate points by -theta, translate by (-cx, -cy)
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        
        # Transformation matrix
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        # Apply transformation to all points
        # points shape: (N, 2)
        # Translate
        pts = points - np.array([cx, cy])
        # Rotate
        pts = pts @ R.T
        
        # For each point, compute the maximum distance to the hexagon boundary
        # Hexagon inequalities for side L centered at origin (flat-top):
        # |x| <= L
        # |y| <= sqrt(3)/2 * L
        # |sqrt(3)/2 * x + 1/2 * y| <= sqrt(3)/2 * L
        
        # L must be >= max(|x|, |y| / (sqrt(3)/2), |sqrt(3)/2 * x + 1/2 * y| / (sqrt(3)/2))
        # L >= max(|x|, 2/sqrt(3) * |y|, |sqrt(3)/2 * x + 1/2 * y| / (sqrt(3)/2))
        # L >= max(|x|, 2/sqrt(3) * |y|, |x + 1/sqrt(3) * y|)
        
        # Let's compute the required L for each point
        x = pts[:, 0]
        y = pts[:, 1]
        
        term1 = np.abs(x)
        term2 = (2.0 / sqrt3) * np.abs(y)
        term3 = np.abs(x + (1.0 / sqrt3) * y)
        
        # Required L for each point is the max of these terms
        L_per_point = np.maximum(term1, np.maximum(term2, term3))
        
        # The overall required L is the max over all points
        return np.max(L_per_point)
    
    # We need to find the optimal center (cx, cy) and rotation theta for the rigid body
    # such that the minimal enclosing hexagon side length is minimized.
    # The rigid body is fixed, so we just optimize cx, cy, theta.
    
    def objective_rigid(params):
        cx, cy, theta = params
        return min_enclosing_hex_side(rigid_body_positions, cx, cy, theta)
    
    # Initial guess: center of mass of the rigid body, theta = 0
    cx0, cy0 = np.mean(rigid_body_positions, axis=0)
    params0 = np.array([cx0, cy0, 0.0])
    
    best_params = params0
    best_L = objective_rigid(params0)
    
    # Use Nelder-Mead to optimize
    try:
        result = minimize(objective_rigid, params0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        if result.fun < best_L:
            best_L = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    restarts = 0
    while time.time() < deadline - 10 and restarts < 10:
        random_theta = random.uniform(0, 2 * np.pi)
        random_cx = cx0 + random.uniform(-1, 1)
        random_cy = cy0 + random.uniform(-1, 1)
        params_rand = np.array([random_cx, random_cy, random_theta])
        
        try:
            result = minimize(objective_rigid, params_rand, method='Nelder-Mead',
                             options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if result.fun < best_L:
                best_L = result.fun
                best_params = result.x
        except Exception:
            pass
        restarts += 1
    
    cx, cy, theta = best_params
    
    # Now compute the final positions of the inner hexagons
    # The rigid body positions are in the local frame.
    # We need to rotate them by theta and translate by (cx, cy).
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    final_positions = rigid_body_positions @ R.T + np.array([cx, cy])
    
    # The inner hexagons are all rotated by theta (since they are part of the rigid body)
    angle_degrees = np.degrees(theta) % 360.0
    
    inner = []
    for i in range(N):
        x, y = final_positions[i]
        inner.append([float(x), float(y), float(angle_degrees)])
    
    center = (float(cx), float(cy))
    side = float(best_L)
    angle = float(angle_degrees)
    
    # Final validation check (simplified, assuming the rigid body is disjoint)
    # We just need to ensure containment is satisfied, which it is by construction.
    # But let's do a quick check to be safe.
    def is_valid_config(inner, L, center, angle):
        # Check containment
        cx, cy = center
        theta = np.radians(angle)
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        for x, y, _ in inner:
            # Transform to outer hexagon frame
            px = x - cx
            py = y - cy
            px_rot, py_rot = R[0, 0]*px + R[0, 1]*py, R[1, 0]*px + R[1, 1]*py
            
            # Check vertices
            for vx, vy in unit_hex_verts:
                # Rotate vertex by -angle (since inner hex is rotated by angle, we rotate back to check in outer frame)
                # Actually, the inner hex is rotated by angle, so its vertices in global frame are:
                # (x + vx*cos(angle) - vy*sin(angle), y + vx*sin(angle) + vy*cos(angle))
                # Then transform to outer frame: subtract (cx, cy), rotate by -angle_outer (which is 0 if outer is flat-top)
                # Wait, the outer hexagon is flat-top, so its frame is aligned with the global axes.
                # The inner hexagon is rotated by angle.
                # So in the global frame, the vertices are:
                vx_global = x + vx * np.cos(np.radians(angle)) - vy * np.sin(np.radians(angle))
                vy_global = y + vx * np.sin(np.radians(angle)) + vy * np.cos(np.radians(angle))
                
                # Check if this vertex is inside the outer hexagon (flat-top, side L, center (cx, cy))
                dx = vx_global - cx
                dy = vy_global - cy
                
                if (abs(dx) > L or 
                    abs(dy) > sqrt3/2 * L or 
                    abs(sqrt3/2 * dx + 0.5 * dy) > sqrt3/2 * L):
                    return False
        
        # Check disjointness (rigid body, so should be fine, but let's check)
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, _ = inner[i]
                x2, y2, _ = inner[j]
                dx = x1 - x2
                dy = y1 - y2
                # Check if d is inside Minkowski hexagon (side 2)
                if (abs(dx) <= 2.0 and 
                    abs(dy) <= sqrt3 and 
                    abs(sqrt3/2 * dx + 0.5 * dy) <= sqrt3):
                    return False
        return True
    
    if not is_valid_config(inner, side, center, angle):
        # Fallback: use a larger side length
        side *= 1.1
        if not is_valid_config(inner, side, center, angle):
            # Very large fallback
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            center = (0.0, 0.0)
            side = 40.0
            angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
