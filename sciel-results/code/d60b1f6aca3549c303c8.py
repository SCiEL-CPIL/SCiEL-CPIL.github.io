import numpy as np
import scipy.optimize as opt
import scipy.spatial as spatial
import time

N = 12

def get_hex_vertices(cx, cy, angle_deg, side=1.0):
    """
    Returns the 6 vertices of a regular hexagon with given center, angle, and side length.
    Angle is in degrees, 0 means flat-top (vertex at right).
    """
    angle_rad = np.deg2rad(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    angles = angle_rad + np.arange(6) * np.pi / 3
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate initial configuration: 3x4 grid of flat-top hexagons
    # Rows i=0..2, Cols j=0..3
    inner = []
    vertices = []
    
    for i in range(3):
        for j in range(4):
            # Flat-top grid spacing
            cx = j * 1.5 + i * 0.75
            cy = i * np.sqrt(3)
            angle = 0.0
            inner.append([cx, cy, angle])
            verts = get_hex_vertices(cx, cy, angle)
            vertices.append(verts)
            
    vertices = np.vstack(vertices)
    
    # Center the cluster
    center = np.mean(vertices, axis=0)
    vertices_centered = vertices - center
    
    # Compute Convex Hull
    hull = spatial.ConvexHull(vertices_centered)
    hull_vertices = vertices_centered[hull.vertices]
    
    # 2. Optimize outer hexagon parameters
    # We want to find min S, center C, angle Theta
    # For a fixed Theta, the optimal C and S can be found by solving:
    # Minimize h (where S = 2h/sqrt(3))
    # Subject to n_k . C >= max_v(n_k . v) - h
    
    def calculate_min_h(theta_deg):
        theta_rad = np.deg2rad(theta_deg)
        # Normals for the 6 sides of the outer hexagon
        # Side normals are at angles theta + 0, 60, ... ?
        # If outer hexagon is flat-top (angle 0), vertices are at 0, 60...
        # Normals are at 30, 90, 150...
        # Let's define normals for a hexagon with "angle" theta.
        # If theta=0, it's flat-top. Normals at 30 + k*60.
        normal_angles = theta_rad + np.pi/6 + np.arange(6) * np.pi / 3
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # M_k = max_v (n_k . v)
        # hull_vertices is (M, 2)
        # normals is (6, 2)
        # dots is (6, M)
        dots = normals @ hull_vertices.T
        M = np.max(dots, axis=1)
        
        # Now solve for C and h
        # Min h
        # s.t. n_k . C - h <= -M_k  => -n_k . C + h >= M_k ?
        # Let's rewrite constraints:
        # n_k . C >= M_k - h
        # n_k . C + h >= M_k  (This is not linear in C, h? It is linear)
        # Wait, the constraint is n_k . (v - C) <= h  =>  n_k . v - n_k . C <= h
        # => n_k . C + h >= n_k . v
        # Since this must hold for all v, it must hold for max_v(n_k . v) = M_k
        # => n_k . C + h >= M_k
        
        # Variables: Cx, Cy, h
        # Objective: Minimize h
        # Constraints:
        # n_kx * Cx + n_ky * Cy + h >= M_k  for k=0..5
        # -n_kx * Cx - n_ky * Cy + h >= -M'_k ?
        # Wait, the hexagon is defined by 6 half planes.
        # The condition "v is inside" is max_k (n_k . (v-C)) <= h.
        # This is equivalent to n_k . (v-C) <= h for all k.
        # So n_k . C + h >= n_k . v for all k, v.
        # So n_k . C + h >= M_k for all k.
        
        # We also need to ensure the hexagon is "centered" in some sense?
        # No, C is a variable.
        
        # However, note that the normals n_k and -n_k are not both present in the set of 6 normals?
        # The 6 normals point outwards.
        # n_0 is at 30 deg. n_3 is at 150+90? No.
        # Angles: 30, 90, 150, 210, 270, 330.
        # n_0 = (cos30, sin30). n_3 = (cos210, sin210) = -n_0.
        # So the constraints are symmetric if the object is centered?
        # The LP will find the optimal C.
        
        # Objective: min h
        # c = [0, 0, 1]
        
        # Inequalities A_ub x <= b_ub
        # We have n_k . C + h >= M_k
        # => -n_k . C - h <= -M_k
        
        A = []
        b = []
        for k in range(6):
            A.append([-normals[k][0], -normals[k][1], -1.0])
            b.append(-M[k])
            
        A = np.array(A)
        b = np.array(b)
        
        # Bounds for C and h
        # C can be anything, but bounded by the hull size.
        # h must be positive.
        bounds = [(-10, 10), (-10, 10), (0, None)]
        
        try:
            res = opt.linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.success:
                return res.fun
        except:
            pass
        return float('inf')

    # Search for optimal theta
    best_h = float('inf')
    best_theta = 0.0
    
    # Scan theta from 0 to 60
    for theta in np.linspace(0, 60, 121):
        h = calculate_min_h(theta)
        if h < best_h:
            best_h = h
            best_theta = theta
            
    S = 2 * best_h / np.sqrt(3)
    
    # Reconstruct the outer hexagon center
    # We need to find the C that was optimal.
    # Let's re-run the LP for best_theta to get C
    theta_rad = np.deg2rad(best_theta)
    normal_angles = theta_rad + np.pi/6 + np.arange(6) * np.pi / 3
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    dots = normals @ hull_vertices.T
    M = np.max(dots, axis=1)
    
    A = []
    b = []
    for k in range(6):
        A.append([-normals[k][0], -normals[k][1], -1.0])
        b.append(-M[k])
    A = np.array(A)
    b = np.array(b)
    bounds = [(-10, 10), (-10, 10), (0, None)]
    
    res = opt.linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
    Cx, Cy, h_val = res.x
    
    # The outer hexagon center in the original coordinate system
    # We centered the hull at origin, so the outer center is (Cx, Cy) relative to hull center.
    # Hull center was 'center' in original coords.
    outer_center = (Cx + center[0], Cy + center[1])
    
    return inner, outer_center, S, best_theta

# EVOLVE-BLOCK-END
