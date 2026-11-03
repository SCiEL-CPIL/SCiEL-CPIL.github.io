import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11

# Helper to generate hex vertices
def get_hex_vertices(cx, cy, angle_rad):
    # Unit hexagon, side 1
    # Vertices at angles angle_rad + k * pi/3
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))

# Helper to calculate min side length of enclosing hexagon
# Outer hexagon centered at (0,0) with rotation 0 (normals at 0, 60, 120, 180, 240, 300)
# Wait, if rotation is 0, what is the orientation?
# Standard: flat top means normals at 90, 150, 210, 270, 330, 30?
# Let's define normals explicitly.
# If angle is 0, let's say the first normal is at 0 deg.
normals = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3)/2],
    [0.5, -np.sqrt(3)/2]
])

def calc_min_side(vertices):
    # vertices: Nx2 array
    # Project onto normals
    # projections = vertices @ normals.T
    # max_proj = max(abs(projections))
    # A = max_proj
    # S = 2 * A / sqrt(3)
    projections = vertices @ normals.T
    max_proj = np.max(np.abs(projections))
    return 2.0 * max_proj / np.sqrt(3)

def objective(params, centers):
    # params: x_c, y_c, phi
    x_c, y_c, phi = params
    # Rotate centers by phi
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    rot_mat = np.array([[cos_p, -sin_p], [sin_p, cos_p]])
    rotated_centers = centers @ rot_mat.T
    # Translate
    translated_centers = rotated_centers + np.array([x_c, y_c])
    
    # Generate all vertices
    # Each center has 6 vertices
    # We can vectorize this
    # angles = phi + k * pi/3
    # v_k = c + (cos(angles), sin(angles))
    
    # Efficient calculation:
    # For each center, vertices are c + R(phi) @ v0
    # where v0 are vertices of unit hex at angle 0
    v0 = get_hex_vertices(0, 0, 0) # 6x2
    # Rotate v0 by phi
    v0_rot = v0 @ rot_mat.T
    # Add to each center
    # vertices = translated_centers[:, np.newaxis, :] + v0_rot[np.newaxis, :, :]
    # vertices shape: 11 x 6 x 2
    vertices = translated_centers[:, np.newaxis, :] + v0_rot[np.newaxis, :, :]
    vertices_flat = vertices.reshape(-1, 2)
    
    return calc_min_side(vertices_flat)

# Define cluster centers for different shapes
def get_cluster_centers(shape_type):
    d = np.sqrt(3) # distance between centers
    if shape_type == 0:
        # 4-3-3-1 trapezoid
        centers = [
            [-1.5*d, 0], [-0.5*d, 0], [0.5*d, 0], [1.5*d, 0],
            [-1.0*d, 1.5], [0.0, 1.5], [1.0*d, 1.5],
            [-1.0*d, -1.5], [0.0, -1.5], [1.0*d, -1.5],
            [-0.5*d, 3.0] # One extra on top
        ]
    elif shape_type == 1:
        # 3-3-3-2 trapezoid
        centers = [
            [-1.5*d, 0], [-0.5*d, 0], [0.5*d, 0],
            [-1.0*d, 1.5], [0.0, 1.5], [1.0*d, 1.5],
            [-1.0*d, -1.5], [0.0, -1.5], [1.0*d, -1.5],
            [-0.5*d, 3.0], [0.5*d, 3.0]
        ]
    elif shape_type == 2:
        # 2-3-3-3 trapezoid
        centers = [
            [-1.5*d, 0], [-0.5*d, 0],
            [-1.0*d, 1.5], [0.0, 1.5], [1.0*d, 1.5],
            [-1.0*d, -1.5], [0.0, -1.5], [1.0*d, -1.5],
            [-1.0*d, -3.0], [0.0, -3.0], [1.0*d, -3.0]
        ]
    elif shape_type == 3:
        # 1-3-3-3-1
        centers = [
            [0.0, 3.0],
            [-1.0*d, 1.5], [0.0, 1.5], [1.0*d, 1.5],
            [-1.0*d, -1.5], [0.0, -1.5], [1.0*d, -1.5],
            [-1.0*d, -3.0], [0.0, -3.0], [1.0*d, -3.0],
            [0.0, -4.5] # Wait, this is 11? 1+3+3+3+1 = 11.
        ]
        # Check adjacency.
        # Row y=3: 0
        # Row y=1.5: -1, 0, 1
        # Row y=-1.5: -1, 0, 1
        # Row y=-3: -1, 0, 1
        # Row y=-4.5: 0
        # This is valid lattice.
    else:
        # Default 7-flower + 4
        centers = [
            [0.0, 0.0],
            [d, 0.0], [-d, 0.0],
            [d/2, 1.5], [-d/2, 1.5],
            [d/2, -1.5], [-d/2, -1.5],
            [2*d, 0.0], # Extra
            [-2*d, 0.0], # Extra
            [d, 1.5*2], # Extra?
            [-d, 1.5*2]  # Extra?
        ]
    return np.array(centers)

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    best_score = float('inf')
    best_params = None
    best_shape_idx = 0
    
    # Try a few shapes
    for shape_idx in range(4):
        if time.time() > deadline:
            break
            
        centers = get_cluster_centers(shape_idx)
        
        # Initial guesses for x_c, y_c, phi
        # x_c, y_c should be near 0
        # phi should be near 0, pi/6, pi/3, etc.
        
        initial_guesses = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, np.pi/6],
            [0.0, 0.0, np.pi/3],
            [0.0, 0.0, np.pi/2],
            [0.0, 0.0, 5*np.pi/6],
        ]
        
        for guess in initial_guesses:
            if time.time() > deadline:
                break
                
            res = minimize(
                objective, 
                guess, 
                args=(centers,), 
                method='Nelder-Mead',
                options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 1000}
            )
            
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                best_shape_idx = shape_idx
                
    if best_params is None:
        # Fallback to seed-like behavior or a simple valid config
        # Just return a large hexagon
        centers = get_cluster_centers(0)
        inner = []
        for c in centers:
            inner.append([c[0], c[1], 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    x_c, y_c, phi = best_params
    centers = get_cluster_centers(best_shape_idx)
    
    # Construct inner list
    inner = []
    phi_deg = np.degrees(phi)
    for c in centers:
        # Rotate center by phi
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        rx = c[0] * cos_p - c[1] * sin_p + x_c
        ry = c[0] * sin_p + c[1] * cos_p + y_c
        inner.append([rx, ry, phi_deg])
        
    return inner, (0.0, 0.0), best_score, 0.0

