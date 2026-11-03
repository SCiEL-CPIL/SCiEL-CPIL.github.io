import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Define the 11 centers
    s3 = np.sqrt(3)
    centers = [
        [0.0, 0.0],
        [s3, 0.0],
        [s3/2, 1.5],
        [-s3/2, 1.5],
        [-s3, 0.0],
        [-s3/2, -1.5],
        [s3/2, -1.5],
        [0.0, 3.0],
        [0.0, -3.0],
        [s3, -3.0],
        [-s3, -3.0]
    ]
    
    # Vertices of a unit hexagon centered at origin
    # Angles 0, 60, 120, 180, 240, 300
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    base_vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Pre-compute global vertices for all inner hexagons
    # Each hexagon has 6 vertices
    # Total 66 vertices
    all_vertices = []
    for c in centers:
        verts = base_vertices + np.array(c)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices) # Shape (66, 2)
    
    def evaluate_alpha(alpha_rad):
        # Normals of the outer hexagon
        # Outer hexagon is rotated by alpha_rad
        # Normals are at alpha_rad + k * 60 deg
        normal_angles = alpha_rad + np.linspace(0, 2*np.pi, 7)[:-1]
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Projection of all vertices onto normals
        # proj[i, j] = dot(all_vertices[i], normals[j])
        proj = all_vertices @ normals.T
        max_proj = np.max(proj)
        
        # Apothem A = max_proj
        # Side L = 2/sqrt(3) * A
        L = (2.0 / s3) * max_proj
        return L

    # Ternary search for alpha in [0, pi/3]
    # The function is periodic with period pi/3
    # We assume unimodality in this interval
    lo = 0.0
    hi = np.pi / 3.0
    
    best_L = float('inf')
    best_alpha = 0.0
    
    # Quick grid search to find a good starting point or just use ternary
    # Ternary search
    for _ in range(50):
        if time.time() > deadline:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        
        f1 = evaluate_alpha(m1)
        f2 = evaluate_alpha(m2)
        
        if f1 < f2:
            hi = m2
        else:
            lo = m1
            
        if f1 < best_L:
            best_L = f1
            best_alpha = m1
        if f2 < best_L:
            best_L = f2
            best_alpha = m2
            
    # Final check at best_alpha
    final_L = evaluate_alpha(best_alpha)
    if final_L < best_L:
        best_L = final_L

    # Prepare output
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    center = [0.0, 0.0]
    side = best_L
    angle_deg = np.deg2rad(best_alpha) # Wait, return angle in degrees?
    # Task says: angle_degrees
    angle_deg = np.rad2deg(best_alpha)
    
    return inner, center, side, angle_deg
# EVOLVE-BLOCK-END
