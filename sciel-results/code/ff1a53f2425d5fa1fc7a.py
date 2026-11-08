import numpy as np
import time

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    sqrt3 = np.sqrt(3)
    
    # 1. Define the 11 inner hexagon centers on a hexagonal lattice
    # Axial coordinates (q, r)
    # Pattern: Center + Ring 1 (6) + 4 from Ring 2
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2)
    ]
    
    centers = []
    for q, r in hex_positions_axial:
        # Flat-top hexagon grid coordinates
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # 2. Generate all vertices of the inner hexagons
    # All inner hexagons are side 1, angle 0
    all_verts = []
    for cx, cy in centers:
        verts = hex_vertices(cx, cy, 1.0, 0.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts) # Shape (66, 2)
    
    # 3. Optimize outer hexagon center and angle
    # The outer hexagon is defined by center (cx, cy), side S, and angle alpha.
    # A point (px, py) is inside if for all 6 normals n_i:
    # (px-cx)*n_i[0] + (py-cy)*n_i[1] <= S * sqrt(3)/2
    # And the opposite side:
    # -(px-cx)*n_i[0] - (py-cy)*n_i[1] <= S * sqrt(3)/2
    # So S >= max( max_i max_p proj, max_i max_p -proj ) / (sqrt(3)/2)
    
    # Initial guess: centroid of vertices
    cx_init = all_verts[:, 0].mean()
    cy_init = all_verts[:, 1].mean()
    
    best_side = float('inf')
    best_center = (cx_init, cy_init)
    best_angle = 0.0
    
    # Search over angles
    # The symmetry of the outer hexagon is 60 degrees.
    # We can search 0 to 60.
    # Use a coarse to fine approach or just a fine grid if fast enough.
    # 66 points * 6 normals * 360 angles = ~140k operations. Very fast.
    
    angles = np.linspace(0, 60, 361) # 0.1 degree resolution
    
    for alpha_deg in angles:
        a = np.radians(alpha_deg)
        # Normals of the outer hexagon faces
        # Face i normal angle: alpha + pi/6 + i*pi/3
        normals = []
        for i in range(6):
            theta = a + np.pi/6 + i * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals) # Shape (6, 2)
        
        # Project all points onto normals
        # dx, dy relative to current center guess (cx_init, cy_init)
        # Note: We should also optimize the center.
        # For a given angle, the optimal center minimizes the max width in each normal direction.
        # This is equivalent to finding the center of the minimal bounding box in the 6 directions.
        # The optimal center is the midpoint of the min and max projections in each direction?
        # Not exactly, it's the center of the hexagon that minimizes the max projection.
        # This is a linear programming problem, but for 2D, we can iterate.
        
        # Simple approach: Use the centroid of the min and max points for each normal?
        # Let's calculate the required side length for the current center (cx_init, cy_init).
        # Then, we can adjust the center to reduce the max projection.
        
        dx = all_verts[:, 0] - cx_init
        dy = all_verts[:, 1] - cy_init
        
        # Projections: (66, 6)
        # proj[i, j] = (dx[i]*nx[j] + dy[i]*ny[j])
        projections = dx @ normals[:, 0] + dy @ normals[:, 1]
        
        # For each normal direction j, we have min_proj[j] and max_proj[j].
        # The width in that direction is max_proj[j] - min_proj[j].
        # The side length S must satisfy S * sqrt(3)/2 >= max( max_j max_proj[j], max_j -min_proj[j] )
        # Actually, the constraint is |proj| <= S * sqrt(3)/2.
        # So S >= max( max_j max_proj[j], max_j -min_proj[j] ) / (sqrt(3)/2)
        
        max_pos = projections.max(axis=0)
        min_neg = projections.min(axis=0)
        
        # The "radius" required in each direction
        radii = np.maximum(max_pos, -min_neg)
        side = radii.max() / (sqrt3 / 2)
        
        if side < best_side:
            best_side = side
            best_center = (cx_init, cy_init)
            best_angle = alpha_deg
        
        # Check time
        if time.time() > deadline:
            break
    
    # Refine: Optimize center for the best angle found
    # The current best_center is just the centroid.
    # We can improve it by finding the center that minimizes the max projection.
    # This is the center of the "Chebyshev ball" in the hexagonal norm.
    # We can use a simple iterative refinement or just check a small grid around the centroid.
    
    a_best = np.radians(best_angle)
    normals_best = []
    for i in range(6):
        theta = a_best + np.pi/6 + i * np.pi / 3
        normals_best.append([np.cos(theta), np.sin(theta)])
    normals_best = np.array(normals_best)
    
    # Try small shifts around the current best center
    # The optimal center is likely close to the centroid.
    # Let's try a small grid of centers.
    cx_c, cy_c = best_center
    
    # Estimate a search radius
    # The cluster is about 5 units wide.
    # Search within 0.5 units.
    search_radius = 0.5
    steps = 10
    dxs = np.linspace(-search_radius, search_radius, steps)
    dys = np.linspace(-search_radius, search_radius, steps)
    
    for dx_off in dxs:
        for dy_off in dys:
            cx_try = cx_c + dx_off
            cy_try = cy_c + dy_off
            
            dx = all_verts[:, 0] - cx_try
            dy = all_verts[:, 1] - cy_try
            
            projections = dx @ normals_best[:, 0] + dy @ normals_best[:, 1]
            
            max_pos = projections.max(axis=0)
            min_neg = projections.min(axis=0)
            
            radii = np.maximum(max_pos, -min_neg)
            side = radii.max() / (sqrt3 / 2)
            
            if side < best_side:
                best_side = side
                best_center = (cx_try, cy_try)
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Final safety margin
    best_side *= 1.0001
    
    # Construct inner list
    inner = [[float(cx), float(cy), float(0.0)] for cx, cy in centers]
    
    return inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)

if __name__ == "__main__":
    # Quick test
    inner, center, side, angle = construct()
    print(f"Side: {side}, Center: {center}, Angle: {angle}")
