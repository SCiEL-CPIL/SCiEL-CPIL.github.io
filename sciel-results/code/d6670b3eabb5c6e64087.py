import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Configuration 1: 3x3 block (9 hexagons) + 2 hexagons attached to one side
    # Axial coordinates for 3x3 block centered at (0,0)
    # q in [-1, 1], r in [-1, 1]
    # But hex grid 3x3 is not a simple square in axial.
    # Let's use a "flower" shape: Center (0,0) + 6 neighbors = 7 hexagons.
    # Add 4 more.
    # To ensure disjointness with margin, we will scale positions.
    
    # Base positions in axial coordinates
    # Center
    base_axial = [(0,0)]
    # Ring 1 (6 neighbors)
    ring1 = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    base_axial.extend(ring1)
    
    # Add 4 more to make 11.
    # Attach to (1,0), (1,-1), (0,-1), (-1,0)?
    # Let's try adding to the "right" side.
    # (2,0) is neighbor of (1,0) and (1,-1)? No, (2,0) neighbor of (1,0) and (2,-1).
    # Let's just add (2,0), (3,0), (2,-1), (3,-1) to form a 2x2 block attached?
    # Simpler: Add (2,0), (2,-1), (3,-1), (3,-2).
    # Let's just use a line of 4 attached to the ring?
    # (2,0), (3,0), (4,0), (5,0). This is very long.
    
    # Better: Add (2,0), (2,-1), (3,-1), (3,-2)
    extra = [(2,0), (2,-1), (3,-1), (3,-2)]
    base_axial.extend(extra)
    
    # Convert to Cartesian (unit hexagons)
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        return x, y

    points = [axial_to_cartesian(q, r) for q, r in base_axial]
    
    # Center the cluster
    cx = np.mean([p[0] for p in points])
    cy = np.mean([p[1] for p in points])
    centered_points = [(p[0]-cx, p[1]-cy) for p in points]
    
    # Scale to ensure disjointness (add small gap)
    scale_factor = 1.02
    scaled_points = [(x * scale_factor, y * scale_factor) for x, y in centered_points]
    
    # Generate vertices for all inner hexagons
    # Unit hexagon vertices relative to center
    hex_verts_rel = [(np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))) for a in range(0, 360, 60)]
    
    all_vertices = []
    for px, py in scaled_points:
        for vx, vy in hex_verts_rel:
            all_vertices.append((px+vx, py+vy))
            
    all_vertices = np.array(all_vertices)
    
    best_S = float('inf')
    best_angle = 0.0
    
    # Search for optimal outer hexagon rotation
    # Due to symmetry, check 0 to 60 degrees
    # Step size 1 degree for precision
    for angle_deg in np.arange(0, 60, 1):
        if time.time() > deadline:
            break
        angle_rad = np.deg2rad(angle_deg)
        
        # Normals of the outer hexagon
        # Outer hexagon is regular, side S.
        # Inequality: V dot n_i <= S * sqrt(3)/2
        # So S >= (2/sqrt(3)) * max(V dot n_i)
        
        max_proj = 0.0
        for k in range(6):
            a = angle_rad + np.deg2rad(k*60)
            nx = np.cos(a)
            ny = np.sin(a)
            projs = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            curr_max = np.max(projs)
            if curr_max > max_proj:
                max_proj = curr_max
        
        S_required = (2.0 / np.sqrt(3.0)) * max_proj
        
        if S_required < best_S:
            best_S = S_required
            best_angle = angle_deg
            
    # If search failed or timed out, use a safe upper bound
    if best_S == float('inf'):
        # Calculate bounding box diagonal as a safe estimate
        xs = all_vertices[:, 0]
        ys = all_vertices[:, 1]
        max_x = np.max(xs) - np.min(xs)
        max_y = np.max(ys) - np.min(ys)
        # Safe side length for a hexagon containing a box of size WxH
        # Worst case is diagonal
        # S >= max(W, H*sqrt(3)/2, ... )
        # Let's just use a large number
        best_S = 10.0
        best_angle = 0.0
        
    # Prepare inner hexagons
    # The inner hexagons are at scaled_points, with angle 0
    inner = [[x, y, 0.0] for x, y in scaled_points]
    
    # Outer hexagon center is (0,0)
    # Outer hexagon side is best_S
    # Outer hexagon angle is best_angle
    # Note: The evaluator might expect the angle to be the rotation of the outer hexagon.
    # The normals were defined relative to angle_deg.
    # So returning best_angle as the outer rotation is correct.
    
    return inner, (0.0, 0.0), best_S, best_angle
