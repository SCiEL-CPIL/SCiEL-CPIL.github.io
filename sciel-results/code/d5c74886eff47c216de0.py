import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # 1. Generate inner hexagon centers in a compact hexagonal lattice pattern
    # We use axial coordinates (q, r) for hex grids.
    # We want to generate N centers closest to the origin.
    
    centers = []
    # Generate candidates in increasing distance from origin
    # Axial to pixel (pointy top):
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # size = 1 (side length)
    
    # We'll generate a sufficient number of hexagons and pick the N closest to origin
    # Distance in axial coords is roughly (|q| + |r| + |q+r|) / 2
    # Or just use Euclidean distance in pixel coords.
    
    candidates = []
    max_radius = 10
    for r in range(-max_radius, max_radius + 1):
        for q in range(-max_radius, max_radius + 1):
            # Pixel coordinates
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            candidates.append((x, y, q, r))
            
    # Sort by distance from origin
    candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
    
    # Take the first N
    inner_centers = []
    for i in range(N):
        x, y, q, r = candidates[i]
        inner_centers.append((x, y))
        
    inner = [[x, y, 0.0] for x, y in inner_centers]
    
    # 2. Determine the minimal enclosing regular hexagon
    # Collect all vertices of the inner hexagons
    # A unit regular hexagon (pointy-top) centered at (cx, cy) has vertices at:
    # (cx + cos(30 + 60k), cy + sin(30 + 60k)) for k=0..5?
    # Pointy top: vertices at 90, 150, 210, 270, 330, 30 degrees.
    # cos(90)=0, sin(90)=1 -> (0, 1)
    # Let's use flat-top for simplicity? No, pointy-top matches the lattice generation above.
    # Vertices of pointy-top unit hexagon:
    # angles: 30, 90, 150, 210, 270, 330 degrees.
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    offsets = np.column_stack([np.cos(angles), np.sin(angles)])
    
    all_vertices = []
    for cx, cy in inner_centers:
        verts = offsets + np.array([cx, cy])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Minimize side length S of outer hexagon containing all_vertices
    # Outer hexagon is centered at (0,0) with rotation theta.
    # Normals of outer hexagon sides:
    # For a regular hexagon, normals are at angles theta, theta+60, ..., theta+300.
    # The distance from center to side is h = S * sqrt(3) / 2.
    # Constraint: |dot(p, n_i)| <= h for all p, all i.
    # So S >= (2/sqrt(3)) * max_i |dot(p, n_i)|.
    # We want to minimize max_i |dot(p, n_i)| over theta.
    
    # Due to symmetry, we only need to check theta in [0, 30 degrees].
    # Let's sample theta finely.
    
    min_S = float('inf')
    best_theta = 0.0
    
    # Sample 360 degrees? No, symmetry is 60 degrees, but the cluster might break symmetry.
    # However, the set of points is fixed. The function f(theta) is periodic with 60 deg.
    # We can just sample 0 to 60.
    
    thetas = np.linspace(0, np.pi/3, 1000)
    
    for theta in thetas:
        # Normals
        normal_angles = theta + np.deg2rad(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project all vertices onto normals
        # dots = all_vertices @ normals.T  (shape: N_verts x 6)
        # We need max abs dot for each normal, then max over normals
        dots = all_vertices @ normals.T
        max_abs_dots = np.max(np.abs(dots), axis=0)
        max_proj = np.max(max_abs_dots)
        
        S = (2.0 / np.sqrt(3.0)) * max_proj
        
        if S < min_S:
            min_S = S
            best_theta = theta
            
    # Return values
    # inner: list of [x, y, angle_degrees]
    # center: (0,0)
    # side: min_S
    # angle: best_theta in degrees
    
    return inner, (0.0, 0.0), float(min_S), float(np.rad2deg(best_theta))
# EVOLVE-BLOCK-END
