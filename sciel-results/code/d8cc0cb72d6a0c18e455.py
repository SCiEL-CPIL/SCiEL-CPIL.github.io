import numpy as np
from scipy.optimize import minimize

def construct():
    # N = 11
    N = 11
    
    # Initial guess: Hexagonal lattice cluster
    # 1 center, 6 around, 4 more
    # Using axial coordinates converted to cartesian
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 3.0 / 2.0 * r
        return x, y

    # Generate a 5x5 grid of axial coordinates
    candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            x, y = axial_to_cartesian(q, r)
            candidates.append((x, y))
            
    # Select 11 closest to origin as initial guess
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_points = np.array(candidates[:N])
    
    # Initial angles: 0
    init_angles = np.zeros(N)
    
    # Flatten parameters: x0, y0, a0, x1, y1, a1, ...
    x0 = np.zeros(3 * N)
    x0[0::3] = init_points[:, 0]
    x0[1::3] = init_points[:, 1]
    x0[2::3] = init_angles
    
    def objective(params):
        # Unpack
        xs = params[0::3]
        ys = params[1::3]
        angles = params[2::3]
        
        # Get vertices of all inner hexagons
        # A unit hexagon has vertices at distance 1 from center
        # Vertex angles: 0, 60, 120, 180, 240, 300 degrees
        base_angles = np.deg2rad(np.arange(0, 360, 60))
        
        all_vertices = []
        for i in range(N):
            cx, cy = xs[i], ys[i]
            rot = np.deg2rad(angles[i])
            verts = np.column_stack([
                cx + np.cos(base_angles + rot),
                cy + np.sin(base_angles + rot)
            ])
            all_vertices.append(verts)
            
        all_vertices = np.vstack(all_vertices)
        
        # Find minimal regular hexagon containing all_vertices
        # We need to optimize the center (cx, cy) and rotation (alpha) of the outer hexagon
        # To make it fast, we can sample alpha or use a nested optimization
        # However, for a fixed set of points, the minimal bounding regular hexagon
        # is found by aligning the hexagon's normals with the convex hull edges?
        # Actually, we can just optimize cx, cy, alpha directly in a nested loop
        # or use a simpler approximation:
        
        # Approximation: The minimal side length S is determined by the max projection
        # onto the 6 normals of the outer hexagon.
        # Let's optimize alpha, cx, cy to minimize S.
        # This is a 3-variable optimization for each call to objective.
        # This might be slow.
        
        # Alternative: Assume the outer hexagon is aligned with the "average" orientation
        # or just optimize alpha, cx, cy using a few iterations of gradient descent
        # or simply check a few candidate alphas.
        
        # For speed, let's assume the optimal outer hexagon rotation is close to 0
        # or 30 degrees relative to the inner cluster.
        # We will test a few alphas.
        
        best_S = float('inf')
        
        # Try a range of alphas
        for alpha_deg in np.linspace(0, 60, 13):
            alpha = np.deg2rad(alpha_deg)
            # Normals of the outer hexagon
            # If vertices are at alpha + k*60, normals are at alpha + 30 + k*60
            normal_angles = alpha + np.deg2rad(30) + np.deg2rad(np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            # Project all vertices onto normals
            # projections shape: (6, M) where M is number of vertices
            projections = np.dot(normals.T, all_vertices.T)
            
            # For each normal direction k, we need:
            # max(proj_k) - min(proj_k) <= S * sqrt(3)
            # Because the width of a regular hexagon of side S in the normal direction is S*sqrt(3)
            
            widths = projections.max(axis=1) - projections.min(axis=1)
            
            # S must satisfy S * sqrt(3) >= max(widths)
            S_candidate = np.max(widths) / np.sqrt(3)
            
            if S_candidate < best_S:
                best_S = S_candidate
                best_alpha = alpha_deg
                # Store center for return?
                # Center is midpoint of min/max projections for the critical normal?
                # We can compute it later.
                
        return best_S

    # Run optimization
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
    
    best_params = result.x
    best_S = result.fun
    
    # Extract final positions
    xs = best_params[0::3]
    ys = best_params[1::3]
    angles = best_params[2::3]
    
    # Determine outer hexagon center and angle
    # We need to find the center (cx, cy) and angle alpha that achieves best_S
    # Re-use the logic from objective to find the best alpha
    all_vertices = []
    base_angles = np.deg2rad(np.arange(0, 360, 60))
    for i in range(N):
        cx, cy = xs[i], ys[i]
        rot = np.deg2rad(angles[i])
        verts = np.column_stack([
            cx + np.cos(base_angles + rot),
            cy + np.sin(base_angles + rot)
        ])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    best_S_final = float('inf')
    best_center = (0, 0)
    best_angle_final = 0
    
    for alpha_deg in np.linspace(0, 60, 13):
        alpha = np.deg2rad(alpha_deg)
        normal_angles = alpha + np.deg2rad(30) + np.deg2rad(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        projections = np.dot(normals.T, all_vertices.T)
        widths = projections.max(axis=1) - projections.min(axis=1)
        S_candidate = np.max(widths) / np.sqrt(3)
        
        if S_candidate < best_S_final:
            best_S_final = S_candidate
            best_angle_final = alpha_deg
            # Find center
            # The center is the average of the midpoints of the min/max projections
            # for the critical directions?
            # Actually, for a regular hexagon, the center is the intersection of the midlines.
            # We can compute the center as the mean of the vertices of the minimal bounding box?
            # Simpler: The center is the point that minimizes the max distance to the boundaries.
            # Let's just use the centroid of the vertices as a good approximation for the center
            # if the shape is roughly symmetric.
            # Or better, compute the center from the projections.
            # For each normal k, the center's projection should be (max_k + min_k) / 2.
            # We can solve for cx, cy using least squares.
            A = np.zeros((6*2, 2))
            b = np.zeros(6*2)
            for k in range(6):
                n = normals[k]
                # proj = n . c
                # We want n . c = (max_k + min_k) / 2
                target = (projections[k].max() + projections[k].min()) / 2.0
                A[k] = n
                b[k] = target
                A[k+6] = -n # Redundant, but helps if we add more constraints
                b[k+6] = -target
            # Least squares
            center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            best_center = (center[0], center[1])
            
    inner = [[xs[i], ys[i], angles[i]] for i in range(N)]
    
    return inner, best_center, best_S_final, best_angle_final

