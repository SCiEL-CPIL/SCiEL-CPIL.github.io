import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Fixed topology: 12 hexagons in a compact hexagonal arrangement
    # Center (0,0), Ring 1 (6), Ring 2 (5 to make 12? No, 1+6+9=16. 1+6=7. 1+6+5=12 is a partial ring 2)
    # Actually, the most compact 12 is a hexagon of side 2 (19 hexes) minus 7? 
    # Or a "flower" of 7 plus 5 more?
    # Let's use the standard hex grid coordinates.
    # A hexagon of side length 2 (in grid units) has 1+3+5+7+9 = 25? No.
    # Side 1: 1
    # Side 2: 1 + 6 = 7
    # Side 3: 1 + 6 + 12 = 19
    # So 12 hexagons will be a side-2 hexagon (7) plus 5 from the next ring.
    # To minimize the bounding box, we should place the 5 extra hexagons symmetrically.
    # The 7 form a compact cluster. The 5 extra can be placed at the "corners" of the next ring.
    # Grid coords (q, r):
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 candidates: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    # We need 5 more. To keep it compact and symmetric, pick 5 from the 12 corners of the side-3 hex.
    # Actually, a "super-hexagon" of 12 isn't a standard shape.
    # Let's just use the 7 central ones and 5 surrounding them in a way that minimizes the max distance from center.
    # The 7 are within distance 1 (grid).
    # The 5 extra should be at distance 2.
    # Let's pick (2,0), (2,-1), (0,-2), (-2,0), (-2,1) ? That's 5.
    # Let's check symmetry.
    # (2,0) -> angle 0
    # (2,-1) -> angle -30
    # (0,-2) -> angle -90
    # (-2,0) -> angle 180
    # (-2,1) -> angle 150
    # This is not rotationally symmetric.
    
    # Alternative: Just use the 12 hexagons of a "hexagonal number" shape? No, 12 isn't a hex number.
    # Best strategy: Use the 7 central hexagons, and add 5 more to balance the shape.
    # Or, simply take the 12 hexagons closest to the origin in the hex lattice.
    # Distances (squared) in hex lattice:
    # 0: (0,0)
    # 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 6 hexes
    # 4: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2), (1,-1)?? No.
    # Let's list coordinates by distance from origin.
    # d=0: (0,0)
    # d=1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # d=2: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    # There are 12 hexes at d=2. We need 5 of them.
    # To minimize the bounding box, we should pick them such that the cluster is as "round" as possible.
    # Let's pick every other one? 12/5 is not integer.
    # Let's try to maximize the minimum distance between the outer hexes?
    # Actually, for a fixed set of points, the optimal orientation is what we will optimize.
    # The choice of WHICH 5 hexes from the d=2 ring matters.
    # Let's try a set that is roughly symmetric.
    # Indices in d=2 ring (0 to 11):
    # 0: (2,0)
    # 1: (2,-1)
    # 2: (2,-2)
    # 3: (1,-2)
    # 4: (0,-2)
    # 5: (-1,-1)
    # 6: (-2,0)
    # 7: (-2,1)
    # 8: (-2,2)
    # 9: (-1,2)
    # 10: (0,2)
    # 11: (1,1)
    
    # Let's pick indices: 0, 2, 4, 6, 8? (Every 2nd one, skipping one between)
    # 0: (2,0)
    # 2: (2,-2)
    # 4: (0,-2)
    # 6: (-2,0)
    # 8: (-2,2)
    # This leaves gaps at 1, 3, 5, 7, 9, 11.
    # Let's try another set: 0, 3, 6, 9, 11?
    # Let's just use a heuristic: pick the 5 that are farthest apart?
    # Or simpler: Since we are optimizing rotation, the specific choice might not be critical if the cluster is "full enough".
    # However, the "holes" in the ring might allow the outer hexagon to be smaller if the points are distributed evenly.
    # Let's use the set: (2,0), (1,-2), (-1,-1), (-2,1), (1,1) ?
    # Let's just use the first 5 from the list for now, or better, a symmetric subset.
    # A symmetric subset of 5 from 12 is hard.
    # Let's use the 7 central + 5 outer.
    # Let's try: (2,0), (0,-2), (-2,0), (-1,2), (1,1) -- wait, let's just code it to try a few specific topologies if time permits, or just one good one.
    # Let's use the topology:
    # Center + Ring 1 (6) + 5 from Ring 2.
    # To be safe and compact, let's pick the 5 from Ring 2 that are "closest" to the vertices of the Ring 1 hexagon?
    # Ring 1 vertices are at angles 0, 60, 120, 180, 240, 300 (in grid space approx).
    # Let's just use the 5 hexes at indices 0, 2, 5, 7, 10 of the d=2 ring?
    # Let's define the points explicitly.
    
    # Grid to Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    grid_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (0, -2), (-2, 0), (-1, 2), (1, 1)
    ]
    # Note: I picked (2,0), (0,-2), (-2,0), (-1,2), (1,1) as the 5 outer ones.
    # Let's check if this is balanced.
    # (2,0) is East.
    # (0,-2) is South-West-ish.
    # (-2,0) is West.
    # (-1,2) is North-West-ish.
    # (1,1) is North-East-ish.
    # This seems reasonably distributed.
    
    points = []
    for q, r in grid_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        points.append([x, y])
    
    points = np.array(points)
    # Center the points
    points -= np.mean(points, axis=0)
    
    # Vertices of a unit hexagon (side 1, angle 0)
    # Angles: 0, 60, 120, 180, 240, 300
    unit_hex_verts = np.array([
        [np.cos(np.radians(a)), np.sin(np.radians(a))] for a in [0, 60, 120, 180, 240, 300]
    ])
    
    # Pre-compute the convex hull of the union of all hexagons?
    # No, we need to check containment in the outer hexagon.
    # The outer hexagon is defined by its center (0,0), side length S, and rotation theta.
    # A point p is inside the outer hexagon if its projection onto the 6 normal directions is <= S * sqrt(3) / 2.
    # The normals of the outer hexagon (rotated by theta) are at angles theta + 30, theta + 90, ...
    
    def objective(params):
        # params: [theta_inner, theta_outer]
        # We can assume the outer hexagon is centered at origin.
        # The inner cluster is rotated by theta_inner.
        # The outer hexagon is rotated by theta_outer.
        # Actually, only the relative angle matters.
        # Let's fix theta_outer = 0 and optimize theta_inner.
        # Or optimize both, but it's equivalent to optimizing the difference.
        # Let's optimize theta_inner, with theta_outer = 0.
        # This reduces parameters to 1.
        # Wait, the problem allows rotating the outer hexagon too.
        # But since the outer hexagon is just a container, its rotation is relative to the inner cluster.
        # So we only need to optimize the relative angle.
        
        theta = params[0]
        
        # Rotate inner points
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # For each inner hexagon, we need to check if all its vertices are inside the outer hexagon.
        # The vertices of the i-th inner hexagon are:
        # v_k = R * (points[i] + R_inner * unit_hex_verts[k])
        # But wait, the inner hexagons can also rotate individually?
        # The task says "each inner hexagon may translate and rotate".
        # In my fixed topology, I assumed they all have the same orientation (aligned with the grid).
        # Is it beneficial to rotate individual hexagons?
        # Probably not significantly for a dense packing, as they are already touching.
        # Let's assume they all have the same orientation as the cluster.
        
        # So, the vertices of the i-th hexagon in the global frame are:
        # v_{i,k} = R * (points[i] + unit_hex_verts[k])
        # Note: unit_hex_verts are already in the local frame.
        # So we just rotate the whole cluster.
        
        # Pre-compute the vertices of all hexagons in the local frame (unrotated cluster)
        # local_verts[i, k] = points[i] + unit_hex_verts[k]
        
        # Then global_verts[i, k] = R @ local_verts[i, k]
        
        # We need the max projection of these points onto the normals of the outer hexagon.
        # Normals of outer hexagon (angle 0):
        # n_j = [cos(30 + 60j), sin(30 + 60j)] for j=0..5
        # Wait, the normals are perpendicular to the sides.
        # For a hexagon with vertices at 0, 60, ..., the sides are at 30, 90, ...
        # So normals are at 30, 90, 150, 210, 270, 330.
        
        normal_angles = np.radians(np.arange(30, 390, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Compute all global vertices
        # local_verts shape: (12, 6, 2)
        local_verts = points[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
        
        # Apply rotation
        # global_verts = local_verts @ R.T ? No.
        # v_global = R @ v_local
        # So for a batch of vectors V (N, 2), V_rot = V @ R.T
        # Here we have (12, 6, 2).
        # We can reshape to (72, 2), multiply, then reshape back.
        flat_local = local_verts.reshape(-1, 2)
        flat_global = flat_local @ R.T
        all_global_verts = flat_global.reshape(12, 6, 2)
        
        # Project onto normals
        # proj[i, k, j] = all_global_verts[i, k, :] dot normals[j]
        # We want max over i, k for each j.
        # proj_max[j] = max_{i,k} (all_global_verts[i, k, :] dot normals[j])
        
        # Efficiently:
        # all_global_verts_flat = all_global_verts.reshape(-1, 2)
        # projections = all_global_verts_flat @ normals.T  # (72, 6)
        # max_proj = np.max(projections, axis=0) # (6,)
        
        all_global_verts_flat = all_global_verts.reshape(-1, 2)
        projections = all_global_verts_flat @ normals.T
        max_proj = np.max(projections, axis=0)
        
        # The condition for containment is:
        # max_proj[j] <= S * sqrt(3) / 2 for all j.
        # So S >= 2/sqrt(3) * max(max_proj).
        
        S = (2.0 / np.sqrt(3.0)) * np.max(max_proj)
        
        return S

    # We only have 1 parameter: theta
    # We can just do a simple grid search or minimize.
    # Since it's 1D, we can sample it finely.
    
    best_S = float('inf')
    best_theta = 0.0
    
    # Sample theta from 0 to 60 degrees (symmetry of hexagon)
    thetas = np.linspace(0, np.pi/3, 1000)
    
    for theta in thetas:
        S = objective([theta])
        if S < best_S:
            best_S = S
            best_theta = theta
    
    # Refine with minimize
    try:
        res = minimize(objective, [best_theta], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000})
        if res.success and res.fun < best_S:
            best_S = res.fun
            best_theta = res.x[0]
    except Exception:
        pass

    # Construct the inner hexagons
    c = np.cos(best_theta)
    s = np.sin(best_theta)
    R = np.array([[c, -s], [s, c]])
    
    inner = []
    for i in range(N):
        # Position is the center of the hexagon
        center = R @ points[i]
        # Angle is the rotation of the hexagon
        angle_deg = np.degrees(best_theta)
        inner.append([float(center[0]), float(center[1]), float(angle_deg)])
    
    # Outer hexagon
    # Center is (0,0)
    # Side is best_S
    # Angle is 0 (since we fixed theta_outer=0)
    
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
