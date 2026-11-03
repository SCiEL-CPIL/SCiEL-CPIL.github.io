import numpy as np
import scipy.optimize as opt
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Helper to generate vertices of a unit hexagon at (cx, cy) with angle rot (radians)
    def get_hex_vertices(cx, cy, rot):
        angles = np.linspace(0, 2*np.pi, 7)[:-1] + rot
        # Unit hexagon vertices are at distance 1 from center
        # Standard orientation: flat top? 
        # Task says "regular hexagons of side length 1".
        # Usually "pointy top" or "flat top" doesn't matter for packing if we can rotate.
        # Let's assume standard mathematical orientation: vertex at (1,0) if rot=0?
        # If rot=0, vertex at (1,0).
        return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))

    # Solve for min enclosing hexagon for a set of points (vertices)
    # Variables: cx, cy, S
    # Constraints: For each point p=(px, py) and each normal angle theta_k:
    # (px - cx) * cos(theta_k) + (py - cy) * sin(theta_k) <= (sqrt(3)/2) * S
    # This is linear in cx, cy, S.
    def solve_lp(points, container_rot):
        # points: array of shape (M, 2)
        M = len(points)
        # 6 normals for the container
        normals = []
        for k in range(6):
            theta = container_rot + k * np.pi / 3
            normals.append((np.cos(theta), np.sin(theta)))
        
        # A matrix of shape (6*M, 3)
        # b vector of shape (6*M,)
        A_rows = []
        b_vals = []
        factor = np.sqrt(3) / 2.0
        
        for px, py in points:
            for nx, ny in normals:
                # nx*(px-cx) + ny*(py-cy) <= factor * S
                # -nx*cx - ny*cy - factor*S <= - (nx*px + ny*py)
                # Wait, linprog minimizes c^T x subject to A_ub x <= b_ub
                # We want to minimize S. So c = [0, 0, 1].
                # Constraint: nx*px - nx*cx + ny*py - ny*cy <= factor*S
                # -nx*cx - ny*cy - factor*S <= -(nx*px + ny*py)
                A_rows.append([-nx, -ny, -factor])
                b_vals.append(-(nx*px + ny*py))
                
        A_ub = np.array(A_rows)
        b_ub = np.array(b_vals)
        c = np.array([0.0, 0.0, 1.0])
        
        try:
            res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
            if res.success:
                return res.x[2] # S
            else:
                return float('inf')
        except Exception:
            return float('inf')

    best_score = float('inf')
    best_result = None
    
    # Candidate Packings
    # 1. 7-cluster + 4 bay hexagons
    # Centers:
    # 0: (0,0)
    # 1-6: distance sqrt(3), angles 0, 60, ...
    # 7-10: distance 3, angles 30, 150, 270, 330? 
    # Let's verify the "bay" positions.
    # The bays are at angles 30, 90, 150, 210, 270, 330.
    # We pick 4. To be compact, maybe pick 30, 150, 210, 330? Or 30, 90, 150, 270?
    # Any 4 will break symmetry.
    # Let's try a set of 4 that are roughly balanced.
    bay_angles = [30, 90, 150, 210] # Radians? No, degrees.
    bay_angles_rad = np.deg2rad(bay_angles)
    
    centers = [(0.0, 0.0)]
    d1 = np.sqrt(3)
    for k in range(6):
        a = k * np.pi / 3
        centers.append((d1 * np.cos(a), d1 * np.sin(a)))
    
    d2 = 3.0
    for a in bay_angles_rad:
        centers.append((d2 * np.cos(a), d2 * np.sin(a)))
        
    # Ensure we have 11 centers
    if len(centers) == 11:
        # Generate all vertices
        points = []
        for cx, cy in centers:
            verts = get_hex_vertices(cx, cy, 0.0)
            points.extend(verts.tolist())
        points = np.array(points)
        
        # Try a few container rotations
        for rot_deg in [0, 15, 30, 45]:
            rot_rad = np.deg2rad(rot_deg)
            S = solve_lp(points, rot_rad)
            if S < best_score:
                best_score = S
                # Store result
                # inner: list of [x, y, angle_deg]
                inner = [[cx, cy, 0.0] for cx, cy in centers]
                # Container center is (0,0) in this frame? 
                # The LP solved for cx, cy of container.
                # We need to re-run to get cx, cy or just assume 0,0 if we centered the points?
                # The points are centered around 0,0. The optimal container center might be slightly off.
                # For simplicity and speed, let's assume the container center is (0,0) 
                # and just find S? No, the LP finds the optimal center.
                # Let's re-run LP to get the full solution for the best rotation.
                
    # Re-run best rotation to get center
    # Actually, let's just return the best S found with center (0,0) for now?
    # No, the LP gives the optimal center.
    
    # Let's do a proper search for the best rotation and extract the full solution.
    
    best_S = float('inf')
    best_solution = None
    best_rot = 0.0
    
    # Use the points from the 7+4 packing
    points = np.array(points)
    
    for rot_deg in np.linspace(0, 60, 13): # 0, 5, 10, ... 60
        if time.time() > deadline:
            break
        rot_rad = np.deg2rad(rot_deg)
        # Solve LP
        M = len(points)
        normals = []
        for k in range(6):
            theta = rot_rad + k * np.pi / 3
            normals.append((np.cos(theta), np.sin(theta)))
        
        A_rows = []
        b_vals = []
        factor = np.sqrt(3) / 2.0
        
        for px, py in points:
            for nx, ny in normals:
                A_rows.append([-nx, -ny, -factor])
                b_vals.append(-(nx*px + ny*py))
                
        A_ub = np.array(A_rows)
        b_ub = np.array(b_vals)
        c = np.array([0.0, 0.0, 1.0])
        
        try:
            res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
            if res.success:
                S = res.x[2]
                if S < best_S:
                    best_S = S
                    best_solution = res.x # [cx, cy, S]
                    best_rot = rot_rad
        except Exception:
            pass

    if best_solution is not None:
        cx, cy, S = best_solution
        # Construct inner list
        inner = [[c[0], c[1], 0.0] for c in centers]
        return inner, (cx, cy), S, np.deg2rad(best_rot)
    else:
        # Fallback: simple line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We need to wrap it in the block if required, or just define it.
# The prompt says "Keep the function names and signatures".
# The current program has construct() inside EVOLVE-BLOCK.
# I will place the new code inside the block.
# EVOLVE-BLOCK-END
