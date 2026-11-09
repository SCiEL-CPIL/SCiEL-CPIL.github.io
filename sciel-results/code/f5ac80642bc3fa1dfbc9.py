"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import time
import math

N = 12

# EVOLVE-BLOCK-START

def _hex_verts(cx, cy, s, ang_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with side s and rotation ang_deg."""
    ang = math.radians(ang_deg)
    verts = []
    for k in range(6):
        theta = ang + k * math.pi / 3.0
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _point_in_polygon(pt, poly):
    """Ray casting algorithm to check if point is inside polygon."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def _check_overlap(v1, v2):
    """
    Check if two hexagons (defined by their 6 vertices) overlap.
    Returns True if they overlap, False otherwise.
    """
    # Check vertices of v1 in v2
    for pt in v1:
        if _point_in_polygon(pt, v2):
            return True
    # Check vertices of v2 in v1
    for pt in v2:
        if _point_in_polygon(pt, v1):
            return True
    # Check edge intersections
    for i in range(6):
        for j in range(6):
            if _segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    return False

def _min_enclosing_hex_side(vertices_list):
    """
    Calculate the minimum side length of a flat-top regular hexagon centered at (0,0) 
    that encloses all given vertices.
    """
    # Normals for flat-top hexagon edges
    # Top: (0, 1)
    # Top-Right: (sqrt(3)/2, 1/2)
    # Bottom-Right: (sqrt(3)/2, -1/2)
    # Bottom: (0, -1)
    # Bottom-Left: (-sqrt(3)/2, -1/2)
    # Top-Left: (-sqrt(3)/2, 1/2)
    normals = np.array([
        [0.0, 1.0],
        [math.sqrt(3)/2, 0.5],
        [math.sqrt(3)/2, -0.5],
        [0.0, -1.0],
        [-math.sqrt(3)/2, -0.5],
        [-math.sqrt(3)/2, 0.5]
    ])
    
    max_proj = 0.0
    # Combine all vertices
    all_verts = np.vstack(vertices_list)
    # Project onto all normals
    # projs shape: (num_verts, 6)
    projs = all_verts @ normals.T
    # We need the max absolute projection? No, the hexagon is centered at (0,0).
    # The constraint is dot(v, n) <= S * apothem_unit for all v, n.
    # But also dot(v, -n) <= S * apothem_unit is covered by dot(-v, n) or just max of abs?
    # Actually, the 6 normals cover all directions. The condition is:
    # For each normal n_i, max_v (v . n_i) <= S * (sqrt(3)/2)
    # AND min_v (v . n_i) >= -S * (sqrt(3)/2) which is equivalent to max_v (-v . n_i) <= S * (sqrt(3)/2)
    # Since we have normals for both directions (n and -n are in the list? No, the list has 6 distinct directions).
    # The 6 normals are symmetric. n and -n are not both in the list, but the set of 6 covers the 360 degrees.
    # Wait, the normals are:
    # 90 deg, 30 deg, -30 deg, -90 deg, -150 deg, 150 deg.
    # These are the outward normals of the 6 faces.
    # The condition for a point (x,y) to be inside a flat-top hexagon of side S centered at origin is:
    # |x| <= S
    # |y| <= S * sqrt(3) / 2
    # |x * sqrt(3) + y| <= S * 3/2 ? No.
    # Let's use the projection method correctly.
    # The support function of the hexagon in direction n is S * apothem = S * sqrt(3)/2.
    # So we need max_{v in V} (v . n_i) <= S * sqrt(3)/2 for all i.
    # Since the normals span all directions, this ensures containment.
    
    max_val = np.max(projs)
    # Note: Since the hexagon is centered at 0, and the set of normals is symmetric (if n is in, -n is effectively covered by the opposite face's normal? 
    # Actually, the 6 normals point to the 6 faces. The constraint is one-sided for each face.
    # Face 1 (Top): y <= a. Normal (0,1).
    # Face 4 (Bottom): -y <= a -> y >= -a. Normal (0,-1).
    # So yes, we just take the max projection for each of the 6 normals.
    # The required S is max_i ( max_v (v . n_i) ) / (sqrt(3)/2).
    
    S = max_val / (math.sqrt(3) / 2.0)
    return max(0.0, S)

def _objective(params, deadline):
    if time.time() > deadline:
        return 1e10
        
    p = params.reshape(N, 3)
    xs = p[:, 0]
    ys = p[:, 1]
    angles = p[:, 2]
    
    verts_list = []
    for i in range(N):
        v = _hex_verts(xs[i], ys[i], 1.0, angles[i])
        verts_list.append(v)
        
    # Check overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if _check_overlap(verts_list[i], verts_list[j]):
                penalty += 10.0
                
    if penalty > 0:
        return 100.0 + penalty # Large penalty to discourage overlaps
        
    S = _min_enclosing_hex_side(verts_list)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 120s limit
    
    # Initial guess: Honeycomb lattice
    # 4 columns, 3 rows? Or 3x4?
    # Let's try a compact arrangement.
    # Hex lattice vectors:
    # v1 = (1.5, sqrt(3)/2)
    # v2 = (0, sqrt(3))
    # This creates a grid.
    # Let's generate 12 positions.
    # 3 rows of 4 hexes.
    v1 = np.array([1.5, math.sqrt(3)/2])
    v2 = np.array([0.0, math.sqrt(3)])
    
    init_params = []
    for i in range(3):
        for j in range(4):
            pos = i * v1 + j * v2
            # Center the grid roughly at 0
            init_params.append([pos[0], pos[1], 0.0])
            
    init_params = np.array(init_params)
    cx = init_params[:, 0].mean()
    cy = init_params[:, 1].mean()
    init_params[:, 0] -= cx
    init_params[:, 1] -= cy
    
    # Bounds for DE
    # The hexagons shouldn't be too far apart.
    # Max side length S is roughly 4-5.
    # Coordinates should be within +/- 5.
    bounds = [(-5, 5), (-5, 5), (-180, 180)] * N
    
    # Run Differential Evolution
    # maxiter and popsize need to be balanced with time.
    # 36 dimensions.
    # Let's try popsize=10, maxiter=50.
    # We will monitor time inside the objective, but DE doesn't easily allow early stopping based on a global clock without wrapping.
    # However, we can set a smaller maxiter and run multiple times if time permits, or just one run.
    # Given the complexity, one run with reasonable parameters is safer.
    
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            args=(deadline,),
            maxiter=40, 
            popsize=8,
            tol=0.01,
            seed=42
        )
        
        best_params = res.x
        best_score = res.fun
        
    except Exception as e:
        # Fallback to initial guess if DE fails
        best_params = init_params
        best_score = _objective(init_params, deadline)
        
    # Verify the best solution found
    p = best_params.reshape(N, 3)
    xs = p[:, 0]
    ys = p[:, 1]
    angles = p[:, 2]
    
    verts_list = []
    valid = True
    for i in range(N):
        v = _hex_verts(xs[i], ys[i], 1.0, angles[i])
        verts_list.append(v)
        
    for i in range(N):
        for j in range(i+1, N):
            if _check_overlap(verts_list[i], verts_list[j]):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If the DE result is invalid, try to repair or fall back to initial
        # Simple repair: if initial is valid, use it.
        init_valid = True
        init_verts = []
        for i in range(N):
            v = _hex_verts(init_params[i,0], init_params[i,1], 1.0, init_params[i,2])
            init_verts.append(v)
        for i in range(N):
            for j in range(i+1, N):
                if _check_overlap(init_verts[i], init_verts[j]):
                    init_valid = False
                    break
            if not init_valid:
                break
                
        if init_valid:
            best_params = init_params
            xs = init_params[:, 0]
            ys = init_params[:, 1]
            angles = init_params[:, 2]
            verts_list = init_verts
        else:
            # Last resort: spread them out more to ensure validity, even if S is larger
            # Or just return a known valid sparse config
            print("Warning: Invalid result from DE and initial guess. Using sparse fallback.")
            # Sparse fallback: place them far apart
            best_params = np.zeros((N, 3))
            for i in range(N):
                best_params[i, 0] = i * 3.0
                best_params[i, 1] = 0.0
                best_params[i, 2] = 0.0
            xs = best_params[:, 0]
            ys = best_params[:, 1]
            angles = best_params[:, 2]
            verts_list = []
            for i in range(N):
                v = _hex_verts(xs[i], ys[i], 1.0, angles[i])
                verts_list.append(v)
                
    # Calculate final S
    S = _min_enclosing_hex_side(verts_list)
    
    # The center of the outer hexagon is (0,0) because we optimized relative to it
    # and the min enclosing hex function assumes center (0,0).
    # However, the "center" returned should be the center of the outer hexagon.
    # Since we calculated S based on vertices relative to (0,0), the outer hexagon is centered at (0,0).
    
    inner = []
    for i in range(N):
        inner.append([float(xs[i]), float(ys[i]), float(angles[i])])
        
    return inner, (0.0, 0.0), float(S), 0.0

# EVOLVE-BLOCK-END
