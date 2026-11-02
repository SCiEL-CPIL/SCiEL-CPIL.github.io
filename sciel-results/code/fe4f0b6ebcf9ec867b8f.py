import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    Vertices are in CCW order.
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _construct_rigid_cluster():
    """
    Constructs a fixed, valid configuration of 11 unit hexagons.
    We use a central hexagon and 6 neighbors (ring 1), plus 4 more from ring 2
    to form a compact shape.
    
    Coordinates are relative to the center of the central hexagon (0,0).
    All inner hexagons are aligned (angle 0).
    
    Positions (x, y):
    0: (0, 0)
    1: (1.5, sqrt(3)/2)
    2: (1.5, -sqrt(3)/2)
    3: (0, sqrt(3))
    4: (0, -sqrt(3))
    5: (-1.5, sqrt(3)/2)
    6: (-1.5, -sqrt(3)/2)
    
    These 7 form a perfect hexagonal cluster of radius 2.
    To add 4 more (total 11), we pick 4 positions from the next ring that
    maintain compactness. 
    Ring 2 positions (distance ~3 from center, or neighbors of ring 1):
    The "corners" of the hexagonal cluster are at (3, 0), (1.5, 2.598), etc.
    Let's pick 4 that fill the gaps or extend the shape symmetrically if possible, 
    or just pick 4 distinct valid non-overlapping spots.
    
    Actually, a standard "11-hexagon" packing might not be a simple lattice slice.
    However, for the purpose of this rigid body strategy, ANY valid non-overlapping
    set of 11 hexagons will work. The tighter the cluster, the smaller the container.
    
    Let's use the 7 central ones, and add 4 more at:
    7: (3.0, 0)  -> Neighbor of 1 and 2
    8: (1.5, 2.598) -> Neighbor of 1 and 3
    9: (-1.5, 2.598) -> Neighbor of 3 and 5
    10: (-3.0, 0) -> Neighbor of 5 and 6
    
    This creates a shape that is roughly a larger hexagon with some corners.
    Let's verify non-overlap.
    Unit hexagons have side 1.
    Center-to-center distance for non-overlap (aligned) is >= 2.
    Distances in hex lattice:
    (0,0) to (1.5, 0.866): sqrt(2.25 + 0.75) = sqrt(3) ~ 1.732. 
    WAIT. For aligned hexagons (flat top or pointy side?), 
    My _hex_vertices uses angle 0 as vertex at (1,0). This is "pointy side" (vertex on x-axis).
    The distance between centers of two touching aligned "pointy-side" hexagons is 2 * apothem? No.
    If vertices are at 0, 60...
    Side length s=1.
    Apothem a = s * sqrt(3)/2.
    If two hexagons are aligned, they touch if the distance between centers is 2 * apothem?
    No, if they are side-by-side horizontally, the distance is 2 * apothem?
    Let's check.
    Hex 1 center (0,0). Rightmost vertex (1,0).
    Hex 2 center (d, 0). Leftmost vertex (d-1, 0).
    If they touch, d-1 = 1 => d=2.
    So distance 2 is the touching distance for aligned hexagons with vertices on x-axis.
    
    My lattice points above:
    (0,0) and (1.5, 0.866).
    dx = 1.5, dy = 0.866.
    dist = sqrt(2.25 + 0.75) = sqrt(3) ≈ 1.732.
    1.732 < 2.0. They OVERLAP.
    
    So the standard hex lattice coordinates for "pointy-side" hexagons need scaling.
    If vertices are at (1,0), (0.5, 0.866)...
    The "neighbors" in a hex tiling are at distance 2 * apothem? No.
    In a hex tiling of side s, the distance between centers is 2 * s * cos(30) = s * sqrt(3).
    Wait.
    Let's re-evaluate the geometry of _hex_vertices.
    Vertex 0: (1, 0).
    Vertex 1: (0.5, 0.866).
    Edge 0-1 length: sqrt(0.25 + 0.75) = 1. Correct.
    
    Distance from center to edge midpoint (apothem):
    Midpoint of 0-1 is (0.75, 0.433).
    Dist = sqrt(0.5625 + 0.1875) = sqrt(0.75) = 0.866 = sqrt(3)/2.
    
    If two such hexagons are placed side-by-side such that they share an edge,
    the distance between centers is 2 * apothem = sqrt(3) ≈ 1.732.
    
    So, for ALIGNED hexagons, the center-to-center distance for touching is sqrt(3).
    
    Let's re-check the lattice points.
    (0,0) and (1.5, 0.866).
    dist = sqrt(1.5^2 + 0.866^2) = sqrt(2.25 + 0.75) = sqrt(3).
    So these DO touch. They do NOT overlap.
    
    So the initial 7 hexagons at:
    (0,0)
    (1.5, 0.866), (1.5, -0.866)
    (0, 1.732), (0, -1.732)
    (-1.5, 0.866), (-1.5, -0.866)
    
    Are all pairwise touching or separated?
    (0,0) to (1.5, 0.866): dist sqrt(3). Touching.
    (1.5, 0.866) to (0, 1.732):
    dx = 1.5, dy = 0.866. dist sqrt(3). Touching.
    
    So this is a valid tiling.
    
    Now add 4 more.
    We want them to be close to the center to minimize the bounding box.
    The "gaps" in this 7-hex cluster are empty. The cluster is a perfect hexagon of "radius" 1 (in hex grid units).
    To add 4 more, we must go to the next ring.
    Next ring positions (distance 2*sqrt(3) from center? No, distance 2 grid steps).
    Grid step vector u = (1.5, 0.866), v = (1.5, -0.866).
    Ring 1: +/-u, +/-v, +/- (u-v) = (0, 1.732).
    Ring 2: 2u, 2v, 2(u-v), u+v, u-v... wait.
    Positions in ring 2 (distance 2 from center in grid steps):
    (3.0, 1.732) = 2u
    (3.0, -1.732) = 2v
    (0, 3.464) = 2(u-v)
    (0, -3.464) = -2(u-v)
    (-3.0, 1.732) = -2u
    (-3.0, -1.732) = -2v
    Also "diagonal" ring 2 positions?
    u+v = (3, 0).
    u-v = (0, 1.732) - this is ring 1.
    
    Let's list Ring 2 candidates (distance 2 steps from center):
    1. (3.0, 0.0)  -> u+v
    2. (1.5, 2.598) -> 2u - (0, 1.732)? No.
       u = (1.5, 0.866). 2u = (3, 1.732).
       (1.5, 2.598) is u + (0, 1.732) = u + (u-v) = 2u - v.
       Is 2u-v a valid grid point? Yes.
       
    Let's just pick 4 symmetric positions from Ring 2.
    (3.0, 0.0)
    (1.5, 2.598)
    (-1.5, 2.598)
    (-3.0, 0.0)
    
    Let's verify they don't overlap with each other or Ring 1.
    (3.0, 0.0) vs (1.5, 0.866) [Ring 1]:
    dx = 1.5, dy = 0.866. dist sqrt(3). Touching. OK.
    
    (3.0, 0.0) vs (1.5, 2.598) [New]:
    dx = 1.5, dy = 2.598.
    dist = sqrt(2.25 + 6.75) = sqrt(9) = 3.0.
    3.0 > sqrt(3). No overlap.
    
    (1.5, 2.598) vs (0, 1.732) [Ring 1]:
    dx = 1.5, dy = 0.866. dist sqrt(3). Touching. OK.
    
    (1.5, 2.598) vs (-1.5, 2.598) [New]:
    dx = 3.0. dist 3.0. No overlap.
    
    (-1.5, 2.598) vs (-1.5, 0.866) [Ring 1]:
    dx = 0, dy = 1.732. dist 1.732 = sqrt(3). Touching. OK.
    
    (-1.5, 2.598) vs (-3.0, 0.0) [New]:
    dx = 1.5, dy = 2.598. dist 3.0. No overlap.
    
    (-3.0, 0.0) vs (-1.5, 0.866) [Ring 1]:
    dx = 1.5, dy = 0.866. dist sqrt(3). Touching. OK.
    
    So this set of 11 is valid and rigid.
    """
    sqrt3 = math.sqrt(3)
    centers = [
        (0.0, 0.0),
        (1.5, sqrt3/2),
        (1.5, -sqrt3/2),
        (0.0, sqrt3),
        (0.0, -sqrt3),
        (-1.5, sqrt3/2),
        (-1.5, -sqrt3/2),
        (3.0, 0.0),
        (1.5, 1.5*sqrt3), # 1.5 * 1.732 = 2.598
        (-1.5, 1.5*sqrt3),
        (-3.0, 0.0)
    ]
    # All angles 0
    angles = [0.0] * 11
    return centers, angles

def _get_cluster_vertices(centers, angles, global_rot_deg, global_tx, global_ty):
    """
    Computes all vertices of the 11 hexagons after applying global rotation and translation.
    Returns a list of (x, y) tuples.
    """
    verts = []
    a_global = math.radians(global_rot_deg)
    cos_g = math.cos(a_global)
    sin_g = math.sin(a_global)
    
    for (cx, cy), ang_deg in zip(centers, angles):
        # Local vertices
        a_local = math.radians(ang_deg)
        for i in range(6):
            theta = a_local + i * math.pi / 3
            vx = cx + 1.0 * math.cos(theta)
            vy = cy + 1.0 * math.sin(theta)
            
            # Global rotation
            gx = vx * cos_g - vy * sin_g
            gy = vx * sin_g + vy * cos_g
            
            # Global translation
            gx += global_tx
            gy += global_ty
            
            verts.append((gx, gy))
            
    return verts

def _min_enclosing_hex_side(verts, outer_rot_deg):
    """
    Given a set of vertices (fixed cluster) and an outer hexagon rotation,
    finds the minimum side length s and center (cx, cy) of a regular hexagon
    with rotation outer_rot_deg that encloses all vertices.
    
    This is a linear programming problem:
    Minimize s
    Subject to:
    For each vertex (x, y) and each of the 3 pairs of parallel sides of the outer hex:
    The projection of (x, y) onto the normal direction n_k must be within [cx - s*apothem, cx + s*apothem]
    where apothem = s * sqrt(3) / 2.
    
    Let R = s * sqrt(3) / 2 be the apothem (distance from center to side).
    The constraints are:
    -R <= dot((x, y) - (cx, cy), n_k) <= R  for k=0,1,2
    
    This implies:
    dot((x, y), n_k) - dot((cx, cy), n_k) <= R
    dot((cx, cy), n_k) - dot((x, y), n_k) <= R
    
    Let u_k = dot((x, y), n_k).
    Let v_k = dot((cx, cy), n_k).
    
    u_k - v_k <= R
    v_k - u_k <= R
    
    This is equivalent to |u_k - v_k| <= R for all vertices x and all k.
    So, R >= max_k ( max_x u_k - v_k, v_k - min_x u_k )
    
    To minimize R (and thus s), for a fixed outer rotation (fixed n_k),
    we can choose cx, cy to minimize the maximum extent.
    
    Let U_k_max = max_x dot((x, y), n_k)
    Let U_k_min = min_x dot((x, y), n_k)
    
    We need to find (cx, cy) such that R is minimized where:
    R >= U_k_max - v_k
    R >= v_k - U_k_min
    
    This means v_k must be in the middle of the range [U_k_min, U_k_max] to minimize the max deviation.
    Specifically, for each k, the optimal v_k is (U_k_max + U_k_min) / 2.
    And the required R_k is (U_k_max - U_k_min) / 2.
    
    So the minimal R for this rotation is max( R_0, R_1, R_2 ).
    And the corresponding optimal center (cx, cy) is determined by:
    dot((cx, cy), n_0) = (U_0_max + U_0_min) / 2
    dot((cx, cy), n_1) = (U_1_max + U_1_min) / 2
    dot((cx, cy), n_2) = (U_2_max + U_2_min) / 2
    
    Note: n_0, n_1, n_2 are not linearly independent (sum is 0). So the center is determined by any 2 of these equations.
    We can solve for cx, cy using the first two.
    
    Returns: (min_side, cx, cy)
    """
    a = math.radians(outer_rot_deg)
    # Normals for the hexagon.
    # Vertices at 0, 60...
    # Sides are perpendicular to 30, 90, 150.
    n_angles = [math.radians(30), math.radians(90), math.radians(150)]
    
    n_vectors = []
    for na in n_angles:
        n_vectors.append((math.cos(a + na), math.sin(a + na)))
        
    # Calculate projections for all vertices
    # verts is list of (x, y)
    # We need max and min projection for each normal
    max_projs = [float('-inf')] * 3
    min_projs = [float('inf')] * 3
    
    for x, y in verts:
        for k in range(3):
            nx, ny = n_vectors[k]
            proj = x * nx + y * ny
            if proj > max_projs[k]:
                max_projs[k] = proj
            if proj < min_projs[k]:
                min_projs[k] = proj
                
    # Calculate required R for each normal
    R_req = []
    for k in range(3):
        R_req.append((max_projs[k] - min_projs[k]) / 2.0)
        
    min_R = max(R_req)
    min_side = min_R * 2.0 / math.sqrt(3)
    
    # Solve for center (cx, cy)
    # v_k = (max_projs[k] + min_projs[k]) / 2
    target_v = [(max_projs[k] + min_projs[k]) / 2.0 for k in range(3)]
    
    # Equations:
    # cx * n0x + cy * n0y = target_v[0]
    # cx * n1x + cy * n1y = target_v[1]
    
    n0x, n0y = n_vectors[0]
    n1x, n1y = n_vectors[1]
    
    det = n0x * n1y - n1x * n0y
    
    if abs(det) < 1e-9:
        # Degenerate, shouldn't happen for hexagon normals
        cx, cy = 0, 0
    else:
        cx = (target_v[0] * n1y - target_v[1] * n0y) / det
        cy = (n0x * target_v[1] - n1x * target_v[0]) / det
        
    return min_side, cx, cy

def _objective_rigid(params):
    """
    params: [global_rot_deg, global_tx, global_ty]
    """
    rot, tx, ty = params
    verts = _get_cluster_vertices(_CENTERS, _ANGLES, rot, tx, ty)
    side, cx, cy = _min_enclosing_hex_side(verts, rot)
    return side

# Pre-compute the rigid cluster
_CENTERS, _ANGLES = _construct_rigid_cluster()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We need to find the optimal global rotation and translation.
    # Translation doesn't affect the size, only the position.
    # So we can just optimize rotation, then calculate optimal translation.
    # But _objective_rigid takes tx, ty.
    # Actually, the size depends ONLY on the rotation of the cluster relative to the container.
    # The container rotation is also a variable.
    # Relative rotation = container_rot - cluster_rot.
    # Since the cluster is rigid, rotating the cluster by -alpha is equivalent to rotating the container by alpha.
    # So we can fix the cluster orientation (0) and optimize the container orientation.
    # And fix the cluster translation to (0,0) for the size calculation,
    # then determine the final container center and cluster translation for the output.
    
    # Let's simplify:
    # 1. Generate vertices of the cluster at (0,0) rotation.
    # 2. Optimize outer_rot to minimize the size.
    # 3. Once optimal outer_rot is found, calculate the required R and center offset.
    # 4. The final output will have the cluster at some (tx, ty) and outer at (cx, cy).
    #    We can set the outer center to (0,0) for simplicity in output?
    #    No, the interface asks for inner, center, side, angle.
    #    "inner rows (x, y, angle_degrees)"
    #    "outer hexagon center (x, y), side length and rotation"
    
    # So:
    # 1. Find best outer_rot.
    # 2. Compute min_side.
    # 3. Compute the optimal center of the outer hexagon relative to the cluster center (0,0).
    #    Let this be (ox, oy).
    # 4. We can place the cluster at (0,0) and the outer hex at (ox, oy).
    #    OR place the outer hex at (0,0) and the cluster at (-ox, -oy).
    #    Let's place the outer hex at (0,0) for cleanliness.
    #    Then the cluster is translated by (-ox, -oy).
    
    # Optimization variables: [outer_rot]
    # We can use a simple grid search or minimize.
    
    # Since the function is periodic and smooth (max of linear functions),
    # it is convex in terms of the rotation?
    # The width of a convex shape as a function of angle is not necessarily convex,
    # but for a hexagon, it is periodic and piecewise smooth.
    # We can use differential_evolution or just a fine grid search.
    # Grid search is deterministic and fast.
    
    best_side = float('inf')
    best_rot = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Grid search for rotation 0 to 60 (symmetry of hexagon)
    # Step 1 degree
    for i in range(61):
        rot_deg = float(i)
        verts = _get_cluster_vertices(_CENTERS, _ANGLES, 0.0, 0.0, 0.0) # Cluster at 0,0
        side, cx, cy = _min_enclosing_hex_side(verts, rot_deg)
        if side < best_side:
            best_side = side
            best_rot = rot_deg
            best_cx = cx
            best_cy = cy
            
    # Refine with minimize around best_rot
    if time.time() < deadline - 10:
        try:
            # Minimize side length
            # We need a function that takes rot and returns side
            def obj_rot(r):
                verts = _get_cluster_vertices(_CENTERS, _ANGLES, 0.0, 0.0, 0.0)
                side, _, _ = _min_enclosing_hex_side(verts, r)
                return side
                
            res = minimize(obj_rot, [best_rot], method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_side:
                best_rot = res.x[0]
                # Recalculate center for this best rot
                verts = _get_cluster_vertices(_CENTERS, _ANGLES, 0.0, 0.0, 0.0)
                best_side, best_cx, best_cy = _min_enclosing_hex_side(verts, best_rot)
        except Exception:
            pass
            
    # Now we have:
    # best_side
    # best_rot (outer rotation)
    # best_cx, best_cy (outer center relative to cluster center at 0,0)
    
    # Output format:
    # inner: list of [x, y, angle]
    # center: (x, y) of outer
    # side: float
    # angle: float (outer rotation)
    
    # We can set outer center to (0,0).
    # Then cluster must be at (-best_cx, -best_cy).
    
    tx = -best_cx
    ty = -best_cy
    
    inner = []
    for i in range(N):
        cx, cy = _CENTERS[i]
        ang = _ANGLES[i]
        # Apply translation
        new_x = cx + tx
        new_y = cy + ty
        inner.append([new_x, new_y, ang])
        
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_rot
    
    return inner, outer_center, outer_side, outer_angle

if __name__ == "__main__":
    r = construct()
    print(r)
