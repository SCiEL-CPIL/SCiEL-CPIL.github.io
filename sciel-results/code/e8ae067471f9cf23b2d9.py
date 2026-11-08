import numpy as np
import itertools
import time

N = 11

def _hexagon_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given angle (degrees) and side length."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _min_enclosing_hexagon(vertices):
    """
    Find the minimum side length S and center (cx, cy) of a regular hexagon 
    that encloses the given vertices.
    Returns (S, cx, cy, phi_deg).
    """
    best_S = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    best_phi = 0.0
    
    # The optimal rotation will be within 0 to 60 degrees.
    # We can sample angles finely.
    # For a fixed angle phi, the minimal enclosing hexagon center and size can be found.
    # The width of the set of points in direction n is W = max(p.dot(n)) - min(p.dot(n)).
    # The apothem A must be at least W/2 for all 6 normal directions.
    # So A(phi) = 0.5 * max_k (W_k(phi)).
    # S = 2 * A / sqrt(3).
    
    # To find the center (cx, cy):
    # For the direction k that determines the max width, the center should be the midpoint
    # of the min and max projections in that direction? 
    # Actually, for a regular hexagon, the constraints are coupled.
    # However, if we just want to minimize S, we can find the optimal center for a given phi 
    # by noting that the hexagon is centrally symmetric. 
    # The center (cx, cy) should be such that the max projection in each of the 6 directions 
    # and the min projection in each of the 6 directions are balanced.
    # Specifically, let P_max_k = max_i (p_i . n_k) and P_min_k = min_i (p_i . n_k).
    # The center projection c_k = cx . n_k.
    # We need c_k + A >= P_max_k and -c_k + A >= -P_min_k (which is c_k - A <= P_min_k).
    # So A >= c_k - P_min_k and A >= P_max_k - c_k.
    # To minimize A, we set c_k = (P_max_k + P_min_k) / 2 and A_k = (P_max_k - P_min_k) / 2.
    # The global A is max_k A_k.
    # The center (cx, cy) is determined by the two independent linear equations from the 
    # directions that are "tight". If multiple directions are tight, we might have a system.
    # But actually, the center that minimizes the max width is the centroid of the points 
    # only if the shape is a circle. For a hexagon, it's more complex.
    # However, a known result is that the minimal enclosing regular polygon can be found 
    # by rotating and taking the center as the average of the "extreme" points?
    # Let's use a simpler approach: 
    # For a given phi, the optimal center (cx, cy) is the solution to minimizing 
    # max_k (P_max_k(cx,cy) - P_min_k(cx,cy)) / 2.
    # Note that P_max_k(cx,cy) = max_i ((p_i - c) . n_k) = max_i (p_i . n_k) - c . n_k.
    # P_min_k(cx,cy) = min_i ((p_i - c) . n_k) = min_i (p_i . n_k) - c . n_k.
    # So Width_k(c) = P_max_k - P_min_k = (max_i p_i . n_k) - (min_i p_i . n_k). 
    # Wait, the width of the *set of points* is independent of the center c!
    # The width of the set of points in direction n is fixed.
    # So A_min(phi) = 0.5 * max_k (Width_k(phi)).
    # This gives the minimal *apothem* required for the hexagon to contain the points 
    # *if the center is chosen optimally*.
    # What is the optimal center?
    # The center c must satisfy:
    # c . n_k <= P_max_k - A
    # c . n_k >= P_min_k + A
    # So for each k, c . n_k must be in [P_min_k + A, P_max_k - A].
    # If we pick A = A_min, there is at least one k where the interval is a single point 
    # (the tight constraint). If there are two such independent directions, c is fixed.
    # If there is only one, c is not unique, but any c in the feasible region works.
    # For the purpose of returning a valid center, we can pick the centroid of the points 
    # or solve for c using the tight constraints.
    # A safe bet: The center of the bounding box of the projections?
    # Actually, the center of the minimal enclosing hexagon is the centroid of the 
    # "tight" vertices of the convex hull of the points.
    # Simpler heuristic that works well: The center is the average of the points 
    # that are furthest in the tight directions.
    # Or even simpler: For the optimal phi, the center (cx, cy) can be found by 
    # taking the midpoint of the min and max points in the two tightest directions.
    
    # Let's refine:
    # 1. Sweep phi.
    # 2. Calculate Widths W_k(phi).
    # 3. A = 0.5 * max(W_k).
    # 4. S = 2 * A / sqrt(3).
    # 5. Determine center.
    #    The constraints on c are:
    #    c . n_k >= P_min_k + A
    #    c . n_k <= P_max_k - A
    #    Let L_k = P_min_k + A and U_k = P_max_k - A.
    #    We need c . n_k in [L_k, U_k] for all k.
    #    Since A is the max width/2, for the tightest direction k*, U_k* = L_k*.
    #    So c . n_k* is fixed.
    #    If there's another tight direction m*, c . n_m* is fixed.
    #    We can solve for c using these equations.
    #    If only one is tight, we can pick c to be the centroid of the points? 
    #    No, the centroid might not satisfy the other inequalities if A is minimal.
    #    Actually, if A is minimal, the feasible region for c is a single point or a line segment.
    #    Let's find c by solving the linear system for the two tightest directions.
    #    If only one is tight, we can pick any c on the line c . n_k* = L_k* that satisfies others.
    #    A good candidate is the centroid of the points projected onto the perpendicular?
    #    Let's just use the centroid of the vertices of the convex hull?
    #    Actually, for regular hexagons, the optimal center is often the centroid of the 
    #    set of points if the set is "balanced".
    #    Let's try: c = centroid of points. Check if it's valid. If not, adjust.
    #    But for lattice packings, the centroid is usually the center of symmetry, which is optimal.
    
    # Let's assume the center is the centroid of the input vertices for the optimal phi.
    # This is true for centrally symmetric sets. Our sets (lattice subsets) might not be symmetric,
    # but the minimal enclosing hexagon center will be close to the centroid.
    # To be safe, we can calculate c explicitly.
    
    for phi_deg in np.linspace(0, 60, 121):
        a = np.radians(phi_deg)
        # Normals at 30, 90, ... + phi
        normals = []
        for k in range(6):
            theta = a + np.radians(30 + k * 60)
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals) # 6x2
        
        # Project points onto normals
        # vertices is Mx2
        projections = vertices @ normals.T # Mx6
        
        max_proj = np.max(projections, axis=0) # 6
        min_proj = np.min(projections, axis=0) # 6
        
        widths = max_proj - min_proj
        max_width = np.max(widths)
        A = 0.5 * max_width
        S = 2.0 * A / np.sqrt(3)
        
        if S < best_S:
            best_S = S
            best_phi = phi_deg
            
            # Find center
            # Tight constraints: where widths is close to max_width
            tight_idx = np.where(np.abs(widths - max_width) < 1e-6)[0]
            
            # We need c such that c . n_k in [min_proj[k] + A, max_proj[k] - A]
            # For tight k, the interval is a point: L_k = U_k = (min_proj[k] + max_proj[k]) / 2
            # Let's solve for c using the first two tight constraints if available.
            
            if len(tight_idx) >= 2:
                # Take first two
                k1, k2 = tight_idx[0], tight_idx[1]
                n1 = normals[k1]
                n2 = normals[k2]
                L1 = (min_proj[k1] + max_proj[k1]) / 2
                L2 = (min_proj[k2] + max_proj[k2]) / 2
                
                # Solve:
                # c_x n1_x + c_y n1_y = L1
                # c_x n2_x + c_y n2_y = L2
                A_mat = np.array([n1, n2])
                b_vec = np.array([L1, L2])
                try:
                    c = np.linalg.solve(A_mat, b_vec)
                    best_cx, best_cy = c[0], c[1]
                except np.linalg.LinAlgError:
                    # Fallback to centroid
                    best_cx, best_cy = np.mean(vertices, axis=0)
            elif len(tight_idx) == 1:
                k1 = tight_idx[0]
                n1 = normals[k1]
                L1 = (min_proj[k1] + max_proj[k1]) / 2
                # c . n1 = L1. 
                # We can pick c to be the projection of the centroid onto the line c . n1 = L1?
                # Or just use centroid and check.
                centroid = np.mean(vertices, axis=0)
                # Project centroid to the line?
                # c = centroid + t * n_perp? No, c must satisfy c . n1 = L1.
                # Let c = L1 * n1 + t * n_perp.
                # To minimize distance to centroid, we project centroid onto the line.
                # t = (centroid - L1*n1) . n_perp
                # n_perp = [-n1[1], n1[0]]
                n_perp = np.array([-n1[1], n1[0]])
                t = np.dot(centroid - L1 * n1, n_perp)
                c = L1 * n1 + t * n_perp
                best_cx, best_cy = c[0], c[1]
            else:
                best_cx, best_cy = np.mean(vertices, axis=0)
                
    return best_S, best_cx, best_cy, best_phi

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    # Generate lattice centers
    # Ring 0
    centers = [(0.0, 0.0)]
    
    # Ring 1 (6 hexes)
    d = np.sqrt(3)
    ring1 = []
    for k in range(6):
        angle = k * np.pi / 3
        ring1.append((d * np.cos(angle), d * np.sin(angle)))
    centers.extend(ring1)
    
    # Ring 2 (12 hexes)
    ring2 = []
    # 6 at distance 2d, angles 0, 60, ...
    for k in range(6):
        angle = k * np.pi / 3
        ring2.append((2 * d * np.cos(angle), 2 * d * np.sin(angle)))
    # 6 at distance d*sqrt(3) = 3? No.
    # Distance for "side" of ring 2:
    # The next ring out from Ring 1.
    # Ring 1 centers are at (d, 0), (d/2, d*sqrt(3)/2), etc.
    # The "corners" of the hexagon of side 2 (in units of hex side 1) are at distance 2d.
    # The "edges" of the hexagon of side 2 are at distance ...
    # Let's just generate all lattice points within radius 2.5 * d.
    # Lattice vectors: v1 = (d, 0), v2 = (d/2, d*sqrt(3)/2)
    # Points: i*v1 + j*v2
    lattice_points = set()
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * d + j * d / 2
            y = j * d * np.sqrt(3) / 2
            # Exclude center and ring 1
            if (x, y) != (0.0, 0.0) and (x, y) not in ring1:
                # Check if it's in ring 2 (distance approx 2d or d*sqrt(3)?)
                # Actually, let's just collect all points with norm < 2.5*d
                if np.sqrt(x*x + y*y) < 2.5 * d:
                    lattice_points.add((round(x, 10), round(y, 10)))
    
    # We need 4 more centers to make 11.
    # Current: 1 + 6 = 7.
    # Candidates: lattice_points (which includes Ring 2 and maybe parts of Ring 3 if any).
    # Let's ensure we only pick from the immediate next ring to keep compactness.
    # The points in `lattice_points` that are not in Ring 1 are the candidates.
    # Let's verify their distances.
    # Ring 2 points should be at distance 2d (6 points) and d*sqrt(3) (6 points)?
    # d = sqrt(3).
    # 2d = 2*sqrt(3) ~ 3.46
    # d*sqrt(3) = 3.
    # Let's just use the 12 closest points to the origin from lattice_points.
    candidates = sorted(lattice_points, key=lambda p: p[0]**2 + p[1]**2)
    # Take the first 12 (or however many are in the second shell)
    # There should be exactly 12 in the second shell of a hex lattice.
    ring2_candidates = candidates[:12]
    
    best_S = float('inf')
    best_config = None
    
    # Enumerate all 4-subsets of ring2_candidates
    # 12 choose 4 = 495
    for subset in itertools.combinations(range(12), 4):
        if time.time() > deadline - 5.0:
            break
            
        # Construct the 11 centers
        current_centers = [(0.0, 0.0)] + ring1 + [ring2_candidates[i] for i in subset]
        
        # Generate vertices for all 11 hexagons (angle 0)
        all_verts = []
        for cx, cy in current_centers:
            v = _hexagon_vertices(cx, cy, 0.0, 1.0)
            all_verts.append(v)
        
        vertices = np.vstack(all_verts)
        
        # Find min enclosing hexagon
        S, cx, cy, phi = _min_enclosing_hexagon(vertices)
        
        if S < best_S:
            best_S = S
            best_config = {
                'centers': current_centers,
                'cx': cx,
                'cy': cy,
                'S': S,
                'phi': phi
            }
            
    if best_config is None:
        # Fallback: just return the 7-hex flower, padded with 4 dummy ones far away? 
        # No, must be valid. 
        # If no valid config found (shouldn't happen), return a large hexagon.
        centers = [(0.0, 0.0)] + ring1 + ring2_candidates[:4]
        all_verts = []
        for cx, cy in centers:
            v = _hexagon_vertices(cx, cy, 0.0, 1.0)
            all_verts.append(v)
        vertices = np.vstack(all_verts)
        S, cx, cy, phi = _min_enclosing_hexagon(vertices)
        best_config = {
            'centers': centers,
            'cx': cx,
            'cy': cy,
            'S': S,
            'phi': phi
        }
        
    inner = [[c[0], c[1], 0.0] for c in best_config['centers']]
    return inner, (best_config['cx'], best_config['cy']), best_config['S'], best_config['phi']
