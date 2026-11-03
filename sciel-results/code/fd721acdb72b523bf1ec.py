import numpy as np
import time
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _sdf_hex_point(p, center, side, angle_deg):
    """
    Signed distance from point p to a regular hexagon centered at 'center' 
    with side length 'side' and rotation 'angle_deg'.
    Positive if outside, negative if inside.
    """
    x, y = p
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    
    # Rotate point into hexagon's local frame
    dx = x - cx
    dy = y - cy
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    
    # Hexagon geometry:
    # Apothem a = side * sqrt(3) / 2
    # Circumradius R = side
    # The hexagon is defined by |lx| <= R, |ly| <= a, and |lx| + |ly|/sqrt(3) <= R? No.
    # Let's use the property that a regular hexagon is the intersection of 6 half-planes.
    # In local coords (aligned with flat top/bottom or pointy top?), 
    # standard vertex at angle 0 is (R, 0).
    # Edges are at distances a from center.
    # The normals to the edges are at angles 30, 90, 150, 210, 270, 330 degrees.
    # Let's check the distance to each face.
    # Face normals in local frame:
    # n_k = (cos(pi/6 + k*pi/3), sin(pi/6 + k*pi/3))
    # Distance to face k: d_k = dot(p_local, n_k) - a
    # The SDF is min_k( d_k ) if we are outside? 
    # Actually, for a convex polygon, the SDF is the minimum distance to the boundary.
    # If the point is inside, it's negative.
    # The distance to the boundary for a point inside is -min_k(dot(p, n_k) - a).
    # Wait, if p is center (0,0), dot=0, d_k = -a. min is -a. Distance is a. Correct.
    # If p is outside, e.g. (R+1, 0). 
    # n_0 = (cos30, sin30) = (sqrt(3)/2, 0.5). dot = (R+1)*sqrt(3)/2.
    # d_0 = (R+1)sqrt(3)/2 - Rsqrt(3)/2 = sqrt(3)/2.
    # Is this the true distance? 
    # The closest point on the boundary might be a vertex.
    # A robust SDF for a regular polygon can be computed by finding the closest point on the boundary.
    
    # Let's use a simpler approach: 
    # 1. Check if point is inside.
    # 2. If inside, return -min_dist_to_boundary.
    # 3. If outside, return min_dist_to_boundary.
    
    # To find min_dist_to_boundary, we can check distance to each edge segment.
    # But that's not differentiable.
    
    # Alternative: Use the formula for SDF of a regular polygon.
    # For a hexagon, we can map the point to the first 60-degree sector using symmetry.
    # Symmetries: 
    # |x|, |y| and swap x,y with scaling? No, hexagon has 6-fold symmetry.
    # We can fold the point into the sector [0, 30] degrees.
    
    # 1. Take absolute values? No, that's for squares.
    # 2. Use rotation and reflection.
    # Let's just compute distance to the 6 edges.
    # Edge k connects vertex k and k+1.
    # Vertices in local frame: v_k = R * (cos(k*pi/3), sin(k*pi/3))
    # Edge k is from v_k to v_{k+1}.
    # Midpoint of edge k: m_k = a * (cos(pi/6 + k*pi/3), sin(pi/6 + k*pi/3))
    # Normal n_k = (cos(pi/6 + k*pi/3), sin(pi/6 + k*pi/3))
    
    # Distance from p to line containing edge k:
    # d_line = dot(p, n_k) - a
    # If d_line < 0, point is inside that half-plane.
    # If all d_line < 0, point is inside hexagon.
    # SDF = -min(0, min_k(d_line_k)) ? No.
    # If inside, SDF = -min_k(dot(p, n_k) - a) ? 
    # Let's test center: dot=0, d=-a. min=-a. SDF = -(-a) = a? No, SDF should be -a.
    # So SDF_inside = min_k(dot(p, n_k) - a). This is negative.
    
    # If outside, we need distance to the nearest edge segment or vertex.
    # The "min_k(d_line_k)" gives distance to the infinite line.
    # If the projection of p onto the line falls within the edge segment, this is the distance.
    # Otherwise, it's distance to a vertex.
    
    # For a hexagon, if p is in the sector defined by vertex k and k+1, 
    # the closest feature is either edge k or vertex k or k+1.
    
    # Let's implement the sector logic.
    # 1. Calculate angle of p in local frame.
    # 2. Find sector.
    # 3. Compute distance to the specific boundary parts.
    
    # This is getting complex for a smooth SDF.
    # Let's use a simpler smooth approximation or just use the "min distance to edges" 
    # which is non-smooth but L-BFGS can handle it if we are careful, 
    # or better, use the SDF formula for regular polygons which is smooth.
    
    # SDF for regular polygon with N sides, circumradius R, apothem a:
    # 1. p = (x, y)
    # 2. p = (|x|, |y|)  -- Fold to first quadrant? No, hexagon symmetry is different.
    # 3. For hexagon, we can fold to 60 degree sector.
    #    - If angle > 60, rotate by -60.
    #    - If angle < 0, rotate by +60.
    #    - If y < 0, reflect y.
    #    - If x < y * sqrt(3), swap and scale?
    
    # Let's use the standard "closest point on boundary" approach but vectorized?
    # No, let's stick to the half-plane method for "inside" check and 
    # distance to vertices/edges for "outside".
    
    # Actually, for optimization, a common trick is to use the "soft" distance.
    # But let's try to get the exact SDF.
    
    # Sector folding for hexagon:
    # 1. abs(x), abs(y) is not enough.
    # 2. Rotate p by -angle to align with a vertex?
    
    # Let's use the property:
    # d = min( dist_to_edge_0, ..., dist_to_edge_5 )
    # where dist_to_edge_k is distance to the segment.
    
    # To make it differentiable, we can use a smooth minimum (log-sum-exp) 
    # or just accept the kinks. L-BFGS-B usually handles kinks okay if they aren't too sharp.
    
    # Let's compute distance to each of the 6 edges.
    verts = []
    for k in range(6):
        ang = k * np.pi / 3
        verts.append([side * np.cos(ang), side * np.sin(ang)])
    verts = np.array(verts)
    
    min_dist = np.inf
    inside = True
    
    for k in range(6):
        p1 = verts[k]
        p2 = verts[(k+1)%6]
        # Vector p2 - p1
        v = p2 - p1
        # Vector p - p1
        w = np.array([lx, ly]) - p1
        
        # Projection of w onto v
        c1 = np.dot(w, v)
        if c1 <= 0:
            # Closest is p1
            d = np.linalg.norm(w)
        else:
            c2 = np.dot(v, v)
            if c2 <= c1:
                # Closest is p2
                d = np.linalg.norm(w - v)
            else:
                # Closest is on segment
                t = c1 / c2
                proj = p1 + t * v
                d = np.linalg.norm(np.array([lx, ly]) - proj)
        
        min_dist = min(min_dist, d)
        
        # Check inside using cross product
        # Edge p1 -> p2. Normal is inward.
        # For CCW vertices, inward normal is to the left.
        # Cross(v, w) > 0 means w is to the left of v (inside).
        cross = v[0]*w[1] - v[1]*w[0]
        if cross < 0:
            inside = False
    
    if inside:
        return -min_dist
    else:
        return min_dist

def _sdf_hex_hex(center1, angle1, center2, angle2, side=1.0):
    """
    Approximate SDF between two hexagons.
    We sample points on the boundary of hex1 and find min distance to hex2.
    This is not the true SDF (which is min over all pairs of points), 
    but for disjoint hexagons, the closest points are usually vertices or on edges.
    Sampling vertices is a good approximation.
    """
    verts1 = _hex_vertices(center1, side, angle1)
    min_dist = np.inf
    for v in verts1:
        d = _sdf_hex_point(v, center2, side, angle2)
        if d < min_dist:
            min_dist = d
    # Also check vertices of hex2 against hex1? 
    # Due to symmetry of distance, min(d(p1, H2), d(p2, H1)) is not exactly d(H1, H2) 
    # but if they are disjoint, the closest points are on the boundaries.
    # Checking vertices of both is safer.
    verts2 = _hex_vertices(center2, side, angle2)
    for v in verts2:
        d = _sdf_hex_point(v, center1, side, angle1)
        if d < min_dist:
            min_dist = d
    return min_dist

def _objective(x, outer_angle):
    """
    x: 3N vector.
    """
    params = x.reshape(N, 3)
    centers = params[:, :2]
    angles = params[:, 2]
    
    # 1. Compute outer side length required
    # Outer hexagon centered at 0,0
    # Max projection of any inner vertex onto outer normals
    max_proj = 0.0
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        # Project all inner vertices
        for i in range(N):
            verts = _hex_vertices(centers[i], 1.0, angles[i])
            projs = np.dot(verts, normal)
            max_proj = max(max_proj, np.max(projs))
            
    side_length = max_proj * 2.0 / np.sqrt(3.0)
    
    # 2. Compute collision penalty
    # We want to maximize the minimum distance between any pair of inner hexagons.
    # Penalty = sum of soft penalties for pairs that are too close.
    # Or simply: if min_dist < 0, penalty = -min_dist^2.
    # Let's use a global min distance approach.
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = _sdf_hex_hex(centers[i], angles[i], centers[j], angles[j])
            if d < min_dist:
                min_dist = d
                
    # If min_dist < 0, they overlap.
    # We want to push them apart.
    # Objective = side_length + LAMBDA * max(0, -min_dist)^2
    # To make it smoother, we can use a sum of penalties for all pairs?
    # No, just the worst case is the bottleneck.
    # But max() is not differentiable.
    # Let's use a soft-min or just sum of squares of negative distances.
    # Sum of squares is differentiable.
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _sdf_hex_hex(centers[i], angles[i], centers[j], angles[j])
            if d < 0:
                penalty += d * d
                
    return side_length + 10.0 * penalty

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_params = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    outer_angles = [0.0, 30.0]
    
    # Initial guesses
    initial_guesses = []
    
    # Guess 1: Hexagonal ring
    pts = [[0,0,0]]
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle), 0.0])
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle), 0.0])
        if len(pts) == 12:
            break
    initial_guesses.append(np.array(pts).flatten())
    
    # Guess 2: Random
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.uniform(-3, 3, (N, 3))
        pts[:, 2] = np.random.choice([0, 30, 60], N)
        initial_guesses.append(pts.flatten())
        
    for outer_angle in outer_angles:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        for guess in initial_guesses:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            try:
                res = minimize(
                    _objective,
                    guess,
                    args=(outer_angle,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                x_opt = res.x
                params_opt = x_opt.reshape(N, 3)
                
                inner_hexes = [[params_opt[i][0], params_opt[i][1], params_opt[i][2]] for i in range(N)]
                
                # Compute side length
                side = _objective(x_opt, outer_angle) # This includes penalty, need raw side
                # Recalculate raw side
                max_proj = 0.0
                for k in range(6):
                    angle = np.deg2rad(outer_angle + k * 60)
                    normal = np.array([np.cos(angle), np.sin(angle)])
                    for i in range(N):
                        verts = _hex_vertices(params_opt[i][0:2], 1.0, params_opt[i][2])
                        projs = np.dot(verts, normal)
                        max_proj = max(max_proj, np.max(projs))
                side = max_proj * 2.0 / np.sqrt(3.0)
                
                # Check validity
                # Simple check: min distance > 0
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        d = _sdf_hex_hex(params_opt[i][0:2], params_opt[i][2], params_opt[j][0:2], params_opt[j][2])
                        if d < 1e-5:
                            valid = False
                            break
                    if not valid: break
                
                if valid and side < best_side:
                    best_side = side
                    best_params = inner_hexes
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = outer_angle
                    
            except Exception as e:
                pass
                
    if best_params is not None:
        return (best_params, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

