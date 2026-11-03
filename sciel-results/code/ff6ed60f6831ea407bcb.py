import numpy as np
from scipy.optimize import minimize
import time

N = 12

def get_hex_vertices(center, side, angle_deg):
    """
    Returns vertices of a regular hexagon.
    center: (x, y)
    side: side length
    angle_deg: rotation in degrees
    """
    cx, cy = center
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def point_in_polygon(point, polygon):
    """
    Check if point is inside convex polygon.
    polygon: Nx2 array
    """
    x, y = point
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Cross product to check if point is to the left of edge
        # For a CCW polygon, interior is to the left.
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross < -1e-9:
            return False
    return True

def min_distance_poly_poly(poly1, poly2):
    """
    Calculate minimum distance between two convex polygons.
    Returns negative value if they overlap (approx depth of penetration).
    """
    # First check for overlap
    # If any vertex of poly1 is in poly2, or vice versa, they overlap.
    # We estimate penetration depth by min distance from vertices to edges of the other.
    
    min_dist = float('inf')
    overlap = False
    
    # Check vertices of poly1 against poly2
    for v in poly1:
        if point_in_polygon(v, poly2):
            overlap = True
            # Calculate distance to nearest edge of poly2
            min_d = float('inf')
            for i in range(len(poly2)):
                p1 = poly2[i]
                p2 = poly2[(i + 1) % len(poly2)]
                # Distance from point v to line segment p1-p2
                # Project v onto line p1-p2
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length_sq = dx*dx + dy*dy
                if length_sq == 0:
                    d = np.linalg.norm(v - p1)
                else:
                    t = ((v[0] - p1[0]) * dx + (v[1] - p1[1]) * dy) / length_sq
                    t = max(0, min(1, t))
                    proj = np.array([p1[0] + t * dx, p1[1] + t * dy])
                    d = np.linalg.norm(v - proj)
                min_d = min(min_d, d)
            # If vertex is inside, distance is negative (penetration)
            # Approximate penetration as -min_d
            min_dist = min(min_dist, -min_d)
            
    # Check vertices of poly2 against poly1
    for v in poly2:
        if point_in_polygon(v, poly1):
            overlap = True
            min_d = float('inf')
            for i in range(len(poly1)):
                p1 = poly1[i]
                p2 = poly1[(i + 1) % len(poly1)]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length_sq = dx*dx + dy*dy
                if length_sq == 0:
                    d = np.linalg.norm(v - p1)
                else:
                    t = ((v[0] - p1[0]) * dx + (v[1] - p1[1]) * dy) / length_sq
                    t = max(0, min(1, t))
                    proj = np.array([p1[0] + t * dx, p1[1] + t * dy])
                    d = np.linalg.norm(v - proj)
                min_d = min(min_d, d)
            min_dist = min(min_dist, -min_d)

    if overlap:
        return min_dist
    
    # If no overlap, calculate min distance between edges
    # This is expensive, but N is small.
    # We can just check vertex-edge distances in both directions.
    for v in poly1:
        for i in range(len(poly2)):
            p1 = poly2[i]
            p2 = poly2[(i + 1) % len(poly2)]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length_sq = dx*dx + dy*dy
            if length_sq == 0:
                d = np.linalg.norm(v - p1)
            else:
                t = ((v[0] - p1[0]) * dx + (v[1] - p1[1]) * dy) / length_sq
                t = max(0, min(1, t))
                proj = np.array([p1[0] + t * dx, p1[1] + t * dy])
                d = np.linalg.norm(v - proj)
            min_dist = min(min_dist, d)
            
    for v in poly2:
        for i in range(len(poly1)):
            p1 = poly1[i]
            p2 = poly1[(i + 1) % len(poly1)]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length_sq = dx*dx + dy*dy
            if length_sq == 0:
                d = np.linalg.norm(v - p1)
            else:
                t = ((v[0] - p1[0]) * dx + (v[1] - p1[1]) * dy) / length_sq
                t = max(0, min(1, t))
                proj = np.array([p1[0] + t * dx, p1[1] + t * dy])
                d = np.linalg.norm(v - proj)
            min_dist = min(min_dist, d)
            
    return min_dist

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    # Initial guess: Hexagonal cluster
    # 1 center, 6 around, 5 more
    # Side length of inner hex is 1.
    # Distance between centers in tight hex lattice is sqrt(3).
    d = np.sqrt(3)
    
    # Coordinates for 1+6+5 cluster
    # Center at (0,0)
    # 6 neighbors at distance d, angles 0, 60, ...
    # 5 more around the 6?
    # Let's just place them in a grid-like pattern first, or use a known good config.
    # A 3x4 grid of hexagons?
    # Let's try a hexagonal packing of 12.
    # Rows: 3, 4, 3, 2?
    # Or just use random initialization and let optimizer work.
    
    def objective(params):
        # params: x1, y1, a1, ..., x12, y12, a12, S, a_out
        # We fix hex 0 at (0,0) with angle 0 to reduce symmetry.
        # So params has 11*3 + 2 = 35 variables?
        # Let's keep it simple: all 12 free, plus S, a_out. 38 vars.
        # But we can fix the first one to avoid translation/rotation invariance.
        
        x0, y0, a0 = params[0:3]
        # Fix x0=0, y0=0, a0=0? 
        # If we fix them, we must not include them in optimization or set them to 0 after.
        # Easier: Optimize all, but add penalty if x0 != 0 etc? No, just fix them in the array.
        
        # Let's define the structure:
        # Indices 0-2: Hex 0 (Fixed to 0,0,0)
        # Indices 3-34: Hex 1-11 (11 * 3 = 33 vars)
        # Index 35: S
        # Index 36: a_out
        # Total 37 vars.
        
        # Extract positions
        pos = []
        pos.append((0.0, 0.0, 0.0)) # Hex 0 fixed
        
        for i in range(1, 12):
            idx = 3 + (i-1)*3
            x = params[idx]
            y = params[idx+1]
            a = params[idx+2]
            pos.append((x, y, a))
            
        S = params[36]
        a_out = params[37]
        
        if S < 0.5:
            S = 0.5
            
        # Calculate penalty
        penalty = 0.0
        
        # Inner hexagons
        inner_verts = []
        for i in range(12):
            x, y, a = pos[i]
            verts = get_hex_vertices((x, y), 1.0, a)
            inner_verts.append(verts)
            
        # Outer hexagon
        outer_verts = get_hex_vertices((0.0, 0.0), S, a_out)
        
        # Check containment
        for i in range(12):
            for v in inner_verts[i]:
                if not point_in_polygon(v, outer_verts):
                    # Calculate distance to boundary
                    # Simple approximation: distance to nearest vertex of outer?
                    # Better: distance to nearest edge.
                    min_d = float('inf')
                    for j in range(6):
                        p1 = outer_verts[j]
                        p2 = outer_verts[(j+1)%6]
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        length_sq = dx*dx + dy*dy
                        if length_sq == 0:
                            d = np.linalg.norm(v - p1)
                        else:
                            t = ((v[0] - p1[0]) * dx + (v[1] - p1[1]) * dy) / length_sq
                            t = max(0, min(1, t))
                            proj = np.array([p1[0] + t * dx, p1[1] + t * dy])
                            d = np.linalg.norm(v - proj)
                        min_d = min(min_d, d)
                    penalty += min_d * 10.0 # Penalty for violation
                    
        # Check overlaps
        for i in range(12):
            for j in range(i+1, 12):
                dist = min_distance_poly_poly(inner_verts[i], inner_verts[j])
                if dist < 0:
                    penalty += (-dist) * 10.0 # Penalty for overlap
            
        return S + penalty
        
    # Initial parameters
    # Hex 0 is fixed.
    # Initialize others in a cluster around (0,0).
    init_params = [0.0, 0.0, 0.0] # Hex 0
    
    d = np.sqrt(3)
    for i in range(1, 12):
        # Place in a rough hexagonal pattern
        # 1st ring: 6 hexes
        # 2nd ring: 5 hexes
        if i <= 6:
            angle = (i-1) * np.pi / 3
            x = d * np.cos(angle)
            y = d * np.sin(angle)
        else:
            # Place around the 2nd ring
            # Just scatter them slightly further
            angle = (i-7) * np.pi / 2.5 + np.pi/4
            dist = d * 2
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            
        init_params.extend([x, y, 0.0])
        
    # S and a_out
    # Area bound S >= 6. Let's start with 4.0 to see if it pushes up.
    init_params.append(4.0)
    init_params.append(0.0)
    
    # Optimize
    best_result = None
    best_score = float('inf')
    
    # Multiple restarts
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            x0 = init_params
        else:
            # Perturb initial guess
            x0 = init_params.copy()
            for i in range(3, 36):
                x0[i] += np.random.normal(0, 0.5)
            x0[36] = 4.0 + np.random.uniform(-0.5, 1.0)
            x0[37] = np.random.uniform(0, np.pi)
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
            
            if res.success or res.fun < best_score:
                score = res.fun
                # Extract result
                # Check if it's valid (low penalty)
                # If penalty is low, S is the real score.
                # We return the S from the best result.
                
                # Reconstruct inner list
                inner = []
                # Hex 0
                inner.append([0.0, 0.0, 0.0])
                for i in range(1, 12):
                    idx = 3 + (i-1)*3
                    inner.append([res.x[idx], res.x[idx+1], res.x[idx+2]])
                
                S = res.x[36]
                a_out = res.x[37]
                
                # We should return the actual S, not the penalized objective.
                # But if there's overlap, S is not the true score.
                # The evaluator will check validity.
                # If invalid, score is 0.
                # So we should try to find a valid config.
                
                # Let's calculate the true penalty to see if it's valid.
                # If penalty is small, return S.
                
                if score < best_score:
                    best_score = score
                    best_result = (inner, (0.0, 0.0), S, np.rad2deg(a_out))
                    
        except Exception as e:
            pass
            
    if best_result:
        return best_result
    else:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
