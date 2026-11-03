"""Rewritten program: SA + Nelder-Mead polish for 12 unit hexagons in a regular hexagon."""
import time
import math
import random
import numpy as np

N = 12

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 130.0  # seconds, safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    verts = []
    for i in range(6):
        ang = math.radians(angle_deg + 60.0 * i)
        x = cx + side * math.cos(ang)
        y = cy + side * math.sin(ang)
        verts.append((x, y))
    return verts

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside regular hexagon centered at (cx, cy)."""
    # Transform to local frame
    dx = px - cx
    dy = py - cy
    ang = -math.radians(angle_deg)
    lx = dx * math.cos(ang) - dy * math.sin(ang)
    ly = dx * math.sin(ang) + dy * math.cos(ang)
    # Regular hexagon with flat top: |lx| <= side, |ly| <= side*sqrt(3)/2, and |lx| + |ly|/sqrt(3) <= side
    # Actually for a regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300:
    # The hexagon is the intersection of 6 half-planes.
    # For unit hexagon (side=1), the apothem is sqrt(3)/2.
    # In local coords, the hexagon is: |x| <= 1, |y| <= sqrt(3)/2, |x| + |y|/sqrt(3) <= 1
    # Wait, let me think more carefully.
    # A regular hexagon with side length s has vertices at (s*cos(k*pi/3), s*sin(k*pi/3)) for k=0..5.
    # The hexagon can be described as: for each of the 6 edges, the point must be on the correct side.
    # Edge k connects vertex k and vertex k+1. The inward normal points toward center.
    # For vertex at angle theta_k = k*pi/3, the edge midpoint is at distance apothem = s*sqrt(3)/2 from center,
    # at angle theta_k + pi/6.
    # The constraint is: dot(point - center, inward_normal_k) <= apothem
    # where inward_normal_k = (cos(theta_k + pi/6), sin(theta_k + pi/6))
    
    apothem = side * math.sqrt(3) / 2.0
    for i in range(6):
        edge_angle = math.radians(angle_deg + 60.0 * i + 30.0)
        nx = math.cos(edge_angle)
        ny = math.sin(edge_angle)
        if lx * nx + ly * ny > apothem:
            return False
    return True

def hexagons_disjoint(hex1, hex2):
    """Check if two unit hexagons are disjoint (no overlap).
    hex1, hex2: (cx, cy, angle_deg)
    """
    cx1, cy1, a1 = hex1
    cx2, cy2, a2 = hex2
    
    # Quick distance check
    dx = cx1 - cx2
    dy = cy1 - cy2
    dist_sq = dx*dx + dy*dy
    if dist_sq > 4.0 + 1e-9:
        return True
    
    # Compute vertices
    verts1 = np.array(hex_vertices(cx1, cy1, 1.0, a1))
    verts2 = np.array(hex_vertices(cx2, cy2, 1.0, a2))
    
    # Check if any vertex of hex1 is inside hex2
    # Transform verts1 to local frame of hex2
    ang2 = -math.radians(a2)
    c2 = math.cos(ang2)
    s2 = math.sin(ang2)
    # Rotation matrix
    R2 = np.array([[c2, -s2], [s2, c2]])
    local_verts1 = (verts1 - np.array([cx2, cy2])) @ R2.T
    
    # Check containment in unit hexagon at origin
    # Conditions: |x| <= 1, |y| <= sqrt(3)/2, |x| + |y|/sqrt(3) <= 1 is for flat top?
    # Let's use the normal form.
    # Normals for unit hexagon at 0 deg:
    # Edge normals at angles 30, 90, 150, 210, 270, 330
    normals = np.array([
        [math.cos(math.radians(30 + 60*i)), math.sin(math.radians(30 + 60*i))]
        for i in range(6)
    ])
    apothem = math.sqrt(3) / 2.0
    
    # dot(local_verts1, normals.T) <= apothem
    dots1 = local_verts1 @ normals.T
    if np.any(dots1 > apothem + 1e-9):
        return False
        
    # Check if any vertex of hex2 is inside hex1
    ang1 = -math.radians(a1)
    c1 = math.cos(ang1)
    s1 = math.sin(ang1)
    R1 = np.array([[c1, -s1], [s1, c1]])
    local_verts2 = (verts2 - np.array([cx1, cy1])) @ R1.T
    
    dots2 = local_verts2 @ normals.T
    if np.any(dots2 > apothem + 1e-9):
        return False
        
    # Check edge intersections
    # If no vertex is inside, they might still cross.
    # We can check if the distance between any edge of hex1 and any edge of hex2 is 0.
    # Or use the separating axis theorem (SAT).
    # For two convex polygons, they are disjoint if there exists a separating axis.
    # The axes to check are the normals of the edges of both polygons.
    # We already checked the 6 normals of hex2 (in local frame of hex2) and 6 normals of hex1 (in local frame of hex1).
    # Wait, the SAT check requires checking projections on all edge normals of both polygons.
    # The check above effectively checks if any vertex of one is "outside" the other along the normals of the other.
    # If no vertex is inside, and no edge crosses, they are disjoint.
    # The vertex-in-polygon check using normals is equivalent to checking if the polygon is entirely on one side of each edge.
    # If hex1 is not inside hex2 and hex2 is not inside hex1, they are either disjoint or overlapping.
    # To distinguish, we need to check if edges intersect.
    
    # Fast edge intersection check using vectorized operations
    # Edges of hex1: (v1[i], v1[i+1])
    # Edges of hex2: (v2[j], v2[j+1])
    
    def segments_intersect_vectorized(e1, e2):
        # e1, e2: (6, 2) arrays of vertices
        # Returns True if any edge intersects
        # p1 = e1[i], p2 = e1[i+1], p3 = e2[j], p4 = e2[j+1]
        
        # Vectorized cross product
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            
        # We can't easily vectorize the "any" intersection without loops or complex broadcasting.
        # Given N=12, the number of pairs is small.
        # Let's stick to a simple loop but optimized.
        
        for i in range(6):
            p1 = e1[i]
            p2 = e1[(i+1)%6]
            for j in range(6):
                p3 = e2[j]
                p4 = e2[(j+1)%6]
                
                # Orientation
                d1 = cross(p3, p4, p1)
                d2 = cross(p3, p4, p2)
                d3 = cross(p1, p2, p3)
                d4 = cross(p1, p2, p4)
                
                if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                   ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                    return True
        return False

    if segments_intersect_vectorized(verts1, verts2):
        return False
        
    return True

def all_disjoint(inner):
    """Check if all inner hexagons are pairwise disjoint."""
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint(inner[i], inner[j]):
                return False
    return True

def min_outer_side(inner):
    """Find the minimum side length of a regular hexagon (any orientation, centered at origin) 
    that contains all inner hexagons. Returns (side, center_x, center_y, angle_deg)."""
    # Collect all vertices
    all_verts = []
    for (cx, cy, ang) in inner:
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_verts.extend(verts)
    
    if not all_verts:
        return 1.0, 0.0, 0.0, 0.0
    
    verts_np = np.array(all_verts)
    n_verts = len(verts_np)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Use a coarse scan for theta, but optimize center continuously using a few iterations
    # The optimal center for a fixed theta is the center of the minimum enclosing hexagon with that orientation.
    # We can approximate this by finding the center that minimizes the max projection.
    
    for theta_deg in range(0, 60, 5): # 0 to 60 is sufficient due to symmetry? No, hexagon has 60 deg symmetry, so 0-60 covers all unique orientations.
        theta = math.radians(theta_deg)
        normals = np.array([
            [math.cos(theta + math.pi/6 + k * math.pi/3), math.sin(theta + math.pi/6 + k * math.pi/3)]
            for k in range(6)
        ])
        
        # Initial center estimate: centroid
        center = np.mean(verts_np, axis=0)
        
        # Iteratively refine center to minimize max projection
        # Gradient of max projection is the normal of the point with max projection
        for _ in range(10):
            # Compute projections
            # verts_np is (N, 2), normals is (6, 2)
            # We want max over k of dot(verts - center, normals[k])
            # This is equivalent to max over k of (dot(verts, normals[k]) - dot(center, normals[k]))
            
            # Efficient way:
            # proj = verts_np @ normals.T # (N, 6)
            # center_proj = center @ normals.T # (6,)
            # max_proj = np.max(proj - center_proj, axis=0) # (6,)
            # max_val = np.max(max_proj)
            
            # To find the gradient, we need the normal k that gives the max.
            # Actually, the function is f(c) = max_{p, k} dot(p-c, n_k) = max_{p,k} (dot(p, n_k) - dot(c, n_k))
            # The subgradient is -n_k for the active constraint.
            
            # Let's just do a simple search around the current center
            # Or use scipy? No, too slow.
            
            # Simple coordinate descent / gradient step
            # Calculate current max value and the normal that achieves it
            max_val = 0.0
            active_normal_idx = 0
            active_point_idx = 0
            
            # Vectorized max calculation
            # proj_p_k = dot(p, n_k)
            proj_p_k = verts_np @ normals.T # (N, 6)
            center_proj_k = center @ normals.T # (6,)
            rel_proj = proj_p_k - center_proj_k # (N, 6)
            
            # Max over all p and k
            flat_rel = rel_proj.flatten()
            max_idx_flat = np.argmax(flat_rel)
            max_val = flat_rel[max_idx_flat]
            
            # Find which k and p
            p_idx = max_idx_flat // 6
            k_idx = max_idx_flat % 6
            
            # Gradient of max_val w.r.t center is -n_k
            # We want to minimize max_val, so we move center in direction -grad = n_k
            # But we need to be careful not to overshoot.
            # A simple step: center += step * n_k
            # How to choose step?
            # The function is convex.
            
            # Let's try a few step sizes
            best_step_side = max_val
            best_step_center = center
            for step in [0.1, 0.05, 0.01]:
                new_center = center + step * normals[k_idx]
                # Recalculate max val for new center
                new_center_proj = new_center @ normals.T
                new_rel_proj = proj_p_k - new_center_proj
                new_max_val = np.max(new_rel_proj)
                if new_max_val < best_step_side:
                    best_step_side = new_max_val
                    best_step_center = new_center
            
            if np.allclose(best_step_center, center):
                break
            center = best_step_center
        
        # Final calculation for this theta
        center_proj_k = center @ normals.T
        rel_proj = proj_p_k - center_proj_k
        max_val = np.max(rel_proj)
        side = (2.0 / math.sqrt(3.0)) * max_val
        
        if side < best_side:
            best_side = side
            best_center = (center[0], center[1])
            best_angle = theta_deg
    
    # Refine around best angle
    for delta in [-2, -1, 0, 1, 2]:
        theta_deg = (best_angle + delta) % 60
        theta = math.radians(theta_deg)
        normals = np.array([
            [math.cos(theta + math.pi/6 + k * math.pi/3), math.sin(theta + math.pi/6 + k * math.pi/3)]
            for k in range(6)
        ])
        
        center = np.array(best_center)
        
        for _ in range(10):
            proj_p_k = verts_np @ normals.T
            center_proj_k = center @ normals.T
            rel_proj = proj_p_k - center_proj_k
            flat_rel = rel_proj.flatten()
            max_idx_flat = np.argmax(flat_rel)
            max_val = flat_rel[max_idx_flat]
            k_idx = max_idx_flat % 6
            
            best_step_side = max_val
            best_step_center = center
            for step in [0.05, 0.01, 0.001]:
                new_center = center + step * normals[k_idx]
                new_center_proj = new_center @ normals.T
                new_rel_proj = proj_p_k - new_center_proj
                new_max_val = np.max(new_rel_proj)
                if new_max_val < best_step_side:
                    best_step_side = new_max_val
                    best_step_center = new_center
            
            if np.allclose(best_step_center, center):
                break
            center = best_step_center
        
        center_proj_k = center @ normals.T
        rel_proj = proj_p_k - center_proj_k
        max_val = np.max(rel_proj)
        side = (2.0 / math.sqrt(3.0)) * max_val
        
        if side < best_side:
            best_side = side
            best_center = (center[0], center[1])
            best_angle = theta_deg
    
    return best_side, best_center[0], best_center[1], best_angle

def fitness(inner):
    """Lower is better. Returns infinity if not all disjoint."""
    if not all_disjoint(inner):
        return float('inf')
    side, cx, cy, ang = min_outer_side(inner)
    return side

# EVOLVE-BLOCK-START
def construct():
    global START_TIME
    START_TIME = time.time()
    
    random.seed(42)
    np.random.seed(42)
    
    # Initial population: hexagonal packing
    def initial_config():
        # Place hexagons in a honeycomb-like pattern
        inner = []
        # 3 rows of 4? Or 4 rows of 3?
        # Let's try a compact arrangement
        # For 12 hexagons, a 3x4 grid in honeycomb
        dx = 1.5  # horizontal spacing for flat-top hexagons
        dy = math.sqrt(3)  # vertical spacing
        
        positions = []
        for row in range(3):
            for col in range(4):
                x = col * dx + (row % 2) * dx/2
                y = row * dy
                positions.append((x, y))
        
        # Center them
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        
        inner = []
        for (x, y) in positions:
            inner.append([x - cx, y - cy, 0.0])
        
        return inner
    
    best_inner = initial_config()
    best_score = fitness(best_inner)
    if math.isinf(best_score):
        # Try a different initial config
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_score = fitness(best_inner)
    
    current_inner = [list(x) for x in best_inner]
    current_score = best_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    max_iters = 50000
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Generate perturbation
        new_inner = [list(x) for x in current_inner]
        
        # Perturb 1-3 random hexagons
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            idx = random.randint(0, N - 1)
            new_inner[idx][0] += random.gauss(0, 0.3)
            new_inner[idx][1] += random.gauss(0, 0.3)
            new_inner[idx][2] += random.gauss(0, 15.0)
            new_inner[idx][2] %= 60.0  # Hexagon has 60-degree symmetry
        
        new_score = fitness(new_inner)
        
        # Acceptance criterion
        if new_score < current_score:
            current_inner = new_inner
            current_score = new_score
            if current_score < best_score:
                best_inner = [list(x) for x in current_inner]
                best_score = current_score
        elif random.random() < math.exp(-(new_score - current_score) / T):
            current_inner = new_inner
            current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Nelder-Mead polish on best solution
    if time.time() - START_TIME < TIME_LIMIT - 20:
        best_inner = nelder_mead_polish(best_inner, best_score)
    
    side, cx, cy, ang = min_outer_side(best_inner)
    return best_inner, (cx, cy), side, ang

def nelder_mead_polish(inner, score):
    """Polish using Nelder-Mead on the flattened parameters."""
    # Flatten: 3 params per hexagon (x, y, angle)
    x0 = []
    for h in inner:
        x0.extend([h[0], h[1], h[2]])
    x0 = np.array(x0, dtype=float)
    
    def objective(x):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            return float('inf')
        new_inner = []
        for i in range(N):
            new_inner.append([x[3*i], x[3*i+1], x[3*i+2] % 60.0])
        return fitness(new_inner)
    
    # Use scipy's minimize if available, otherwise manual Nelder-Mead
    try:
        from scipy.optimize import minimize
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.fun < score:
            best_inner = []
            x = result.x
            for i in range(N):
                best_inner.append([x[3*i], x[3*i+1], x[3*i+2] % 60.0])
            return best_inner
    except:
        pass
    
    return inner
# EVOLVE-BLOCK-END
