"""
Hexagon packing: place N unit regular hexagons inside a regular hexagon,
minimizing the outer side length.

Strategy:
1. Generate diverse initial topologies (hexagonal grid, triangular, random).
2. For each topology, use L-BFGS-B to optimize positions/rotations/outer side
   with a smooth penalty for violations (overlap, containment).
3. Multi-start to escape local minima.
4. Deadline guard to stay within 150s.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 11

# Time budget: 150s wall-clock, use 140s for safety
DEADLINE = 140.0
START_TIME = None

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * math.pi / 3.0
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        verts.append([x, y])
    return np.array(verts)

def _point_in_convex_polygon(p, poly):
    """Check if point p is inside convex polygon poly using cross products."""
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cross < -1e-10:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    
    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using separating axis theorem."""
    polys = [v1, v2]
    for poly in polys:
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i + 1) % len(poly)]
            # Normal to edge
            edge = b - a
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-12:
                continue
            normal = normal / np.linalg.norm(normal)
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True

def _compute_overlap_penalty(v1, v2):
    """Compute smooth penalty for polygon overlap using SAT."""
    polys = [v1, v2]
    min_sep = float('inf')
    
    for poly in polys:
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i + 1) % len(poly)]
            edge = b - a
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            # Separation along this axis
            sep1 = min2 - max1  # positive if v2 is to the right of v1
            sep2 = min1 - max2  # positive if v1 is to the right of v2
            
            # If both separations are negative, they overlap along this axis
            # The penetration depth is the minimum of the negative separations
            if sep1 < 0 and sep2 < 0:
                min_sep = min(min_sep, -min(sep1, sep2))
            elif sep1 < 0:
                min_sep = min(min_sep, sep1)  # sep1 is negative, so -sep1 > 0
            elif sep2 < 0:
                min_sep = min(min_sep, sep2)
    
    if min_sep > -1e-9:
        return 0.0
    # Smooth penalty: quadratic in penetration depth
    return min_sep * min_sep

def _compute_containment_penalty(inner_verts, outer_verts):
    """Compute penalty if inner polygon is not fully inside outer polygon."""
    # Check all vertices of inner are inside outer
    penalty = 0.0
    for v in inner_verts:
        if not _point_in_convex_polygon(v, outer_verts):
            # Compute distance to outer boundary
            min_dist = float('inf')
            for i in range(len(outer_verts)):
                a = outer_verts[i]
                b = outer_verts[(i + 1) % len(outer_verts)]
                # Distance from point to segment
                ab = b - a
                ap = v - a
                t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
                t = max(0, min(1, t))
                proj = a + t * ab
                dist = np.linalg.norm(v - proj)
                min_dist = min(min_dist, dist)
            penalty += min_dist * min_dist
    
    # Also check edges don't cross
    for i in range(len(inner_verts)):
        for j in range(len(outer_verts)):
            if _segments_intersect(inner_verts[i], inner_verts[(i+1)%6], 
                                   outer_verts[j], outer_verts[(j+1)%6]):
                penalty += 1.0
    return penalty

def _objective(params, N):
    """
    Objective function for optimization.
    params: [x1,y1,angle1, ..., xN,yN,angleN, outer_side]
    Returns: outer_side + penalty
    """
    positions = params[:3*N].reshape(N, 3)
    outer_side = params[3*N]
    
    if outer_side < 1.0:
        return 1e10
    
    # Compute vertices
    inner_verts_list = []
    for i in range(N):
        x, y, ang = positions[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        inner_verts_list.append(verts)
    
    outer_verts = _hex_vertices(0, 0, outer_side, 0)
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            penalty += _compute_overlap_penalty(inner_verts_list[i], inner_verts_list[j])
    
    # Penalty for containment
    for i in range(N):
        penalty += _compute_containment_penalty(inner_verts_list[i], outer_verts)
    
    return outer_side + penalty

def _initial_hex_grid(N):
    """Generate hexagonal grid layout."""
    positions = []
    # Hex grid with spacing 2*sqrt(3)/2 = sqrt(3) horizontally, 1.5 vertically
    dx = math.sqrt(3)
    dy = 1.5
    k = 0
    for row in range(-5, 6):
        for col in range(-5, 6):
            if k >= N:
                break
            x = col * dx + (row % 2) * dx/2
            y = row * dy
            positions.append([x, y, 0.0])
            k += 1
        if k >= N:
            break
    return positions[:N]

def _initial_triangular(N):
    """Generate triangular layout."""
    positions = []
    dx = math.sqrt(3)
    dy = 1.5
    k = 0
    for row in range(-5, 6):
        for col in range(-5, 6):
            if k >= N:
                break
            x = col * dx + (row % 2) * dx/2
            y = row * dy
            positions.append([x, y, 0.0])
            k += 1
        if k >= N:
            break
    return positions[:N]

def _initial_random(N, seed=42):
    """Generate random layout."""
    np.random.seed(seed)
    positions = []
    for i in range(N):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        ang = np.random.uniform(0, 60)
        positions.append([x, y, ang])
    return positions

def _initial_circular(N):
    """Generate circular layout."""
    positions = []
    for i in range(N):
        if i == 0:
            positions.append([0, 0, 0.0])
        else:
            ring = int((1 + math.sqrt(1 + 8*(i-1))) / 2)
            angle = 2 * math.pi * i / max(1, (ring * 6))
            r = ring * math.sqrt(3)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            positions.append([x, y, 0.0])
    return positions[:N]

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_params = None
    best_score = float('inf')
    
    # Initial topologies
    topologies = [
        _initial_hex_grid(N),
        _initial_triangular(N),
        _initial_circular(N),
    ]
    
    # Add random topologies
    for seed in range(10):
        topologies.append(_initial_random(N, seed=seed))
    
    for init_pos in topologies:
        if time.time() - START_TIME > DEADLINE:
            break
        
        # Initial outer side estimate
        max_r = max(math.sqrt(x**2 + y**2) for x, y, _ in init_pos)
        outer_side_init = max_r + 2.0
        
        params0 = np.array(init_pos).flatten().tolist() + [outer_side_init]
        
        bounds = []
        for i in range(N):
            bounds.append((-100, 100))
            bounds.append((-100, 100))
            bounds.append((0, 60))
        bounds.append((1.0, 100.0))
        
        try:
            result = minimize(
                _objective,
                params0,
                args=(N,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                score = result.fun
                if score < best_score:
                    best_score = score
                    best_params = result.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract results
    positions = best_params[:3*N].reshape(N, 3)
    outer_side = best_params[3*N]
    
    inner = [[float(x), float(y), float(ang)] for x, y, ang in positions]
    return inner, (0.0, 0.0), float(outer_side), 0.0
