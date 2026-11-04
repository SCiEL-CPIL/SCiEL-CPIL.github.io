import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START

# Precompute base hexagon vertices (unit side length, centered at origin, angle 0)
# Vertices are at angles 0, 60, 120, 180, 240, 300 degrees
_BASE_VERTS = np.array([
    [math.cos(i * math.pi / 3.0), math.sin(i * math.pi / 3.0)] 
    for i in range(6)
])

def _get_hex_vertices(center, angle_rad=0.0):
    """
    Compute the 6 vertices of a unit regular hexagon 
    rotated by angle_rad and translated by center.
    """
    if angle_rad != 0.0:
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = _BASE_VERTS @ R.T
    else:
        verts = _BASE_VERTS
    return verts + center

def _compute_min_side(centers, angles=None):
    """
    Compute the minimum side length S of the outer hexagon 
    (centered at origin, angle 0) 
    that contains all inner hexagons.
    """
    if angles is None:
        angles = np.zeros(N)
        
    # Outer hexagon normals (for side length S, apothem is S*sqrt(3)/2)
    # Normals are at 0, 60, ... 300 degrees
    outer_normals = np.array([
        [math.cos(k * math.pi / 3.0), math.sin(k * math.pi / 3.0)]
        for k in range(6)
    ])
    
    max_extent = 0.0
    for i in range(N):
        verts = _get_hex_vertices(centers[i], angles[i])
        # Project vertices onto outer normals
        projs = verts @ outer_normals.T
        # We need |proj| <= S * sqrt(3) / 2
        # So S >= max(|proj|) * 2 / sqrt(3)
        max_abs = np.max(np.abs(projs))
        if max_abs > max_extent:
            max_extent = max_abs
            
    S = max_extent * 2.0 / math.sqrt(3.0)
    return S

def _overlap_penalty(centers, angles=None, tol=1e-5):
    """
    Approximate overlap penalty based on center distances.
    For aligned hexagons, min center distance is sqrt(3).
    This is a smooth approximation for optimization.
    """
    if angles is None:
        angles = np.zeros(N)
        
    penalty = 0.0
    d_min = math.sqrt(3.0)
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < d_min:
                # Quadratic penalty for violation
                penalty += (d_min - dist) ** 2
    return penalty

def _exact_overlap_check(centers, angles, tol=1e-4):
    """
    Check for exact overlaps using SAT or polygon intersection.
    Returns True if any overlap exists.
    """
    # Get all vertices
    verts_list = [_get_hex_vertices(centers[i], angles[i]) for i in range(N)]
    
    for i in range(N):
        for j in range(i + 1, N):
            v1 = verts_list[i]
            v2 = verts_list[j]
            # Simple check: if any vertex of v1 is inside v2 or vice versa
            # Or use distance
            # For speed, we'll use a simplified SAT check for intersection
            if _polygons_intersect(v1, v2):
                return True
    return False

def _polygons_intersect(v1, v2):
    """
    Check if two convex polygons (6x2 arrays) intersect using SAT.
    """
    def get_normals(verts):
        normals = []
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i+1)%6]
            edge = p2 - p1
            # Normal perpendicular to edge
            n = np.array([-edge[1], edge[0]])
            n = n / np.linalg.norm(n)
            normals.append(n)
        return normals

    n1 = get_normals(v1)
    n2 = get_normals(v2)
    
    for n in n1 + n2:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # If there is a gap, they don't intersect
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def _seed_solution():
    """
    Generate a valid seed solution using hexagonal lattice.
    1 center, 6 in ring 1, 4 in ring 2.
    """
    d = math.sqrt(3)
    centers = []
    angles = []
    
    # Center
    centers.append([0, 0])
    angles.append(0.0)
    
    # Ring 1
    for i in range(6):
        theta = i * math.pi / 3.0
        cx = d * math.cos(theta)
        cy = d * math.sin(theta)
        centers.append([cx, cy])
        angles.append(0.0)
        
    # Ring 2 candidates
    # Axial coordinates (q, r) for ring 2
    ring2_pos = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                # Convert axial to pixel (pointy-top? No, our hex is flat-top? 
                # _BASE_VERTS has vertex at (1,0), so it's flat-top? 
                # Vertex at 0 deg is (1,0). Edge is between 0 and 60.
                # This is a "pointy-top" hexagon? No, vertex at 0 is on the x-axis.
                # Standard axial to pixel for pointy-top:
                # x = size * sqrt(3) * (q + r/2)
                # y = size * 3/2 * r
                # But our "size" is side length 1.
                # Distance between centers of touching hexagons is sqrt(3).
                # So the "radius" in the grid is sqrt(3).
                # Let's just generate positions on the circle of radius 2*sqrt(3) for ring 2?
                # No, ring 2 centers are at distance 2*sqrt(3) from center?
                # Ring 1 distance: sqrt(3).
                # Ring 2 distance: 2*sqrt(3) for the "corners" of the hex grid?
                # Let's use the axial formula with scale sqrt(3).
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_pos.append((x, y))
                
    # We need 4 from these 12.
    # To minimize the bounding box, we should pick 4 that are spread out.
    # The 12 positions are at angles roughly every 30 degrees?
    # Let's sort by angle and pick every 3rd one.
    ring2_pos.sort(key=lambda p: math.atan2(p[1], p[0]))
    selected = [ring2_pos[0], ring2_pos[3], ring2_pos[6], ring2_pos[9]]
    
    for pos in selected:
        centers.append(list(pos))
        angles.append(0.0)
        
    centers = np.array(centers)
    angles = np.array(angles)
    
    return centers, angles

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # Initialize with seed
    centers0, angles0 = _seed_solution()
    
    # Optimize positions only (angles fixed at 0)
    # Variables: x1, y1, ..., x11, y11
    x0 = centers0.flatten()
    
    def objective(params):
        if time.time() > deadline - 2.0:
            return 1e10
        centers = params.reshape(N, 2)
        S = _compute_min_side(centers)
        pen = _overlap_penalty(centers)
        return S + 1000.0 * pen

    # Bounds: roughly within [-5, 5]
    bounds = [(-5, 5)] * (2 * N)
    
    best_S = _compute_min_side(centers0)
    best_centers = centers0.copy()
    
    # Nelder-Mead Optimization
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                objective, x0, 
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000, 'maxfev': 10000}
            )
            if res.fun < best_S:
                new_centers = res.x.reshape(N, 2)
                new_S = _compute_min_side(new_centers)
                # Verify exact overlaps
                if not _exact_overlap_check(new_centers, np.zeros(N)):
                    best_S = new_S
                    best_centers = new_centers
        except Exception:
            pass
            
    # Try Differential Evolution if time permits (more robust global search)
    if time.time() < deadline - 30.0:
        try:
            # DE is slow, use small popsize
            res_de = differential_evolution(
                objective, bounds, 
                seed=42, 
                maxiter=20, 
                popsize=10, 
                tol=1e-4,
                polish=False
            )
            if res_de.fun < best_S:
                new_centers = res_de.x.reshape(N, 2)
                new_S = _compute_min_side(new_centers)
                if not _exact_overlap_check(new_centers, np.zeros(N)):
                    best_S = new_S
                    best_centers = new_centers
        except Exception:
            pass

    # Final refinement with Nelder-Mead on the best found
    if time.time() < deadline - 5.0:
        try:
            x_best = best_centers.flatten()
            res_final = minimize(
                objective, x_best, 
                method='Nelder-Mead',
                options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 2000, 'maxfev': 5000}
            )
            if res_final.fun < best_S:
                new_centers = res_final.x.reshape(N, 2)
                new_S = _compute_min_side(new_centers)
                if not _exact_overlap_check(new_centers, np.zeros(N)):
                    best_S = new_S
                    best_centers = new_centers
        except Exception:
            pass

    # Format output
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], 0.0])
        
    return inner, (0.0, 0.0), best_S, 0.0

# EVOLVE-BLOCK-END
