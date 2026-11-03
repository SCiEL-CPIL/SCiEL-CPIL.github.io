"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
DEADLINE = time.time() + 130  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
    angles = a + np.arange(6) * np.pi / 3.0
    verts_x = cx + side * np.cos(angles)
    verts_y = cy + side * np.sin(angles)
    return np.column_stack((verts_x, verts_y))

def polygon_area_signed(poly):
    """Signed area of polygon given as (n,2) array."""
    n = len(poly)
    if n < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

def polygons_overlap(p1, p2):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    n1, n2 = len(p1), len(p2)
    # Get edges and their normals
    for i in range(n1):
        ax = p1[i]
        bx = p1[(i + 1) % n1]
        ex = bx - ax
        # Normal perpendicular to edge
        nx, ny = -ex[1], ex[0]
        # Project both polygons onto this axis
        proj1 = p1 @ np.array([nx, ny])
        proj2 = p2 @ np.array([nx, ny])
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return False
    for i in range(n2):
        ax = p2[i]
        bx = p2[(i + 1) % n2]
        ex = bx - ax
        nx, ny = -ex[1], ex[0]
        proj1 = p1 @ np.array([nx, ny])
        proj2 = p2 @ np.array([nx, ny])
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return False
    return True

def hex_in_hex(inner_verts, outer_verts):
    """Check if all vertices of inner polygon are inside outer polygon."""
    # Use point-in-polygon test for each vertex of inner
    n_out = len(outer_verts)
    for v in inner_verts:
        inside = False
        j = n_out - 1
        for i in range(n_out):
            if ((outer_verts[i, 1] > v[1]) != (outer_verts[j, 1] > v[1])):
                x_intersect = (outer_verts[j, 0] - outer_verts[i, 0]) * (v[1] - outer_verts[i, 1]) / (outer_verts[j, 1] - outer_verts[i, 1]) + outer_verts[i, 0]
                if v[0] < x_intersect:
                    inside = not inside
            j = i
        if not inside:
            return False
    return True

def objective(params, outer_center, outer_side, outer_angle):
    """
    Objective: minimize max distance from outer center to any vertex of any inner hex,
    plus penalty if any constraint is violated.
    params: 12 * 3 = 36 values: (x1, y1, a1, x2, y2, a2, ...)
    """
    inner_centers = params[:2*N].reshape(N, 2)
    inner_angles = params[2*N:]
    
    # Compute outer hexagon vertices
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_angle, outer_side)
    
    max_dist_sq = 0.0
    penalty = 0.0
    
    # Check each inner hex
    inner_polys = []
    for i in range(N):
        cx, cy = inner_centers[i]
        ang = inner_angles[i]
        verts = hex_vertices(cx, cy, ang, 1.0)
        inner_polys.append(verts)
        
        # Check containment: all inner vertices must be inside outer
        for v in verts:
            # Point in polygon test
            inside = False
            j = len(outer_verts) - 1
            for k in range(len(outer_verts)):
                if ((outer_verts[k, 1] > v[1]) != (outer_verts[j, 1] > v[1])):
                    x_int = (outer_verts[j, 0] - outer_verts[k, 0]) * (v[1] - outer_verts[k, 1]) / (outer_verts[j, 1] - outer_verts[k, 1]) + outer_verts[k, 0]
                    if v[0] < x_int:
                        inside = not inside
                j = k
            if not inside:
                penalty += 10.0
                # Distance to boundary approximation
                for k in range(len(outer_verts)):
                    ax = outer_verts[k]
                    bx = outer_verts[(k+1) % len(outer_verts)]
                    # Distance from point to line segment
                    ab = bx - ax
                    t = np.dot(v - ax, ab) / np.dot(ab, ab)
                    t = max(0, min(1, t))
                    closest = ax + t * ab
                    d = np.linalg.norm(v - closest)
                    if d < 0.1:
                        penalty += d * 50.0
    
    # Check pairwise disjointness
    for i in range(N):
        for j in range(i+1, N):
            if polygons_overlap(inner_polys[i], inner_polys[j]):
                penalty += 10.0
    
    if penalty > 0:
        return penalty
    
    # Max distance from outer center to any inner vertex
    for i in range(N):
        for v in inner_polys[i]:
            d_sq = (v[0] - outer_center[0])**2 + (v[1] - outer_center[1])**2
            if d_sq > max_dist_sq:
                max_dist_sq = d_sq
    return np.sqrt(max_dist_sq)

def construct():
    start_time = time.time()
    
    # Outer hexagon centered at origin, angle 0, side length to be determined
    # We'll search over inner positions and compute required outer side
    
    # Strategy: 
    # 1. Use differential evolution to find good inner configurations
    # 2. For each configuration, compute the minimum outer hex side that contains them
    
    def compute_min_outer_side(params):
        """Given inner hex config, find min outer hex side (centered at origin, angle 0)."""
        inner_centers = params[:2*N].reshape(N, 2)
        inner_angles = params[2*N:]
        
        inner_polys = [hex_vertices(inner_centers[i,0], inner_centers[i,1], inner_angles[i], 1.0) for i in range(N)]
        
        # Check pairwise disjointness
        for i in range(N):
            for j in range(i+1, N):
                if polygons_overlap(inner_polys[i], inner_polys[j]):
                    return 1e10
        
        # Binary search on outer side length
        # Outer hex with side s has apothem = s * sqrt(3)/2
        # Need to find min s such that all inner verts are inside
        
        lo, hi = 1.0, 20.0
        for _ in range(40):
            mid = (lo + hi) / 2
            outer_verts = hex_vertices(0, 0, 0, mid)
            ok = True
            for poly in inner_polys:
                for v in poly:
                    inside = False
                    j = len(outer_verts) - 1
                    for k in range(len(outer_verts)):
                        if ((outer_verts[k, 1] > v[1]) != (outer_verts[j, 1] > v[1])):
                            x_int = (outer_verts[j, 0] - outer_verts[k, 0]) * (v[1] - outer_verts[k, 1]) / (outer_verts[j, 1] - outer_verts[k, 1]) + outer_verts[k, 0]
                            if v[0] < x_int:
                                inside = not inside
                        j = k
                    if not inside:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                hi = mid
            else:
                lo = mid
        return hi
    
    # Initial best
    best_params = None
    best_side = 1e10
    
    # Differential evolution over 36 parameters
    # Bounds: centers in [-5,5], angles in [0, 360]
    bounds = [(-5, 5)] * 2*N + [(0, 360)] * N
    
    try:
        # First pass: coarse DE
        result = differential_evolution(
            lambda p: compute_min_outer_side(p),
            bounds,
            maxiter=200,
            tol=0.01,
            popsize=15,
            seed=42,
            polish=False,
            workers=1
        )
        if time.time() > DEADLINE - 5:
            # Return trivial solution
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
        
        side = result.fun
        if result.fun < best_side:
            best_side = result.fun
            best_params = result.x.copy()
        
        # Second pass: local refinement
        if time.time() < DEADLINE - 10:
            local_result = minimize(
                lambda p: compute_min_outer_side(p),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            side2 = local_result.fun
            if side2 < best_side:
                best_side = side2
                best_params = local_result.x.copy()
        
        # Extract inner configuration
        inner_centers = best_params[:2*N].reshape(N, 2)
        inner_angles = best_params[2*N:]
        inner = [[inner_centers[i,0], inner_centers[i,1], inner_angles[i]] for i in range(N)]
        
        return inner, (0.0, 0.0), best_side, 0.0
    
    except Exception as e:
        # Fallback to trivial solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
