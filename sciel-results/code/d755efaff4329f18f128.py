"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    # Optimization: Pre-compute cos/sin for angles if possible, but angle varies.
    # For a unit hexagon, vertices are at distance 1 from center.
    # Angle 0: vertex at (1, 0).
    # Standard math: vertex i at angle i*pi/3 + rotation.
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def sat_max_overlap_depth(v1, v2):
    """
    Calculate the maximum overlap depth between two convex polygons using SAT.
    Returns 0 if they are disjoint (gap > 0).
    Returns positive value if they overlap, representing the minimum distance to separate them.
    v1, v2: lists of (x, y) tuples.
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    min_overlap = float('inf')
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        # Check for gap
        # If max1 < min2, they are separated along this axis.
        # The gap is min2 - max1.
        # If they overlap, the "penetration" is min(max1, max2) - max(min1, min2).
        
        if max1 < min2:
            return 0.0 # Disjoint
        elif max2 < min1:
            return 0.0 # Disjoint
        else:
            # Overlap along this axis
            # The amount they overlap is the intersection length minus the gap?
            # Actually, for separation distance, we want the minimum gap required to separate.
            # If they overlap, the separation distance is negative?
            # Let's define overlap_depth = min(max1, max2) - max(min1, min2)
            # If this is > 0, they overlap by that amount.
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
                
    return min_overlap if min_overlap < float('inf') else 0.0

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    verts_cache = []
    for i in range(n):
        verts_cache.append(get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2])))
        
    for i in range(n):
        vi = verts_cache[i]
        for j in range(i + 1, n):
            vj = verts_cache[j]
            if sat_max_overlap_depth(vi, vj) > eps:
                return False
    return True

def compute_overlap_penalty(inner):
    """
    Compute a smooth penalty for overlaps based on SAT max overlap depth.
    This is exact for convex polygons.
    """
    penalty = 0.0
    n = len(inner)
    verts_cache = []
    for i in range(n):
        verts_cache.append(get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2])))
        
    for i in range(n):
        vi = verts_cache[i]
        for j in range(i + 1, n):
            vj = verts_cache[j]
            depth = sat_max_overlap_depth(vi, vj)
            if depth > 0:
                # Squared penalty to make it smooth and emphasize large overlaps
                penalty += depth * depth
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit (10s buffer)
    
    # 1. Define initial configurations
    # We will use Differential Evolution to search the space.
    # Variables: [x1, y1, a1, ..., xN, yN, aN, cx, cy, angle]
    # Total dims: 3*N + 3
    
    # Bounds for inner hexagons:
    # To fit 11 unit hexagons, the container side will be roughly 3-4 units.
    # Let's set a generous bounding box for inner centers: [-10, 10]
    # Angles: [0, 360]
    
    x_bounds = (-10, 10)
    y_bounds = (-10, 10)
    a_bounds = (0, 360)
    cx_bounds = (-10, 10)
    cy_bounds = (-10, 10)
    oa_bounds = (0, 360)
    
    bounds = []
    for _ in range(N):
        bounds.append(x_bounds)
        bounds.append(y_bounds)
        bounds.append(a_bounds)
    bounds.append(cx_bounds)
    bounds.append(cy_bounds)
    bounds.append(oa_bounds)
    
    def objective(params):
        # Unpack params
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append((x, y, a))
        
        oc = (params[N*3], params[N*3+1])
        oa = params[N*3+2]
        
        side = compute_required_side(inner_list, oc, oa)
        
        # Penalty for overlaps
        # We want to minimize side, but strictly avoid overlap.
        # If overlap exists, we add a large penalty.
        # Using SAT depth ensures we penalize actual geometric overlap, not just center proximity.
        penalty = compute_overlap_penalty(inner_list)
        
        # If penalty is significant, return a large value to guide DE away
        if penalty > 1e-4:
            return 100.0 + 1000.0 * penalty
            
        return side

    # Run Differential Evolution
    # scipy's DE is robust. We limit maxiter and popsize to fit time budget.
    # 120s limit. DE can be slow. Let's try a modest population.
    
    best_result = None
    best_val = float('inf')
    
    # Strategy: Run DE for a fixed time, then polish.
    # We can use a wrapper to stop DE if time runs out, but scipy DE doesn't support time-based stopping directly in older versions easily without a custom callback or just limiting iterations.
    # Let's estimate: 36 dimensions. Popsize 10 -> 360 evals per generation.
    # Each eval involves SAT for 55 pairs of hexagons (11*10/2).
    # 55 pairs * 12 axes * 6 vertices = ~4000 ops per eval.
    # 4000 ops is fast. 1000 evals is < 1 second.
    # So we can afford many iterations.
    
    try:
        # maxiter: 200 generations
        # popsize: 15
        # tol: 1e-4
        # Seed for reproducibility in local dev, but random in prod is fine.
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            init='latinhypercube',
            polish=True, # scipy will run Nelder-Mead on the best result
            maxfev=5000 # Cap function evaluations to prevent runaway
        )
        
        if res.success or res.fun < best_val:
            best_val = res.fun
            best_result = res.x
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Fallback or Refinement:
    # If DE didn't find a good solution, or to improve it, we can do a manual local polish
    # on the best candidate found, or a random start if DE failed completely.
    
    if best_result is None:
        # Fallback to a heuristic lattice if DE fails
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (-1, 2), (-2, 1)
        ][:N]
        
        centers = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))
            
        best_result = np.array([v for c in centers for v in [c[0], c[1], 0.0]] + [0, 0, 0])
        best_val = objective(best_result)
        
    # Final Polish:
    # Take the best result from DE (which includes polish=True, but we can do another tight polish
    # specifically on the side length, assuming no overlap, to be sure).
    
    # Check if the current best is valid (disjoint)
    def check_valid(params):
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append((x, y, a))
        return check_all_disjoint(inner_list, eps=1e-5)
    
    if not check_valid(best_result):
        # If the best result from DE has overlaps, we need to fix them or find a valid one.
        # DE's objective penalized overlaps, so it *should* be valid.
        # But if it's not, we might have a numerical issue or the penalty wasn't high enough.
        # Let's try to minimize the penalty first, then the side.
        
        def objective_penalty(params):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append((x, y, a))
            return compute_overlap_penalty(inner_list)
            
        try:
            # Try to remove overlaps
            res_fix = minimize(objective_penalty, best_result, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_fix.fun < 1e-4:
                best_result = res_fix.x
                best_val = objective(best_result)
            else:
                # If we can't fix it, we are in trouble. 
                # Try a fresh random start with high penalty weight?
                # For now, assume DE found a valid one. If not, return the best effort.
                pass
        except Exception:
            pass

    # Final verification and side calculation
    inner_list = []
    for i in range(N):
        x = best_result[i*3]
        y = best_result[i*3+1]
        a = best_result[i*3+2]
        inner_list.append((x, y, a))
        
    oc = (best_result[N*3], best_result[N*3+1])
    oa = best_result[N*3+2]
    
    # One last local optimization on the outer container and inner positions
    # to minimize side length, strictly enforcing disjointness via penalty if it slips.
    
    def objective_final(params):
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append((x, y, a))
        oc = (params[N*3], params[N*3+1])
        oa = params[N*3+2]
        
        side = compute_required_side(inner_list, oc, oa)
        penalty = compute_overlap_penalty(inner_list)
        
        if penalty > 1e-6:
            return side + 10000.0 * penalty
        return side

    # Use Nelder-Mead for final polish
    try:
        if time.time() < deadline - 5:
            res_final = minimize(objective_final, best_result, method='Nelder-Mead', 
                                 options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_final.fun < best_val:
                best_result = res_final.x
                best_val = res_final.fun
    except Exception:
        pass

    # Extract final values
    inner_list = []
    for i in range(N):
        x = best_result[i*3]
        y = best_result[i*3+1]
        a = best_result[i*3+2]
        inner_list.append((x, y, a))
        
    oc = (best_result[N*3], best_result[N*3+1])
    oa = best_result[N*3+2]
    
    # Calculate final side
    side = compute_required_side(inner_list, oc, oa)
    
    # Ensure we return the correct format
    # inner: list of (x, y, angle_degrees)
    # center: (x, y)
    # side: float
    # angle: float
    
    return inner_list, oc, side, oa

# EVOLVE-BLOCK-END
