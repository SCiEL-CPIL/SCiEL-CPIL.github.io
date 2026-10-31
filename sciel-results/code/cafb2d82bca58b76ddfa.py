"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    # Helper to get vertices of a unit hexagon
    def get_hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + math.radians(60 * k)
            vx = cx + math.cos(theta)
            vy = cy + math.sin(theta)
            verts.append((vx, vy))
        return verts

    # Helper to get the 6 outward normals of a unit hexagon
    # The normals point to the midpoints of the edges.
    # For a vertex at angle theta, the edge normal is at theta + 30 degrees?
    # Vertices are at 0, 60, ...
    # Edge between 0 and 60 is at 30.
    # So normals are at 30, 90, 150, 210, 270, 330 relative to the hexagon's orientation.
    def get_hex_normals(angle_deg):
        angle_rad = math.radians(angle_deg)
        normals = []
        for k in range(6):
            theta = angle_rad + math.radians(30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            normals.append((nx, ny))
        return normals

    # Calculate projection range of a hexagon on a given normal (nx, ny)
    def project_hex(cx, cy, angle_deg, nx, ny):
        verts = get_hex_vertices(cx, cy, angle_deg)
        dots = [vx * nx + vy * ny for vx, vy in verts]
        return min(dots), max(dots)

    # Calculate overlap penalty between two hexagons
    def overlap_penalty(c1, a1, c2, a2):
        # Get normals for both hexagons
        normals1 = get_hex_normals(a1)
        normals2 = get_hex_normals(a2)
        
        max_gap = -1e9
        for nx, ny in normals1 + normals2:
            min1, max1 = project_hex(c1[0], c1[1], a1, nx, ny)
            min2, max2 = project_hex(c2[0], c2[1], a2, nx, ny)
            
            # Gap is positive if disjoint, negative if overlapping
            # Gap = min(max1, max2) - max(min1, min2)
            gap = min(max1, max2) - max(min1, min2)
            if gap > max_gap:
                max_gap = gap
                
        # If max_gap > 0, they are disjoint. Penalty is 0.
        # If max_gap < 0, they overlap. Penalty is max_gap^2.
        if max_gap > 0:
            return 0.0
        else:
            return max_gap * max_gap

    # Calculate containment penalty for one hexagon inside outer hexagon
    # Outer hexagon: center (ocx, ocy), side s, angle 0 (flat top? or pointy top?)
    # Let's assume outer hexagon is fixed orientation (angle 0).
    # We need to check if inner hexagon vertices are inside outer hexagon.
    # Outer hexagon half-planes:
    # Normals at 30, 90, 150, 210, 270, 330.
    # Inradius r = s * sqrt(3) / 2.
    # Condition: dot(v - c_outer, n) <= r for all normals n.
    def containment_penalty(cx, cy, angle_deg, ocx, ocy, s):
        verts = get_hex_vertices(cx, cy, angle_deg)
        r = s * sqrt3 / 2
        penalty = 0.0
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            for vx, vy in verts:
                dx = vx - ocx
                dy = vy - ocy
                proj = dx * nx + dy * ny
                if proj > r:
                    penalty += (proj - r) ** 2
        return penalty

    def objective(params):
        n = len(params) // 3
        pos = []
        angs = []
        for i in range(n):
            pos.append([params[3*i], params[3*i+1]])
            angs.append(params[3*i+2])
        
        # We fix outer center to (0,0) to reduce degrees of freedom?
        # The task allows outer center to be optimized, but symmetry suggests (0,0) is optimal for minimal side length.
        # Let's assume outer center is (0,0) and rotation 0.
        # If the inner cluster is off-center, it will take more space. Centering it is optimal.
        ocx = 0.0
        ocy = 0.0
        s = params[-1] # Last param is side length
        
        if s < 1.5: # Minimum possible for 1 hex
            return 1e6

        total_penalty = 0.0
        
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                total_penalty += overlap_penalty(pos[i], angs[i], pos[j], angs[j])
        
        # Containment penalties
        for i in range(n):
            total_penalty += containment_penalty(pos[i][0], pos[i][1], angs[i], ocx, ocy, s)
            
        # Objective: minimize side length + large penalty for violations
        return s + 100.0 * total_penalty

    # Initial heuristic positions
    # 3-4-4 packing
    positions = []
    angles = []
    rows = [3, 4, 4]
    y_offset = 0.0
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_offset
            positions.append([x, y])
            angles.append(0.0)
        y_offset += 1.5
    
    # Center the heuristic
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx
        p[1] -= cy
        
    # Estimate initial side length
    max_dist = 0
    for p in positions:
        max_dist = max(max_dist, math.hypot(p[0], p[1]))
    s_init = max_dist + 1.5

    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
        bounds.append((-30, 30))
    bounds.append((2.0, 10.0)) # Side length
    
    # Multi-start DE
    best_result = None
    best_score = 1e9
    
    # Run DE multiple times with different seeds
    for seed in [42, 123, 456]:
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=seed,
                init='sobol' # Better initial population
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
                
        except Exception:
            continue
            
    if best_result is None:
        # Fallback to heuristic
        s = s_init
        inner = []
        for i in range(N):
            inner.append([positions[i][0], positions[i][1], angles[i]])
        return inner, (0.0, 0.0), s, 0.0

    # Extract solution
    pos = []
    angs = []
    for i in range(N):
        pos.append([best_result[3*i], best_result[3*i+1]])
        angs.append(best_result[3*i+2])
    s = best_result[-1]
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angs[i]])
        
    return inner, (0.0, 0.0), s, 0.0
# EVOLVE-BLOCK-END
