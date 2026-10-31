"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    def get_hex_vertices(center, angle_deg, side=1.0):
        """Returns list of (x, y) vertices for a regular hexagon."""
        cx, cy = center
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + math.radians(60 * k)
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def hex_support_distance(c1, a1, c2, a2):
        """
        Compute minimum distance between two hexagons using support functions.
        Returns negative value if overlapping, positive if separated.
        Uses the 6 edge normals of each hexagon.
        """
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        
        # Edge normals for hexagon 1 (at angles 30, 90, 150, 210, 270, 330 degrees)
        # relative to its rotation
        min_dist = float('inf')
        
        for k in range(6):
            # Normal direction for hexagon 1
            theta1 = math.radians(30 + 60 * k) + math.radians(a1)
            nx1 = math.cos(theta1)
            ny1 = math.sin(theta1)
            
            # Support of hexagon 1 in direction (nx1, ny1)
            # For a regular hexagon with side 1, the support in normal direction is inradius = sqrt(3)/2
            # But we need to check if the projection exceeds the inradius
            proj1 = dx * nx1 + dy * ny1
            
            # For hexagon 2, we need the support in the opposite direction
            # The support of hexagon 2 in direction (-nx1, -ny1) 
            # is the maximum of dot products of vertices with (-nx1, -ny1)
            # which equals -min of dot products with (nx1, ny1)
            
            # For a regular hexagon, the support function in any direction is:
            # h(u) = r * max_k (cos(u - (30+60k) + a2)) where r = side * sqrt(3)/2
            # Actually, for a regular hexagon, h(u) = inradius / cos(min_angle_diff)
            # where min_angle_diff is the minimum angular difference from u to any edge normal
            
            # Simpler: just compute using vertices
            v2 = get_hex_vertices(c2, a2)
            max_proj2 = max(-v[0]*nx1 - v[1]*ny1 for v in v2)
            # Actually, we want the support in direction -n: max dot(v, -n) = -min(dot(v, n))
            min_proj2 = min(v[0]*nx1 + v[1]*ny1 for v in v2)
            support2_in_neg_n = -min_proj2
            
            # The distance along normal n1 is: proj1 - support2_in_neg_n
            # But we also need to account for the inradius of hex1
            inradius = sqrt3 / 2
            dist_along_n1 = proj1 - support2_in_neg_n - inradius
            
            if dist_along_n1 < min_dist:
                min_dist = dist_along_n1
            
            # Also check the opposite normal
            theta1_op = theta1 + math.pi
            nx1_op = math.cos(theta1_op)
            ny1_op = math.sin(theta1_op)
            
            proj1_op = -dx * nx1_op - dy * ny1_op  # = -proj1
            
            v2_op = get_hex_vertices(c2, a2)
            min_proj2_op = min(v[0]*nx1_op + v[1]*ny1_op for v in v2_op)
            support2_in_neg_n_op = -min_proj2_op
            
            dist_along_n1_op = proj1_op - support2_in_neg_n_op - inradius
            
            if dist_along_n1_op < min_dist:
                min_dist = dist_along_n1_op
        
        return min_dist

    def hexagons_overlap_fast(c1, a1, c2, a2):
        """
        Check if two hexagons overlap using support function distance.
        Returns True if they overlap.
        """
        # Quick bounding box check
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist = math.hypot(dx, dy)
        if dist > 2.0 + 1e-9:
            return False, 0.0
        
        # Use support function distance
        min_dist = hex_support_distance(c1, a1, c2, a2)
        if min_dist < -1e-9:
            return True, min_dist
        return False, 0.0

    def boundary_violation_fast(c, a, s_outer):
        """
        Check if hexagon at c, angle a is inside outer hexagon with side s_outer, center (0,0), rotation 0.
        Returns max violation amount (0 if inside).
        """
        verts = get_hex_vertices(c, a)
        max_viol = 0.0
        inradius_outer = s_outer * sqrt3 / 2
        
        for p in verts:
            px, py = p
            # Check against 6 edges of outer hexagon
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = px * nx + py * ny
                if proj > inradius_outer:
                    viol = proj - inradius_outer
                    if viol > max_viol:
                        max_viol = viol
        return max_viol

    def objective(params):
        # params: 2*N positions + N angles + 1 side
        pos = []
        for i in range(N):
            pos.append([params[2*i], params[2*i+1]])
        angs = [params[2*N + i] for i in range(N)]
        s = params[3*N]
        
        if s < 2.0:
            return 1e6
            
        penalty = 0.0
        
        # Check overlaps with support function distance
        for i in range(N):
            for j in range(i+1, N):
                overlap, dist = hexagons_overlap_fast(pos[i], angs[i], pos[j], angs[j])
                if overlap:
                    # Penalty proportional to overlap depth
                    penalty += 100.0 * (-dist)
        
        # Check boundary
        for i in range(N):
            viol = boundary_violation_fast(pos[i], angs[i], s)
            if viol > 0:
                penalty += 1000.0 * viol
        
        return s + penalty

    def objective_focused(params):
        """Objective that focuses on minimizing side length with stronger penalties."""
        pos = []
        for i in range(N):
            pos.append([params[2*i], params[2*i+1]])
        angs = [params[2*N + i] for i in range(N)]
        s = params[3*N]
        
        if s < 2.0:
            return 1e6
            
        penalty = 0.0
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                overlap, dist = hexagons_overlap_fast(pos[i], angs[i], pos[j], angs[j])
                if overlap:
                    penalty += 500.0 * (-dist)
        
        # Check boundary
        for i in range(N):
            viol = boundary_violation_fast(pos[i], angs[i], s)
            if viol > 0:
                penalty += 5000.0 * viol
        
        return s + penalty

    # Generate multiple initial configurations
    configs = []
    
    # Configuration 1: 3-4-4 grid
    rows = [3, 4, 4]
    y_offset = 0.0
    pos_init = []
    ang_init = []
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_offset
            pos_init.append([x, y])
            ang_init.append(0.0)
        y_offset += 1.5
    xs = [p[0] for p in pos_init]
    ys = [p[1] for p in pos_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in pos_init:
        p[0] -= cx
        p[1] -= cy
    configs.append((pos_init, ang_init))
    
    # Configuration 2: 4-3-4 grid
    rows = [4, 3, 4]
    y_offset = 0.0
    pos_init = []
    ang_init = []
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_offset
            pos_init.append([x, y])
            ang_init.append(0.0)
        y_offset += 1.5
    xs = [p[0] for p in pos_init]
    ys = [p[1] for p in pos_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in pos_init:
        p[0] -= cx
        p[1] -= cy
    configs.append((pos_init, ang_init))
    
    # Configuration 3: 2-3-4-2 grid
    rows = [2, 3, 4, 2]
    y_offset = 0.0
    pos_init = []
    ang_init = []
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_offset
            pos_init.append([x, y])
            ang_init.append(0.0)
        y_offset += 1.5
    xs = [p[0] for p in pos_init]
    ys = [p[1] for p in pos_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in pos_init:
        p[0] -= cx
        p[1] -= cy
    configs.append((pos_init, ang_init))
    
    # Configuration 4: Random initialization
    pos_init = []
    ang_init = []
    random.seed(42)
    for i in range(N):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        a = random.uniform(0, 360)
        pos_init.append([x, y])
        ang_init.append(a)
    configs.append((pos_init, ang_init))
    
    # Run DE for each config with multiple stages
    best_solution = None
    best_score = float('inf')
    
    for idx, (p_init, a_init) in enumerate(configs):
        bounds = []
        for i in range(N):
            bounds.append((-4, 4))
            bounds.append((-4, 4))
        for i in range(N):
            bounds.append((-180, 180))
        bounds.append((2.5, 8.0))
        
        x0 = []
        for i in range(N):
            x0.append(p_init[i][0])
            x0.append(p_init[i][1])
        for i in range(N):
            x0.append(a_init[i])
        
        # Estimate initial S
        max_dist = 0
        for p in p_init:
            max_dist = max(max_dist, math.hypot(p[0], p[1]))
        x0.append(max_dist + 1.0)
        
        # Stage 1: Coarse optimization
        try:
            result1 = differential_evolution(
                objective,
                bounds,
                x0=x0,
                maxiter=50,
                popsize=30,
                tol=1e-4,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=42 + idx,
                polish=False
            )
            
            if result1.fun < best_score:
                best_score = result1.fun
                best_solution = result1.x.copy()
        except Exception as e:
            pass
        
        # Stage 2: Fine optimization from best solution
        if best_solution is not None and best_score < 100:
            try:
                result2 = differential_evolution(
                    objective_focused,
                    bounds,
                    x0=best_solution,
                    maxiter=80,
                    popsize=25,
                    tol=1e-5,
                    mutation=(0.3, 0.8),
                    recombination=0.7,
                    seed=100 + idx,
                    polish=True
                )
                
                if result2.fun < best_score:
                    best_score = result2.fun
                    best_solution = result2.x.copy()
            except Exception as e:
                pass
    
    if best_solution is None:
        # Fallback
        return [[0,0,0] for _ in range(N)], (0,0), 10.0, 0.0

    # Extract solution
    pos = []
    for i in range(N):
        pos.append([best_solution[2*i], best_solution[2*i+1]])
    angs = [best_solution[2*N + i] for i in range(N)]
    s = best_solution[3*N]
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], angs[i]])
        
    return inner, (0.0, 0.0), s, 0.0
# EVOLVE-BLOCK-END
