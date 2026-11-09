"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (evaluator limit 120s, keep 10s buffer)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Separating Axis Theorem for convex polygons
    # Returns (is_disjoint, min_penetration_depth, contact_normal)
    # If disjoint, depth is 0 and normal is None.
    def sat_check(poly1, poly2):
        # Get axes (normals to edges) from both polygons
        axes = []
        for poly in (poly1, poly2):
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                norm = math.sqrt(dx*dx + dy*dy)
                if norm == 0:
                    continue
                nx = -dy / norm
                ny = dx / norm
                axes.append((nx, ny))
        
        min_pen = float('inf')
        best_axis = None
        
        for axis in axes:
            # Project poly1
            min1 = float('inf')
            max1 = -float('inf')
            for p in poly1:
                proj = p[0]*axis[0] + p[1]*axis[1]
                if proj < min1: min1 = proj
                if proj > max1: max1 = proj
                
            # Project poly2
            min2 = float('inf')
            max2 = -float('inf')
            for p in poly2:
                proj = p[0]*axis[0] + p[1]*axis[1]
                if proj < min2: min2 = proj
                if proj > max2: max2 = proj
                
            # Check for separation
            if max1 < min2 or max2 < min1:
                return True, 0.0, None # Disjoint
                
            # Calculate overlap on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_pen:
                min_pen = overlap
                best_axis = axis
        
        return False, min_pen, best_axis

    # Helper: Calculate minimum outer hexagon side length containing all given hexagons
    # Outer hexagon is centered at (0,0) with rotation angle outer_angle_deg
    def calc_outer_side(hexagons, outer_angle_deg):
        # Outer hexagon is regular.
        # The condition for a point (x,y) to be inside a regular hexagon of side S
        # centered at origin with "flat-top" orientation (vertices at 0, 60, ...) rotated by alpha:
        # The hexagon is the intersection of 6 half-planes.
        # The distance from origin to each side (apothem) is A = S * SQRT3 / 2.
        # The normal vectors to the sides are at angles 30, 90, 150, 210, 270, 330 + alpha.
        
        base_angles = [30, 90, 150, 210, 270, 330]
        max_proj = 0.0
        
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for a in base_angles:
                    rad = math.radians(a + outer_angle_deg)
                    nx = math.cos(rad)
                    ny = math.sin(rad)
                    # Distance from origin to the line containing the side
                    # The point must satisfy nx*x + ny*y <= A
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        
        # S = 2 * A / SQRT3
        return 2.0 * max_proj / SQRT3

    # Generate initial compact lattice configuration
    # Hexagonal tiling coordinates (axial)
    # (0,0) center
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) neighbors
    # (2,0), (2,-1), (2,1), (1,1), (1,-2) etc for next layer
    # We need 11 hexagons.
    # Layer 0: 1
    # Layer 1: 6
    # Layer 2: 12 -> we take 4 more to make 11? No, 1+6=7. Need 4 more.
    # Let's pick 4 from the next layer to keep it compact.
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, 1), (0, 2) # 4 extra from layer 2
    ]
    
    def axial_to_cartesian(q, r, spacing=1.7320508 + 0.1): # SQRT3 is touching, add small gap
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Initial positions
    init_positions = []
    for q, r in axial_coords:
        x, y = axial_to_cartesian(q, r)
        init_positions.append([x, y, 0.0])
        
    # Center the initial configuration
    cx_avg = sum(p[0] for p in init_positions) / N
    cy_avg = sum(p[1] for p in init_positions) / N
    init_positions = [[p[0] - cx_avg, p[1] - cy_avg, p[2]] for p in init_positions]

    best_score = float('inf')
    best_config = None
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    steps_per_temp = 1000
    
    current_config = [p[:] for p in init_positions]
    
    # Precompute vertices for current config? No, angles change.
    
    def get_config_vertices(config):
        verts_list = []
        for i in range(N):
            x, y, a = config[i]
            verts_list.append(get_hex_vertices(x, y, a))
        return verts_list

    def evaluate_config(config):
        verts_list = get_config_vertices(config)
        
        # Check collisions
        total_pen = 0.0
        # We want to minimize side length, but collisions are forbidden.
        # In SA, we allow collisions with a penalty.
        for i in range(N):
            for j in range(i+1, N):
                disjoint, pen, _ = sat_check(verts_list[i], verts_list[j])
                if not disjoint:
                    total_pen += pen
                    
        # Calculate side length
        # To speed up, we can just use one angle, or check a few.
        # For SA, checking 1 angle is fast. We refine at the end.
        side = calc_outer_side(verts_list, 0.0)
        
        # Objective: Side length + Large Penalty * Total Penetration
        penalty_weight = 100.0
        return side + penalty_weight * total_pen

    current_cost = evaluate_config(current_config)
    
    while T > T_min and time.time() < deadline - 10.0:
        for _ in range(steps_per_temp):
            if time.time() > deadline - 10.0:
                break
                
            # Generate neighbor
            new_config = [p[:] for p in current_config]
            
            # Move one random hexagon
            idx = random.randint(0, N-1)
            
            # Random translation
            dx = random.gauss(0, 0.5)
            dy = random.gauss(0, 0.5)
            new_config[idx][0] += dx
            new_config[idx][1] += dy
            
            # Random rotation
            da = random.gauss(0, 10)
            new_config[idx][2] += da
            
            # Recenter the whole cluster to keep it near origin?
            # This prevents drift and keeps the bounding box centered.
            # However, recentering changes all positions, which might be costly for SA acceptance logic if not careful.
            # But it removes the 2 DOF of global translation.
            cx_avg = sum(p[0] for p in new_config) / N
            cy_avg = sum(p[1] for p in new_config) / N
            for i in range(N):
                new_config[i][0] -= cx_avg
                new_config[i][1] -= cy_avg
                
            new_cost = evaluate_config(new_config)
            
            # Acceptance criterion
            delta_cost = new_cost - current_cost
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                current_config = new_config
                current_cost = new_cost
                
                # Track best non-colliding config
                # Check if current is valid
                verts_list = get_config_vertices(current_config)
                is_valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        disjoint, _, _ = sat_check(verts_list[i], verts_list[j])
                        if not disjoint:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                        
                if is_valid:
                    # Check best side with a few angles
                    best_side_local = float('inf')
                    for oa in range(0, 60, 10):
                        s = calc_outer_side(verts_list, oa)
                        if s < best_side_local:
                            best_side_local = s
                            
                    if best_side_local < best_score:
                        best_score = best_side_local
                        best_config = [p[:] for p in current_config]
                        
        T *= alpha
        
    # Refine best config found
    if best_config is not None:
        # Final refinement: fine-tune positions and angles to minimize side length
        # Use a simple local search or just fine-grained angle search
        
        final_config = [p[:] for p in best_config]
        
        # Try to shrink further by moving hexagons towards center if possible?
        # Or just find the best outer angle.
        
        verts_list = get_config_vertices(final_config)
        
        best_side = float('inf')
        best_oa = 0.0
        # Fine angle search
        for oa in range(0, 60, 1):
            s = calc_outer_side(verts_list, oa)
            if s < best_side:
                best_side = s
                best_oa = oa
                
        # One more pass: try to push hexagons together slightly?
        # If we have time, do a few steps of "compression"
        # This is risky. Let's stick to the SA result.
        
        # Construct return
        inner_list = []
        for i in range(N):
            inner_list.append([final_config[i][0], final_config[i][1], final_config[i][2]])
            
        return (inner_list, (0.0, 0.0), best_side, best_oa)
        
    # Fallback: return initial config if SA failed to find valid one
    # (Unlikely given initial config is valid)
    verts_list = get_config_vertices(init_positions)
    side = calc_outer_side(verts_list, 0.0)
    inner_list = [[p[0], p[1], p[2]] for p in init_positions]
    return (inner_list, (0.0, 0.0), side, 0.0)

# EVOLVE-BLOCK-END
