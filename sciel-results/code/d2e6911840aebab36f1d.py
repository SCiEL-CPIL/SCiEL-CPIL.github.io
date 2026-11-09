"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    sqrt3 = math.sqrt(3)
    inradius = sqrt3 / 2.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        return [(cx + side * math.cos(angle_rad + k * math.pi / 3),
                 cy + side * math.sin(angle_rad + k * math.pi / 3)) for k in range(6)]

    def compute_min_side_smooth(positions, outer_angle, beta):
        """
        Compute a smooth approximation of the minimum outer side length.
        Uses log-sum-exp on the projections of all vertices onto the 6 normal axes.
        """
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        # Normals of the outer hexagon (apothem directions)
        normals = [(math.cos(math.radians(30 + k * 60)), math.sin(math.radians(30 + k * 60))) for k in range(6)]

        projs = []
        for ix, iy, iang in positions:
            iv = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in iv:
                rx = vx * cos_a - vy * sin_a
                ry = vx * sin_a + vy * cos_a
                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    projs.append(proj)
        
        if not projs:
            return 0.0
            
        max_p = max(projs)
        exp_sum = 0.0
        for p in projs:
            exp_sum += math.exp(beta * (p - max_p))
        
        smooth_side = max_p + (1.0 / beta) * math.log(exp_sum)
        return smooth_side / inradius

    def compute_min_side_exact(positions, outer_angle):
        """Compute the exact minimum outer side length."""
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        normals = [(math.cos(math.radians(30 + k * 60)), math.sin(math.radians(30 + k * 60))) for k in range(6)]

        max_proj = 0.0
        for ix, iy, iang in positions:
            iv = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in iv:
                rx = vx * cos_a - vy * sin_a
                ry = vx * sin_a + vy * cos_a
                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    if proj > max_proj:
                        max_proj = proj
        
        return max_proj / inradius

    def sat_overlap(p1, p2):
        """Return penetration depth > 0 if overlapping, 0 if disjoint."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d = math.sqrt(dx*dx + dy*dy)
        if d >= 2.0:
            return 0.0
        if d < 1e-9:
            return 2.0

        v1 = hexagon_vertices(p1[0], p1[1], 1.0, p1[2])
        v2 = hexagon_vertices(p2[0], p2[1], 1.0, p2[2])

        axes = []
        for verts in (v1, v2):
            for i in range(6):
                x1, y1 = verts[i]
                x2, y2 = verts[(i+1) % 6]
                ex, ey = x2-x1, y2-y1
                L = math.sqrt(ex*ex + ey*ey)
                if L > 1e-12:
                    axes.append((-ey/L, ex/L))
        axes.append((dx/d, dy/d))

        min_gap = float('inf')
        for nx, ny in axes:
            p1min = min(vx*nx+vy*ny for vx,vy in v1)
            p1max = max(vx*nx+vy*ny for vx,vy in v1)
            p2min = min(vx*nx+vy*ny for vx,vy in v2)
            p2max = max(vx*nx+vy*ny for vx,vy in v2)
            gap = max(p2min - p1max, p1min - p2max)
            if gap > 0:
                return 0.0
            min_gap = min(min_gap, -gap)
        return min_gap if min_gap < float('inf') else 0.0

    def objective_smooth(positions, outer_angle, beta, pen_weight=100.0):
        """
        Objective for SA: Smooth outer side + penalty for overlaps.
        """
        side = compute_min_side_smooth(positions, outer_angle, beta)
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p = sat_overlap(positions[i], positions[j])
                if p > 0:
                    pen += p * p
        return side + pen_weight * pen

    def objective_exact(positions, outer_angle, pen_weight=1000.0):
        """
        Objective for NM refinement: Exact outer side + strong penalty for overlaps.
        """
        side = compute_min_side_exact(positions, outer_angle)
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p = sat_overlap(positions[i], positions[j])
                if p > 0:
                    pen += p * p
        return side + pen_weight * pen

    def make_honeycomb():
        """Generate initial positions on a hexagonal lattice."""
        positions = []
        positions.append([0.0, 0.0, 0.0])
        # First ring: 6 hexagons
        ring_dist = sqrt3 + 0.01 # Slightly more than touching
        for k in range(6):
            a = math.radians(60*k)
            positions.append([ring_dist*math.cos(a), ring_dist*math.sin(a), 0.0])
        
        # Second ring: We need 4 more to reach 11.
        # The corners of the hexagon of hexagons are at distance 2*ring_dist from center?
        # Actually, a full hexagon of hexagons has 1 (center) + 6 (ring 1) + 12 (ring 2) = 19.
        # We only need 11. So we take center + ring 1 + 4 from ring 2.
        # Ring 2 positions are at hexagonal coordinates (q,r) with max(|q|,|r|,|q+r|)=2.
        # Let's just place 4 more in the "corners" or sides.
        # A good strategy is to place them at the midpoints of the gaps in ring 1?
        # Or just extend the lattice.
        
        # Let's use axial coordinates for ring 2
        # Directions for ring 2:
        # (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-1), (1,-2), (0,-1)... wait.
        # Let's just pick 4 distinct directions from the 12 possible ring 2 spots.
        # To keep symmetry, pick 4 that are evenly distributed?
        # Or just pick the first 4 generated.
        
        ring2_coords = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                if max(abs(q), abs(r), abs(q+r)) == 2:
                    ring2_coords.append((q, r))
        
        # Sort by angle to ensure even distribution
        def hex_to_polar(q, r):
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            return math.atan2(y, x), x, y
            
        ring2_coords.sort(key=lambda cr: hex_to_polar(cr[0], cr[1])[0])
        
        # Pick 4 evenly spaced from the 12
        indices = [0, 3, 6, 9] # Every 3rd one
        for idx in indices:
            q, r = ring2_coords[idx]
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            positions.append([x, y, 0.0])
            
        return positions

    def to_params(positions):
        params = []
        for p in positions:
            params.extend([p[0], p[1], p[2]])
        return np.array(params)

    def from_params(params):
        positions = []
        for i in range(N):
            positions.append([params[3*i], params[3*i+1], params[3*i+2]])
        return positions

    # --- Simulated Annealing with Rigid Clusters ---
    
    # Initialize
    current_positions = make_honeycomb()
    # Relax initial overlaps if any (should be minimal with 0.01 gap)
    # For SA, we can start with slight overlaps if we have penalty, but better to start valid.
    
    current_obj = objective_smooth(current_positions, 0.0, 1.0)
    
    best_positions = [p[:] for p in current_positions]
    best_obj = current_obj
    
    # Define clusters based on initial honeycomb neighbors
    # 0 is center. 1-6 are ring 1. 7-10 are ring 2.
    bonds = []
    for i in range(1, 7):
        bonds.append((0, i))
    for i in range(1, 7):
        bonds.append((i, (i % 6) + 1))
    # Connect ring 2 to their nearest ring 1 neighbors
    # This requires knowing the geometry. 
    # Simpler: Just define a few static clusters for SA moves.
    # Cluster 1: Center + Ring 1 (7 hexagons)
    # Cluster 2: Individual ring 2 hexagons
    # Cluster 3: Pairs of adjacent ring 1 hexagons
    
    # Let's use dynamic cluster generation based on proximity
    def get_cluster(start_idx, max_size):
        """Get a random connected cluster of size <= max_size starting from start_idx."""
        # Simple BFS/DFS on the current configuration based on distance
        cluster = [start_idx]
        visited = {start_idx}
        
        # Find neighbors within distance < 2.1
        potential_neighbors = []
        for j in range(N):
            if j == start_idx:
                continue
            dx = current_positions[start_idx][0] - current_positions[j][0]
            dy = current_positions[start_idx][1] - current_positions[j][1]
            if math.sqrt(dx*dx + dy*dy) < 2.1:
                potential_neighbors.append(j)
        
        # Randomly add neighbors
        random.shuffle(potential_neighbors)
        
        while potential_neighbors and len(cluster) < max_size:
            next_idx = potential_neighbors.pop(0)
            if next_idx not in visited:
                cluster.append(next_idx)
                visited.add(next_idx)
                # Add new neighbors of next_idx
                for k in range(N):
                    if k not in visited and k != next_idx:
                        dx = current_positions[next_idx][0] - current_positions[k][0]
                        dy = current_positions[next_idx][1] - current_positions[k][1]
                        if math.sqrt(dx*dx + dy*dy) < 2.1:
                            if k not in potential_neighbors:
                                potential_neighbors.append(k)
        return cluster

    temp = 10.0
    beta = 1.0
    steps = 0
    max_steps = 50000
    
    while steps < max_steps and time.time() < deadline - 30.0:
        steps += 1
        
        temp *= 0.995
        beta = min(100.0, beta + 0.05)
        
        move_type = random.random()
        old_positions = [p[:] for p in current_positions]
        
        if move_type < 0.6:
            # Single hexagon move
            idx = random.randint(0, N-1)
            dx = random.gauss(0, 0.2)
            dy = random.gauss(0, 0.2)
            current_positions[idx][0] += dx
            current_positions[idx][1] += dy
            current_positions[idx][2] += random.gauss(0, 10.0)
            
        elif move_type < 0.8:
            # Rotation only
            idx = random.randint(0, N-1)
            current_positions[idx][2] += random.gauss(0, 10.0)
            
        else:
            # Cluster move
            start_idx = random.randint(0, N-1)
            cluster = get_cluster(start_idx, 4)
            
            # Compute centroid
            cx = sum(current_positions[i][0] for i in cluster) / len(cluster)
            cy = sum(current_positions[i][1] for i in cluster) / len(cluster)
            
            dx = random.gauss(0, 0.2)
            dy = random.gauss(0, 0.2)
            rot = random.gauss(0, 5.0)
            
            r_rad = math.radians(rot)
            cos_r = math.cos(r_rad)
            sin_r = math.sin(r_rad)
            
            for i in cluster:
                x = current_positions[i][0] - cx
                y = current_positions[i][1] - cy
                new_x = x * cos_r - y * sin_r + cx + dx
                new_y = x * sin_r + y * cos_r + cy + dy
                current_positions[i][0] = new_x
                current_positions[i][1] = new_y
                current_positions[i][2] += rot

        new_obj = objective_smooth(current_positions, 0.0, beta)
        
        if new_obj < current_obj or random.random() < math.exp(-(new_obj - current_obj) / temp):
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_positions = [p[:] for p in current_positions]
        else:
            # Revert
            current_positions = old_positions

    # --- Refinement with Nelder-Mead ---
    
    # Start NM from the best position found in SA
    x0 = to_params(best_positions)
    
    # Stage 1: Smooth objective NM
    # This helps navigate the landscape if SA got stuck near a kink
    def nm_obj_smooth(params):
        positions = from_params(params)
        return objective_smooth(positions, 0.0, 50.0, pen_weight=100.0)
    
    try:
        if time.time() < deadline - 15.0:
            res1 = minimize(nm_obj_smooth, x0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
            if res1.fun < best_obj:
                best_positions = from_params(res1.x)
                best_obj = res1.fun
    except:
        pass

    # Stage 2: Exact objective NM
    # This refines the exact side length and ensures no overlap
    def nm_obj_exact(params):
        positions = from_params(params)
        return objective_exact(positions, 0.0, pen_weight=1000.0)
    
    try:
        if time.time() < deadline - 5.0:
            res2 = minimize(nm_obj_exact, to_params(best_positions), method='Nelder-Mead', 
                            options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            if res2.fun < best_obj:
                best_positions = from_params(res2.x)
                best_obj = res2.fun
    except:
        pass

    # Final Validation and Side Calculation
    # Check for overlaps
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if sat_overlap(best_positions[i], best_positions[j]) > 0:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a known valid configuration if NM failed
        # Try to relax the best_positions found
        # Simple relaxation: push apart
        for _ in range(50):
            overlap_found = False
            for i in range(N):
                for j in range(i+1, N):
                    pen = sat_overlap(best_positions[i], best_positions[j])
                    if pen > 0:
                        overlap_found = True
                        # Push apart along the line connecting centers
                        dx = best_positions[i][0] - best_positions[j][0]
                        dy = best_positions[i][1] - best_positions[j][1]
                        d = math.sqrt(dx*dx + dy*dy)
                        if d < 1e-9:
                            dx, dy = 1, 0
                            d = 1
                        else:
                            dx /= d
                            dy /= d
                        push = pen / 2.0 + 0.001
                        best_positions[i][0] += dx * push
                        best_positions[i][1] += dy * push
                        best_positions[j][0] -= dx * push
                        best_positions[j][1] -= dy * push
            if not overlap_found:
                break
        
        # Re-check
        for i in range(N):
            for j in range(i+1, N):
                if sat_overlap(best_positions[i], best_positions[j]) > 0:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Last resort: honeycomb
            best_positions = make_honeycomb()

    # Compute final exact side length
    # Try a few outer angles to see if we can squeeze it tighter
    # Since we fixed outer to 0 in SA/NM, we should check if rotating the outer
    # hexagon relative to the inner cluster helps.
    # Actually, rotating the outer hexagon is equivalent to rotating the inner cluster
    # by -angle. Since the inner cluster is fixed in our solution, we just need to
    # find the best outer angle.
    
    best_side = float('inf')
    best_angle = 0.0
    
    for angle in np.arange(0, 60, 1.0):
        side = compute_min_side_exact(best_positions, angle)
        if side < best_side:
            best_side = side
            best_angle = angle
            
    # Refine angle
    if best_angle > 0 and best_angle < 59:
        for angle in np.arange(max(0, best_angle-2), min(60, best_angle+2), 0.1):
            side = compute_min_side_exact(best_positions, angle)
            if side < best_side:
                best_side = side
                best_angle = angle

    return best_positions, (0.0, 0.0), best_side, best_angle

# EVOLVE-BLOCK-END
