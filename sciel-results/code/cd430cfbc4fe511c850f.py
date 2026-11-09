import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Initial configuration: The grid from the failed program, but scaled up slightly
    # to ensure strict non-overlap (distance > sqrt(3))
    d_base = math.sqrt(3)
    scale = 1.05 # 5% expansion to guarantee gap
    
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1),
        (-1, 0), (-1, 1), (0, 1)
    ]
    axial_coords.extend([
        (2, 0),
        (2, -1),
        (1, -2),
        (0, -2)
    ])
    
    centers = []
    for q, r in axial_coords:
        x = d_base * scale * (q + r / 2.0)
        y = d_base * scale * (math.sqrt(3) / 2.0) * r
        centers.append([x, y])
        
    angles = [0.0] * N
    
    # Helper to calculate cost
    def get_cost(centers, angles):
        # Check overlaps
        penalty = 0.0
        min_dist_sq = 3.0 # (sqrt(3))^2
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_dist_sq:
                    # Penalty for overlap
                    # Approximate overlap depth
                    dist = math.sqrt(dist_sq)
                    overlap = math.sqrt(3) - dist
                    penalty += overlap * overlap * 100.0 # Heavy penalty
        
        if penalty > 0:
            return penalty, float('inf')
            
        # Calculate enclosing hexagon side
        # Get all vertices
        verts = []
        for i in range(N):
            cx, cy = centers[i]
            theta = math.radians(angles[i])
            for k in range(6):
                a = theta + math.radians(60 * k)
                vx = cx + math.cos(a)
                vy = cy + math.sin(a)
                verts.append([vx, vy])
        verts = np.array(verts)
        
        centroid = np.mean(verts, axis=0)
        
        best_side = float('inf')
        # Scan angles for enclosing hexagon
        # Optimization: We can optimize the center too, but centroid is a good start.
        # To save time, we might just check a few angles or use a simpler bound.
        # However, for N=11, 61 angles is fast.
        for theta_deg in range(0, 61, 2): # 2 degree resolution for speed in SA
            theta = math.radians(theta_deg)
            max_proj = 0.0
            for k in range(6):
                a = theta + math.radians(60 * k)
                nx = math.cos(a)
                ny = math.sin(a)
                projs = (verts[:, 0] - centroid[0]) * nx + (verts[:, 1] - centroid[1]) * ny
                local_max = np.max(np.abs(projs))
                if local_max > max_proj:
                    max_proj = local_max
            side = 2.0 * max_proj / math.sqrt(3)
            if side < best_side:
                best_side = side
                
        return 0.0, best_side

    # SA Parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    best_cost, best_side = get_cost(centers, angles)
    best_centers = [c[:] for c in centers]
    best_angles = angles[:]
    
    current_centers = [c[:] for c in centers]
    current_angles = angles[:]
    current_cost, current_side = best_cost, best_side
    
    steps = 0
    while time.time() < deadline - 5.0:
        if T < T_min:
            break
            
        # Propose move
        idx = np.random.randint(N)
        move_type = np.random.random()
        
        if move_type < 0.7:
            # Translate
            step_size = 0.1 * (1 + T)
            current_centers[idx][0] += np.random.normal(0, step_size)
            current_centers[idx][1] += np.random.normal(0, step_size)
        else:
            # Rotate
            current_angles[idx] += np.random.normal(0, 10.0)
            
        new_cost, new_side = get_cost(current_centers, current_angles)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            current_cost = new_cost
            current_side = new_side
            if new_cost == 0 and new_side < best_side:
                best_side = new_side
                best_centers = [c[:] for c in current_centers]
                best_angles = current_angles[:]
                
        T *= alpha
        steps += 1
        if steps % 1000 == 0:
            # Cooldown or reheat? No, just keep going.
            pass
            
    # Final refinement: If we have a valid solution, try to minimize side length
    # using a tighter scan on the best solution found
    if best_cost == 0:
        # Do a local search on center and angle of the outer hexagon for the best configuration
        # The get_cost function uses centroid and coarse angle scan.
        # We can do a finer scan here for the final result.
        verts = []
        for i in range(N):
            cx, cy = best_centers[i]
            theta = math.radians(best_angles[i])
            for k in range(6):
                a = theta + math.radians(60 * k)
                vx = cx + math.cos(a)
                vy = cy + math.sin(a)
                verts.append([vx, vy])
        verts = np.array(verts)
        
        centroid = np.mean(verts, axis=0)
        
        final_best_side = float('inf')
        final_best_angle = 0.0
        
        # Fine scan
        for theta_deg in range(0, 61, 1):
            theta = math.radians(theta_deg)
            max_proj = 0.0
            for k in range(6):
                a = theta + math.radians(60 * k)
                nx = math.cos(a)
                ny = math.sin(a)
                projs = (verts[:, 0] - centroid[0]) * nx + (verts[:, 1] - centroid[1]) * ny
                local_max = np.max(np.abs(projs))
                if local_max > max_proj:
                    max_proj = local_max
            side = 2.0 * max_proj / math.sqrt(3)
            if side < final_best_side:
                final_best_side = side
                final_best_angle = theta_deg
                
        # Return best valid solution
        inner = [[best_centers[i][0], best_centers[i][1], best_angles[i]] for i in range(N)]
        return inner, (centroid[0], centroid[1]), final_best_side, final_best_angle
    else:
        # If no valid solution found, return the initial scaled one (which is valid)
        # and calculate its side length
        verts = []
        for i in range(N):
            cx, cy = centers[i]
            theta = math.radians(angles[i])
            for k in range(6):
                a = theta + math.radians(60 * k)
                vx = cx + math.cos(a)
                vy = cy + math.sin(a)
                verts.append([vx, vy])
        verts = np.array(verts)
        centroid = np.mean(verts, axis=0)
        
        final_best_side = float('inf')
        final_best_angle = 0.0
        for theta_deg in range(0, 61, 1):
            theta = math.radians(theta_deg)
            max_proj = 0.0
            for k in range(6):
                a = theta + math.radians(60 * k)
                nx = math.cos(a)
                ny = math.sin(a)
                projs = (verts[:, 0] - centroid[0]) * nx + (verts[:, 1] - centroid[1]) * ny
                local_max = np.max(np.abs(projs))
                if local_max > max_proj:
                    max_proj = local_max
            side = 2.0 * max_proj / math.sqrt(3)
            if side < final_best_side:
                final_best_side = side
                final_best_angle = theta_deg
                
        inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
        return inner, (centroid[0], centroid[1]), final_best_side, final_best_angle

# EVOLVE-BLOCK-END
