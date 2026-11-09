"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    sqrt3 = math.sqrt(3)
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def compute_min_side(inner_positions, outer_center, outer_angle):
        """Compute minimum outer side to contain all inner hexagons."""
        cx, cy = outer_center
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        
        # Pre-compute normals for the outer hexagon in local coordinates
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        for ix, iy, iang in inner_positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                # Rotate to align with outer hexagon
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                
                # Find maximum projection on each of 6 normals
                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    # Required side: proj / (sqrt(3)/2)
                    required_side = proj / (sqrt3 / 2)
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def check_overlap_fast(pos1, pos2):
        """
        Fast overlap check. Returns 0 if disjoint, positive penalty if overlapping.
        Uses bounding box first, then SAT.
        """
        # Bounding box check
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.0:
            return 0.0
        if dist_sq < 0.1:
            return 1.0
        
        v1 = hexagon_vertices(pos1[0], pos1[1], 1.0, pos1[2])
        v2 = hexagon_vertices(pos2[0], pos2[1], 1.0, pos2[2])
        
        # Collect all normals
        normals = []
        for i in range(6):
            # Edge 1
            ex = v1[(i+1)%6][0] - v1[i][0]
            ey = v1[(i+1)%6][1] - v1[i][1]
            nx = -ey
            ny = ex
            length = math.sqrt(nx*nx + ny*ny)
            if length > 1e-9:
                normals.append((nx/length, ny/length))
            
            # Edge 2
            ex = v2[(i+1)%6][0] - v2[i][0]
            ey = v2[(i+1)%6][1] - v2[i][1]
            nx = -ey
            ny = ex
            length = math.sqrt(nx*nx + ny*ny)
            if length > 1e-9:
                normals.append((nx/length, ny/length))
        
        # Check for separation along each normal
        min_overlap = float('inf')
        for nx, ny in normals:
            # Project both hexagons onto this normal
            proj1 = [v[0]*nx + v[1]*ny for v in v1]
            proj2 = [v[0]*nx + v[1]*ny for v in v2]
            
            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)
            
            # If separated, no overlap
            if max1 <= min2 or max2 <= min1:
                return 0.0
            
            # Calculate overlap in this direction
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
        
        # If we get here, they overlap
        return min_overlap

    def total_penalty_fast(inner_positions):
        """Sum of exact overlap penalties for all pairs."""
        penalty = 0.0
        n = len(inner_positions)
        for i in range(n):
            for j in range(i+1, n):
                p = check_overlap_fast(inner_positions[i], inner_positions[j])
                if p > 0:
                    penalty += p
        return penalty

    # Initialize positions in a hexagonal lattice pattern
    # We use a "radius 2" hexagon (19 cells) and pick the 11 most central ones.
    coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) <= 2:
                coords.append((q, r))
    
    def dist_to_center(coord):
        q, r = coord
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        return x*x + y*y
        
    coords.sort(key=dist_to_center)
    selected_coords = coords[:N]
    
    inner_positions = []
    for q, r in selected_coords:
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        inner_positions.append([x, y, 0.0])
    
    # Simulated Annealing
    current_temp = 0.5
    cooling_rate = 0.9995
    min_temp = 0.01
    best_inner = [p[:] for p in inner_positions]
    best_score = float('inf')
    
    def evaluate(positions, outer_angle):
        side = compute_min_side(positions, (0.0, 0.0), outer_angle)
        pen = total_penalty_fast(positions)
        return side + 1000.0 * pen

    current_score = evaluate(inner_positions, 0.0)
    best_score = current_score
    
    steps = 0
    max_steps = 50000
    
    while current_temp > min_temp and time.time() < deadline - 5.0:
        # Perturb one random hexagon
        idx = random.randint(0, N-1)
        old_pos = inner_positions[idx][:]
        new_pos = old_pos[:]
        
        # Move
        dx = random.gauss(0, 0.05)
        dy = random.gauss(0, 0.05)
        new_pos[0] += dx
        new_pos[1] += dy
        
        # Rotate
        d_angle = random.gauss(0, 2.0)
        new_pos[2] += d_angle
        
        # Apply change
        inner_positions[idx] = new_pos
        
        new_score = evaluate(inner_positions, 0.0)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / current_temp):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_inner = [p[:] for p in inner_positions]
        else:
            # Revert
            inner_positions[idx] = old_pos
            
        current_temp *= cooling_rate
        steps += 1
        
        if time.time() > deadline - 10:
            break

    # Use the best found inner positions
    # Now, find the best outer angle for this configuration
    best_side = float('inf')
    best_angle = 0.0
    for outer_angle in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]:
        if time.time() > deadline - 2:
            break
        side = compute_min_side(best_inner, (0.0, 0.0), outer_angle)
        if side < best_side:
            best_side = side
            best_angle = outer_angle
    
    # Verify no overlap in best_inner
    final_penalty = total_penalty_fast(best_inner)
    if final_penalty > 1e-6:
        # Fallback to a safe, known valid packing
        safe_inner = []
        for i in range(N):
            if i == 0:
                safe_inner.append([0.0, 0.0, 0.0])
            else:
                r = 2.1 * (i // 6 + 1) 
                angle = math.radians(60 * (i % 6))
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                safe_inner.append([x, y, 0.0])
        
        # Find best angle for safe inner
        for outer_angle in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]:
            if time.time() > deadline - 1:
                break
            side = compute_min_side(safe_inner, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
                best_angle = outer_angle
                best_inner = safe_inner
        
    return best_inner, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
