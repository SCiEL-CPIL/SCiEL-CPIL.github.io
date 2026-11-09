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

    def check_overlap_sat(pos1, pos2):
        """
        Returns True if hexagons overlap, False if disjoint.
        Uses Separating Axis Theorem (SAT).
        """
        # Bounding box check for quick rejection
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        if dx*dx + dy*dy > 4.0: # Max distance for contact is 2.0 (side length 1 -> circumradius 1)
            return False
        
        v1 = hexagon_vertices(pos1[0], pos1[1], 1.0, pos1[2])
        v2 = hexagon_vertices(pos2[0], pos2[1], 1.0, pos2[2])
        
        # Check separation along normals of both hexagons
        for verts in (v1, v2):
            for i in range(6):
                # Edge vector
                ex = verts[(i+1)%6][0] - verts[i][0]
                ey = verts[(i+1)%6][1] - verts[i][1]
                # Normal vector
                nx = -ey
                ny = ex
                length = math.sqrt(nx*nx + ny*ny)
                if length < 1e-9: continue
                nx /= length
                ny /= length
                
                # Project both hexagons onto this normal
                proj1 = [v[0]*nx + v[1]*ny for v in v1]
                proj2 = [v[0]*nx + v[1]*ny for v in v2]
                
                min1, max1 = min(proj1), max(proj1)
                min2, max2 = min(proj2), max(proj2)
                
                # If separated, no overlap
                if max1 <= min2 or max2 <= min1:
                    return False
        
        # If no separating axis found, they overlap
        return True

    def is_valid_config(inner_positions):
        """Check if all pairs of hexagons are disjoint."""
        n = len(inner_positions)
        for i in range(n):
            for j in range(i+1, n):
                if check_overlap_sat(inner_positions[i], inner_positions[j]):
                    return False
        return True

    # 1. Define Rigid Clusters
    # We decompose the 11 hexagons into 3 rigid clusters based on honeycomb adjacency.
    # Cluster 0: Center (1 hex)
    # Cluster 1: 3 hexagons in a "L" shape or line? Let's use a compact triangle of 3.
    # Cluster 2: Remaining 7? That's too big.
    
    # Better decomposition for flexibility:
    # Cluster 0: Center (1)
    # Cluster 1: 3 neighbors (forming a triangle)
    # Cluster 2: 3 neighbors (forming a triangle)
    # Cluster 3: 3 neighbors (forming a triangle)
    # Total: 1 + 3 + 3 + 3 = 10. We have 11.
    # Let's do:
    # C0: 1 (center)
    # C1: 4 (a small diamond/rhombus of 4)
    # C2: 4 (a small diamond/rhombus of 4)
    # C3: 2 (a pair)
    # Total: 11.
    
    # Initial positions for honeycomb (side length 1, so distance between centers is sqrt(3) * 1? No.
    # Distance between centers of touching unit hexagons is sqrt(3).
    # Wait, side length 1. Circumradius 1. Inradius sqrt(3)/2.
    # Distance between centers = 2 * Inradius = sqrt(3).
    
    d = sqrt3
    
    # Generate base honeycomb coords for 11 hexes
    # (0,0) is center.
    # Neighbors at distance d.
    base_coords = []
    # Center
    base_coords.append((0.0, 0.0))
    
    # 6 neighbors
    for k in range(6):
        angle = math.radians(30 + k * 60)
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        base_coords.append((x, y))
        
    # Next ring? We need 11. 1 + 6 = 7. Need 4 more.
    # Pick 4 from the next ring (distance 2*d? No, distance between (1,0) and (2,0) is d).
    # Let's just pick 4 adjacent to the first ring to keep it compact.
    # (d*cos(30), d*sin(30)) neighbor in direction 30 is at distance d further.
    # Actually, let's just use a fixed set of 11 indices from a larger honeycomb.
    
    # Simpler: Just define the clusters with relative offsets.
    
    # Cluster 0: [0] (Center)
    # Cluster 1: [1, 2, 3, 4] (4 hexes)
    # Cluster 2: [5, 6, 7, 8] (4 hexes)
    # Cluster 3: [9, 10] (2 hexes)
    
    # We need to assign specific base positions to these indices to form a valid initial packing.
    # Let's use the standard "radius 2" honeycomb subset.
    # Indices in a hex grid (axial coords):
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    
    # Let's pick:
    # 0: (0,0)
    # 1: (1,0)
    # 2: (0,1)
    # 3: (-1,1)
    # 4: (-1,0)
    # 5: (0,-1)
    # 6: (1,-1)
    # 7: (2,0)
    # 8: (1,1)
    # 9: (0,2)
    # 10: (-1,2)
    
    axial_coords = [
        (0,0),
        (1,0),
        (0,1),
        (-1,1),
        (-1,0),
        (0,-1),
        (1,-1),
        (2,0),
        (1,1),
        (0,2),
        (-1,2)
    ]
    
    # Convert to Cartesian
    def axial_to_cartesian(q, r):
        x = d * (q + r/2.0)
        y = d * (3.0/2.0 * r) # Wait, standard conversion: x = size * sqrt(3) * (q + r/2), y = size * 3/2 * r.
        # Here size=1.
        return x, y

    base_positions = [axial_to_cartesian(q, r) for q, r in axial_coords]
    
    # Define Clusters: List of lists of indices
    clusters = [
        [0],       # Cluster 0
        [1, 2, 3, 4], # Cluster 1
        [5, 6, 7, 8], # Cluster 2
        [9, 10]     # Cluster 3
    ]
    
    # Cluster States: [dx, dy, angle]
    # Initial state: Identity
    cluster_states = [[0.0, 0.0, 0.0] for _ in clusters]
    
    def get_inner_positions(cluster_states):
        """Calculate absolute positions of all 11 hexagons given cluster states."""
        positions = [None] * N
        for c_idx, state in enumerate(cluster_states):
            dx, dy, ang_deg = state
            ang_rad = math.radians(ang_deg)
            cos_a = math.cos(ang_rad)
            sin_a = math.sin(ang_rad)
            
            for h_idx in clusters[c_idx]:
                bx, by = base_positions[h_idx]
                # Rotate base position
                rx = bx * cos_a - by * sin_a
                ry = bx * sin_a + by * cos_a
                # Translate
                positions[h_idx] = [rx + dx, ry + dy, ang_deg]
        return positions

    def evaluate_config(cluster_states, outer_angle):
        """
        Returns (side_length, valid).
        If not valid, side_length is inf.
        """
        inner_positions = get_inner_positions(cluster_states)
        
        # Check validity
        if not is_valid_config(inner_positions):
            return float('inf'), False
            
        side = compute_min_side(inner_positions, (0.0, 0.0), outer_angle)
        return side, True

    # Optimization Loop
    best_side = float('inf')
    best_cluster_states = [s[:] for s in cluster_states]
    best_outer_angle = 0.0
    best_inner_positions = get_inner_positions(cluster_states)
    
    current_cluster_states = [s[:] for s in cluster_states]
    current_outer_angle = 0.0
    
    # Initial Evaluation
    cur_side, cur_valid = evaluate_config(current_cluster_states, current_outer_angle)
    if cur_valid:
        best_side = cur_side
        best_cluster_states = [s[:] for s in current_cluster_states]
        best_outer_angle = current_outer_angle
        best_inner_positions = get_inner_positions(current_cluster_states)

    # Simulated Annealing on Cluster States + Outer Angle
    temp = 1.0
    cooling = 0.999
    min_temp = 0.01
    
    while temp > min_temp and time.time() < deadline - 5.0:
        # Perturb
        # 1. Move/Rotate a random cluster
        c_idx = random.randint(0, len(clusters)-1)
        old_state = current_cluster_states[c_idx][:]
        
        current_cluster_states[c_idx][0] += random.gauss(0, 0.1)
        current_cluster_states[c_idx][1] += random.gauss(0, 0.1)
        current_cluster_states[c_idx][2] += random.gauss(0, 5.0)
        
        # 2. Perturb Outer Angle slightly
        old_outer_angle = current_outer_angle
        current_outer_angle += random.gauss(0, 1.0)
        
        # Evaluate
        new_side, new_valid = evaluate_config(current_cluster_states, current_outer_angle)
        
        if new_valid:
            delta = new_side - cur_side if cur_valid else 0
            if cur_valid:
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    cur_side = new_side
                    # Accept
                else:
                    # Revert
                    current_cluster_states[c_idx] = old_state
                    current_outer_angle = old_outer_angle
            else:
                # Previous was invalid, so any valid move is an improvement
                cur_side = new_side
                
            if new_side < best_side:
                best_side = new_side
                best_cluster_states = [s[:] for s in current_cluster_states]
                best_outer_angle = current_outer_angle
                best_inner_positions = get_inner_positions(current_cluster_states)
        else:
            # New config is invalid
            if cur_valid:
                # Reject, revert
                current_cluster_states[c_idx] = old_state
                current_outer_angle = old_outer_angle
            else:
                # Both invalid, keep current (which is also invalid) or revert? 
                # Revert to keep exploring around last valid or just random walk in invalid space?
                # Better to revert to avoid getting stuck in deep invalid regions
                current_cluster_states[c_idx] = old_state
                current_outer_angle = old_outer_angle
                
        temp *= cooling

    # Final Polish: Refine Outer Angle for the best found cluster configuration
    # We know best_cluster_states is valid.
    # Try a few specific outer angles to see if we can squeeze the side length.
    final_inner = get_inner_positions(best_cluster_states)
    
    # Check if final_inner is still valid (it should be, as it's the same config)
    if is_valid_config(final_inner):
        candidates = [best_outer_angle]
        # Add some local variations
        for delta_a in [-5, -2, 0, 2, 5]:
            candidates.append(best_outer_angle + delta_a)
            
        min_side_final = float('inf')
        min_angle_final = best_outer_angle
        
        for ang in candidates:
            side = compute_min_side(final_inner, (0.0, 0.0), ang)
            if side < min_side_final:
                min_side_final = side
                min_angle_final = ang
                
        best_side = min_side_final
        best_outer_angle = min_angle_final
        best_inner_positions = final_inner

    # Fallback if best_side is still inf (should not happen if initial config is valid)
    if math.isinf(best_side):
        # Use a very safe, spread out config
        safe_inner = []
        for i in range(N):
            if i == 0:
                safe_inner.append([0.0, 0.0, 0.0])
            else:
                r = 2.5 * (i // 6 + 1) 
                angle = math.radians(60 * (i % 6))
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                safe_inner.append([x, y, 0.0])
        
        # Find best angle for safe inner
        min_side = float('inf')
        min_ang = 0.0
        for ang in [0, 15, 30, 45, 60, 75]:
            side = compute_min_side(safe_inner, (0.0, 0.0), ang)
            if side < min_side:
                min_side = side
                min_ang = ang
                best_inner_positions = safe_inner
        best_side = min_side
        best_outer_angle = min_ang

    return best_inner_positions, (0.0, 0.0), best_side, best_outer_angle
# EVOLVE-BLOCK-END
