"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Helper: get vertices of a regular hexagon with side length 1
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)
    
    # Helper: check if two hexagons overlap using SAT (Separating Axis Theorem)
    def hexagons_overlap(v1, v2):
        """Check if two convex hexagons overlap using SAT."""
        axes = []
        for i in range(6):
            edge = v1[i] - v1[(i+1) % 6]
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
            edge2 = v2[i] - v2[(i+1) % 6]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 or max2 < min1:
                return False
        return True
        
    def check_all_overlaps(positions, angles):
        """Check if any pair of hexagons overlaps."""
        verts = [hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)]
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(verts[i], verts[j]):
                    return True
        return False

    # Helper: compute minimum outer hexagon side length needed to contain all inner hexagons
    def compute_min_side(positions, angles, outer_angle_deg):
        """
        positions: list of (x, y) for each inner hexagon
        angles: list of rotation angles for each inner hexagon
        outer_angle_deg: rotation of outer hexagon
        Returns the minimum side length of outer hexagon to contain all inner hexagons.
        """
        # Get all vertices of all inner hexagons
        all_verts = []
        for i in range(N):
            cx, cy = positions[i]
            a = angles[i]
            verts = hex_vertices(cx, cy, a)
            all_verts.append(verts)
        
        if not all_verts:
            return 0.0
            
        all_verts = np.vstack(all_verts)
        
        theta = math.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            a = theta + k * math.pi / 3
            normals.append(np.array([math.cos(a), math.sin(a)]))
        
        max_proj = 0
        for normal in normals:
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        
        side = 2 * max_proj / math.sqrt(3)
        return side

    def get_config_score(positions, angles, outer_angle):
        """Return side length if valid, else infinity."""
        if check_all_overlaps(positions, angles):
            return float('inf')
        return compute_min_side(positions, angles, outer_angle)

    def make_honeycomb(rows, cols, angle_deg=0.0, outer_angle=0.0):
        """Generate a honeycomb grid configuration."""
        positions = []
        angles = []
        for r in range(rows):
            for c in range(cols):
                x = c * math.sqrt(3) + (r % 2) * math.sqrt(3) / 2
                y = r * 1.5
                positions.append((x, y))
                angles.append(angle_deg)
        
        # Center the configuration
        if not positions:
            return [], [], outer_angle
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        positions = [(x - cx, y - cy) for x, y in positions]
        
        return positions, angles, outer_angle

    def make_circular(radius, angle_deg=0.0, outer_angle=0.0):
        """Generate a circular arrangement."""
        positions = []
        angles = []
        for i in range(N):
            a = 2 * math.pi * i / N
            x = radius * math.cos(a)
            y = radius * math.sin(a)
            positions.append((x, y))
            angles.append(angle_deg)
        return positions, angles, outer_angle

    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    # List of candidate configurations to try
    # Each item is (positions, angles, outer_angle)
    candidates = []
    
    # 1. Honeycomb 3x4 (standard)
    candidates.append(make_honeycomb(4, 3, 0.0, 0.0))
    # 2. Honeycomb 3x4 (rotated hexagons)
    candidates.append(make_honeycomb(4, 3, 30.0, 0.0))
    # 3. Honeycomb 2x6
    candidates.append(make_honeycomb(2, 6, 0.0, 0.0))
    # 4. Honeycomb 6x2
    candidates.append(make_honeycomb(6, 2, 0.0, 0.0))
    
    # 5. Circular Cluster 1+6+5
    def make_circular_cluster(indices):
        positions = [(0.0, 0.0)]
        angles = [0.0]
        for k in range(6):
            a = k * math.pi / 3
            x = math.sqrt(3) * math.cos(a)
            y = math.sqrt(3) * math.sin(a)
            positions.append((x, y))
            angles.append(0.0)
            
        ring2_coords = [
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1),
            (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -2), (2, -1)
        ]
        
        for idx in indices:
            u, v = ring2_coords[idx]
            x = u * math.sqrt(3) + v * math.sqrt(3) / 2
            y = v * 1.5
            positions.append((x, y))
            angles.append(0.0)
            
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        positions = [(x - cx, y - cy) for x, y in positions]
        return positions, angles

    # Try different selections for the 5 outer hexagons
    candidates.append((make_circular_cluster([0, 2, 4, 6, 8])[0], make_circular_cluster([0, 2, 4, 6, 8])[1], 0.0))
    candidates.append((make_circular_cluster([0, 2, 4, 6, 10])[0], make_circular_cluster([0, 2, 4, 6, 10])[1], 0.0))
    candidates.append((make_circular_cluster([0, 3, 5, 7, 9])[0], make_circular_cluster([0, 3, 5, 7, 9])[1], 0.0))
    
    # Try different outer angles for the best candidates
    for base in [make_honeycomb(4, 3, 0.0, 0.0), make_circular_cluster([0, 2, 4, 6, 8])]:
        for oa in [0, 15, 30, 45, 60, 75, 90]:
            candidates.append((base[0], base[1], oa))

    # Local search / Shake function
    def local_search(positions, angles, outer_angle, iterations=1000):
        """
        Perform local search by randomly perturbing positions.
        Uses Simulated Annealing to escape local minima.
        """
        current_pos = [p[:] for p in positions]
        current_ang = angles[:]
        current_score = get_config_score(current_pos, current_ang, outer_angle)
        
        if current_score == float('inf'):
            return current_pos, current_ang, outer_angle, float('inf')
            
        best_local_score = current_score
        best_local_pos = [p[:] for p in current_pos]
        best_local_ang = current_ang[:]
        
        temp = 1.0
        cool_rate = 0.999
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
                
            # Pick a random hexagon to move
            i = np.random.randint(N)
            
            # Random perturbation, step size decreases over time?
            # Keep it constant for now, SA handles acceptance
            step = 0.2
            dx = np.random.uniform(-step, step)
            dy = np.random.uniform(-step, step)
            
            new_pos = current_pos[:]
            new_pos[i] = (current_pos[i][0] + dx, current_pos[i][1] + dy)
            
            # Check validity
            if not check_all_overlaps(new_pos, current_ang):
                new_score = compute_min_side(new_pos, current_ang, outer_angle)
                
                # Accept if better
                if new_score < current_score:
                    current_pos = new_pos
                    current_score = new_score
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local_pos = [p[:] for p in new_pos]
                        best_local_ang = current_ang[:]
                else:
                    # Accept worse with probability exp(-delta/temp)
                    delta = new_score - current_score
                    if np.random.rand() < math.exp(-delta / temp):
                        current_pos = new_pos
                        current_score = new_score
                        if new_score < best_local_score:
                            best_local_score = new_score
                            best_local_pos = [p[:] for p in new_pos]
                            best_local_ang = current_ang[:]
                        
            temp *= cool_rate

        return best_local_pos, best_local_ang, outer_angle, best_local_score

    # Evaluate candidates and run local search on the best ones
    for pos, ang, oa in candidates:
        if time.time() > deadline - 10:
            break
            
        score = get_config_score(pos, ang, oa)
        
        if score < best_score:
            best_score = score
            best_positions = pos
            best_angles = ang
            best_outer_angle = oa
            
        # If the initial candidate is valid, try to improve it with local search
        if score < float('inf'):
            # Run a short local search to see if we can improve
            # Limit iterations based on time
            remaining_time = deadline - time.time()
            if remaining_time > 5:
                # Estimate iterations possible (rough guess)
                # We can afford more iterations with the optimized check
                iters = int(min(2000, remaining_time * 100))
                new_pos, new_ang, new_oa, new_score = local_search(pos, ang, oa, iterations=iters)
                
                if new_score < best_score:
                    best_score = new_score
                    best_positions = new_pos
                    best_angles = new_ang
                    best_outer_angle = new_oa

    # Fallback: If no valid configuration was found (should not happen with honeycomb),
    # generate a very sparse configuration to ensure validity.
    if best_positions is None or best_score == float('inf'):
        # Sparse grid
        positions = []
        angles = []
        for i in range(N):
            r = i // 4
            c = i % 4
            positions.append((c * 4.0, r * 4.0))
            angles.append(0.0)
        best_positions = positions
        best_angles = angles
        best_outer_angle = 0.0
        best_score = compute_min_side(positions, angles, 0.0)

    # Final cleanup: ensure we return the best valid configuration
    # Re-center to be safe (though local search kept them centered-ish)
    if best_positions:
        cx = sum(p[0] for p in best_positions) / N
        cy = sum(p[1] for p in best_positions) / N
        final_positions = [(x - cx, y - cy) for x, y in best_positions]
    else:
        final_positions = [(0,0)] * N
        final_angles = [0] * N
        final_outer = 0
        best_score = 10.0 # Large penalty

    inner = [[final_positions[i][0], final_positions[i][1], best_angles[i]] for i in range(N)]
    
    return inner, (0.0, 0.0), best_score, best_outer_angle
# EVOLVE-BLOCK-END
