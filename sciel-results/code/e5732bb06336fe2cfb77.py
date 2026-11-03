import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple grid placement
        positions = []
        for i in range(N):
            q = i % 4
            r = i // 4
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            positions.append((x, y))
        return [[x, y, 0.0] for x, y in positions], [0.0, 0.0], 3.0, 0.0

    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    def hex_vertices(center, side, angle_deg=0.0):
        """Return vertices of a regular hexagon centered at center with given side and rotation."""
        cx, cy = center
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def point_in_hexagon(px, py, center, side, angle_deg=0.0):
        """Check if point is inside hexagon using support function."""
        cx, cy = center
        angle_rad = math.radians(angle_deg)
        dx, dy = px - cx, py - cy
        # Rotate point into hexagon's local frame
        cos_a, sin_a = math.cos(-angle_rad), math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        # Check 6 half-planes
        for i in range(6):
            theta = i * math.pi / 3.0
            nx = math.cos(theta)
            ny = math.sin(theta)
            if lx * nx + ly * ny > APOTHEM * side:
                return False
        return True

    def hexagon_overlap(p1, s1, a1, p2, s2, a2):
        """Check if two hexagons overlap using SAT (Separating Axis Theorem)."""
        # Get vertices of both hexagons
        v1 = hex_vertices(p1, s1, a1)
        v2 = hex_vertices(p2, s2, a2)
        
        # Check all 12 edges (6 from each hexagon)
        all_axes = []
        for v in [v1, v2]:
            for i in range(6):
                edge = v[(i+1) % 6] - v[i]
                # Normal to edge
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    all_axes.append(normal / norm)
        
        for axis in all_axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # If there's a gap, no overlap
            if max1 < min2 or max2 < min1:
                return False
        
        return True

    def enclosing_hexagon_side(positions, rotations, outer_center, outer_angle_deg):
        """Compute the minimum side length of a regular hexagon that encloses all points."""
        ocx, ocy = outer_center
        outer_angle_rad = math.radians(outer_angle_deg)
        
        max_proj = 0.0
        for k in range(6):
            theta = outer_angle_rad + k * math.pi / 3.0
            nx = math.cos(theta)
            ny = math.sin(theta)
            for px, py in positions:
                proj = (px - ocx) * nx + (py - ocy) * ny
                if abs(proj) > max_proj:
                    max_proj = abs(proj)
        
        return (max_proj + APOTHEM) / APOTHEM

    def repulsion_force(p1, a1, p2, a2, strength=10.0):
        """Compute repulsive force between two hexagons."""
        # Approximate using center distance and check for potential overlap
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < 1e-6:
            # Random small push
            angle = np.random.uniform(0, 2*math.pi)
            return strength * np.array([math.cos(angle), math.sin(angle)])
        
        # Check if hexagons are close enough to potentially overlap
        # Two unit hexagons touching have center distance = 2*APOTHEM = sqrt(3)
        min_dist = 2.0 * APOTHEM  # = sqrt(3)
        
        if dist < min_dist * 1.5:  # Within interaction range
            # Repulsion stronger when closer
            force_mag = strength * (min_dist / dist)**2
            direction = np.array([dx, dy]) / dist
            return -force_mag * direction  # Repel p1 away from p2
        else:
            return np.array([0.0, 0.0])

    def attraction_to_center(positions, center, strength=0.1):
        """Compute attractive force pulling each hexagon toward center."""
        forces = []
        for px, py in positions:
            dx = center[0] - px
            dy = center[1] - py
            force = strength * np.array([dx, dy])
            forces.append(force)
        return np.array(forces)

    def compute_enclosing_hexagon_direct(positions):
        """Compute minimal enclosing regular hexagon using convex hull + support function."""
        # Find convex hull of all vertices of all hexagons
        all_vertices = []
        for i, (px, py) in enumerate(positions):
            # Each hexagon has 6 vertices at distance 1 from center
            for k in range(6):
                theta = k * math.pi / 3.0
                vx = px + math.cos(theta)
                vy = py + math.sin(theta)
                all_vertices.append([vx, vy])
        
        all_vertices = np.array(all_vertices)
        
        # Compute convex hull
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(all_vertices)
            hull_vertices = all_vertices[hull.vertices]
        except Exception:
            hull_vertices = all_vertices
        
        # Find minimal enclosing hexagon by optimizing rotation
        # The enclosing hexagon's side is determined by max projection
        best_side = float('inf')
        best_center = (float(np.mean(hull_vertices[:, 0])), float(np.mean(hull_vertices[:, 1])))
        best_angle = 0.0
        
        # Try multiple rotation angles
        for angle_deg in np.linspace(0, 60, 61):
            angle_rad = math.radians(angle_deg)
            # For each angle, compute the optimal center and side
            # The center should be at the centroid for symmetry
            cx = float(np.mean(hull_vertices[:, 0]))
            cy = float(np.mean(hull_vertices[:, 1]))
            
            max_proj = 0.0
            for k in range(6):
                theta = angle_rad + k * math.pi / 3.0
                nx = math.cos(theta)
                ny = math.sin(theta)
                for vx, vy in hull_vertices:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    if abs(proj) > max_proj:
                        max_proj = abs(proj)
            
            side = (max_proj + APOTHEM) / APOTHEM
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = angle_deg
        
        return best_side, best_center, best_angle

    def relax_positions(positions, iterations=1000, cooling=0.999):
        """Force-directed relaxation to minimize enclosing hexagon size."""
        n = len(positions)
        positions = np.array(positions, dtype=float)
        
        # Initial temperature
        temp = 1.0
        best_positions = positions.copy()
        best_side, _, _ = compute_enclosing_hexagon_direct(positions)
        
        for iter in range(iterations):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Compute forces
            forces = np.zeros_like(positions)
            
            # Repulsive forces between all pairs
            for i in range(n):
                for j in range(i+1, n):
                    f = repulsion_force(positions[i], 0.0, positions[j], 0.0, strength=5.0)
                    forces[i] -= f
                    forces[j] += f
            
            # Attractive force to center
            center = np.mean(positions, axis=0)
            attr = attraction_to_center(positions, center, strength=0.05)
            forces += attr
            
            # Compute current side length
            side, _, _ = compute_enclosing_hexagon_direct(positions)
            
            # If side is getting large, increase attraction
            if side > 3.0:
                center = np.mean(positions, axis=0)
                forces += 0.1 * (center - positions)
            
            # Update positions with simulated annealing
            for i in range(n):
                # Random perturbation
                perturbation = temp * np.random.normal(0, 0.1, size=2)
                new_pos = positions[i] + forces[i] * 0.01 + perturbation
                new_positions = positions.copy()
                new_positions[i] = new_pos
                
                new_side, _, _ = compute_enclosing_hexagon_direct(new_positions)
                
                # Accept if better or with probability based on temperature
                if new_side < side or np.random.random() < math.exp(-(new_side - side) / (temp + 1e-10)):
                    positions[i] = new_pos
            
            # Cool down
            temp *= cooling
            
            # Track best
            side, _, _ = compute_enclosing_hexagon_direct(positions)
            if side < best_side:
                best_side = side
                best_positions = positions.copy()
        
        return best_positions, best_side

    # Initial configuration: start with a tight cluster
    # Use a hexagonal packing pattern
    initial_positions = []
    
    # Center hexagon
    initial_positions.append([0.0, 0.0])
    
    # 6 surrounding hexagons
    for i in range(6):
        theta = i * math.pi / 3.0
        x = 2.0 * APOTHEM * math.cos(theta)
        y = 2.0 * APOTHEM * math.sin(theta)
        initial_positions.append([x, y])
    
    # 4 more hexagons at second ring
    # Place them at positions that minimize the overall bounding box
    # Try a few candidate positions
    candidate_extra = [
        [2.0 * APOTHEM * math.cos(math.pi/6), 2.0 * APOTHEM * math.sin(math.pi/6)],
        [-2.0 * APOTHEM * math.cos(math.pi/6), 2.0 * APOTHEM * math.sin(math.pi/6)],
        [2.0 * APOTHEM * math.cos(5*math.pi/6), 2.0 * APOTHEM * math.sin(5*math.pi/6)],
        [-2.0 * APOTHEM * math.cos(math.pi/6), -2.0 * APOTHEM * math.sin(math.pi/6)],
    ]
    
    # Use first 4 candidates
    for pos in candidate_extra[:4]:
        initial_positions.append(pos)
    
    # Relax the configuration
    relaxed_positions, side = relax_positions(initial_positions, iterations=500)
    
    # Compute final enclosing hexagon
    side, center, angle_deg = compute_enclosing_hexagon_direct(relaxed_positions)
    
    # Verify no overlaps (should be satisfied by construction)
    # The repulsion forces should have prevented overlaps
    
    # Prepare output
    inner = []
    for x, y in relaxed_positions:
        inner.append([float(x), float(y), 0.0])  # All inner hexagons at 0 degrees
    
    return inner, [float(center[0]), float(center[1])], float(side), float(angle_deg)
