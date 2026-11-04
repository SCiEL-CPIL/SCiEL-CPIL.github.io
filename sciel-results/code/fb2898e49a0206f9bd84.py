import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120.0  # safety margin before 150s kill
    
    def hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle + i * math.pi / 3
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts
    
    def sat_min_separation(verts_a, verts_b):
        """Compute minimum separation between two convex polygons using SAT.
        Returns the minimum distance (positive if separated, negative if overlapping)."""
        # Get all edges from both polygons
        edges = []
        for v in verts_a:
            edges.append(v)
        for v in verts_b:
            edges.append(v)
        
        # Get all unique edge normals
        all_edges = []
        for i in range(len(verts_a)):
            p1 = verts_a[i]
            p2 = verts_a[(i+1) % len(verts_a)]
            all_edges.append((p1, p2))
        for i in range(len(verts_b)):
            p1 = verts_b[i]
            p2 = verts_b[(i+1) % len(verts_b)]
            all_edges.append((p1, p2))
        
        min_sep = float('inf')
        for p1, p2 in all_edges:
            # Edge vector
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            edge_len = math.sqrt(ex*ex + ey*ey)
            if edge_len < 1e-12:
                continue
            # Normal vector
            nx = -ey / edge_len
            ny = ex / edge_len
            
            # Project all vertices onto normal
            proj_a = [v[0]*nx + v[1]*ny for v in verts_a]
            proj_b = [v[0]*nx + v[1]*ny for v in verts_b]
            
            min_a = min(proj_a)
            max_a = max(proj_a)
            min_b = min(proj_b)
            max_b = max(proj_b)
            
            # Check for overlap on this axis
            if max_a < min_b or max_b < min_a:
                # Separated on this axis
                sep = min(min_b - max_a, min_a - max_b)
                min_sep = min(min_sep, sep)
            else:
                # Overlapping on this axis, separation is 0 or negative
                # The actual separation is 0 if they just touch, negative if overlapping
                # For SAT, if they overlap on all axes, they intersect
                # We need to find the minimum separation, which could be negative
                # If overlapping on this axis, contribution to min_sep is 0 (or negative)
                # Actually for minimum separation, if they overlap on at least one axis,
                # the min separation is determined by the axis with least overlap
                pass
        
        # If no separating axis found, polygons overlap
        # We need a different approach: compute actual minimum distance
        # For simplicity, use a point-to-edge distance approach
        return _min_distance_polygons(verts_a, verts_b)
    
    def _min_distance_polygons(verts_a, verts_b):
        """Compute minimum distance between two convex polygons.
        Returns negative if overlapping."""
        # Check if any vertex of A is inside B or vice versa
        def point_in_polygon(px, py, verts):
            n = len(verts)
            inside = False
            for i in range(n):
                x1, y1 = verts[i]
                x2, y2 = verts[(i+1) % n]
                if ((y1 > py) != (y2 > py)) and (px < (x2-x1)*(py-y1)/(y2-y1)+x1):
                    inside = not inside
            return inside
        
        # Check if any vertex is inside the other polygon
        for v in verts_a:
            if point_in_polygon(v[0], v[1], verts_b):
                return -1.0  # overlapping
        for v in verts_b:
            if point_in_polygon(v[0], v[1], verts_a):
                return -1.0  # overlapping
        
        # Compute minimum distance between edges
        min_dist = float('inf')
        for i in range(len(verts_a)):
            p1 = verts_a[i]
            p2 = verts_a[(i+1) % len(verts_a)]
            for j in range(len(verts_b)):
                p3 = verts_b[j]
                p4 = verts_b[(j+1) % len(verts_b)]
                d = _seg_seg_dist(p1, p2, p3, p4)
                min_dist = min(min_dist, d)
        
        return min_dist
    
    def _seg_seg_dist(p1, p2, p3, p4):
        """Minimum distance between two line segments."""
        # Check if segments intersect
        if _segments_intersect(p1, p2, p3, p4):
            return 0.0
        
        # Point-to-segment distances
        d1 = _pt_seg_dist(p1, p3, p4)
        d2 = _pt_seg_dist(p2, p3, p4)
        d3 = _pt_seg_dist(p3, p1, p2)
        d4 = _pt_seg_dist(p4, p1, p2)
        
        return min(d1, d2, d3, d4)
    
    def _pt_seg_dist(px, py, ax, ay, bx, by):
        """Distance from point to segment."""
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.sqrt((px-ax)**2 + (py-ay)**2)
        t = ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t))
        cx = ax + t*dx
        cy = ay + t*dy
        return math.sqrt((px-cx)**2 + (py-cy)**2)
    
    def _segments_intersect(p1, p2, p3, p4):
        """Check if two segments intersect."""
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    def compute_required_side(positions, angles, outer_center, outer_angle):
        """Compute the minimum side length of outer hexagon to contain all inner hexagons."""
        ox, oy = outer_center
        oa = math.radians(outer_angle)
        
        # Outer hexagon vertices
        outer_verts = []
        # We'll compute the required side by checking how far each inner hex vertex is
        # from the outer hexagon center in the direction of each outer hexagon normal
        
        # For a regular hexagon with side s, the distance from center to each side (apothem) is s*sqrt(3)/2
        # The distance from center to each vertex is s
        
        # For each inner hexagon vertex, we need to find the minimum s such that
        # the vertex is inside the outer hexagon
        
        # A point (x,y) is inside a regular hexagon centered at origin with side s and angle a
        # if for each of the 6 face normals, the projection is within apothem = s*sqrt(3)/2
        
        max_s = 0.0
        for pos, ang in zip(positions, angles):
            px, py = pos
            ia = math.radians(ang)
            # Get vertices of inner hexagon
            for k in range(6):
                va = ia + k * math.pi / 3
                vx = px + math.cos(va)
                vy = py + math.sin(va)
                
                # Check against outer hexagon faces
                # Outer hexagon face normals
                for m in range(6):
                    na = oa + m * math.pi / 3
                    # Normal direction
                    nx = math.cos(na + math.pi/2)  # perpendicular to face direction
                    ny = math.sin(na + math.pi/2)
                    
                    # Actually, for a regular hexagon, the face normal at angle na is in direction na
                    # The face is at distance s*sqrt(3)/2 from center in direction na
                    # Point is inside if dot(point - center, normal_dir) <= s*sqrt(3)/2
                    
                    # Face normal direction (outward) for face m
                    fnx = math.cos(oa + m * math.pi / 3)
                    fny = math.sin(oa + m * math.pi / 3)
                    
                    # Relative position from outer center
                    rx = vx - ox
                    ry = vy - oy
                    
                    # Projection onto face normal
                    proj = rx * fnx + ry * fny
                    
                    # Must be <= apothem = s * sqrt(3) / 2
                    # So s >= proj * 2 / sqrt(3)
                    if proj > 0:
                        required_s = proj * 2.0 / math.sqrt(3.0)
                        max_s = max(max_s, required_s)
        
        return max_s
    
    def energy(positions, angles, outer_center, outer_angle):
        """Energy = required outer side length."""
        return compute_required_side(positions, angles, outer_center, outer_angle)
    
    # Initialize with honeycomb-like arrangement
    def initial_config():
        # Honeycomb positions for 12 hexagons
        # Use axial coordinates converted to Cartesian
        # Hex centers in honeycomb: distance between adjacent centers = 2*apothem = sqrt(3)
        dx = math.sqrt(3)
        dy = 1.5
        
        # Arrange in a compact cluster
        # Row-based: rows of 2,3,4,3 or similar
        # Let's use a 3-4-3-2 pattern or just place them densely
        
        # Simple approach: place in a grid-like honeycomb
        configs = []
        # Center at origin, place 12 hexagons in honeycomb
        # Axial coords: (q, r) -> x = dx*(q + r/2), y = dy*r
        
        # Pick 12 compact positions
        axial_positions = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # 7 hexagons (center + ring)
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1),  # 5 more = 12 total
        ]
        
        positions = []
        for q, r in axial_positions:
            x = dx * (q + r / 2.0)
            y = dy * r
            positions.append([x, y])
        
        angles = [0.0] * N
        return positions, angles
    
    positions, angles = initial_config()
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    # Simulated Annealing
    best_energy = energy(positions, angles, outer_center, outer_angle)
    best_positions = [p[:] for p in positions]
    best_angles = angles[:]
    best_outer_center = outer_center[:]
    best_outer_angle = outer_angle
    
    T = 1.0
    T_min = 0.001
    cooling = 0.995
    max_iter = 50000
    
    np.random.seed(42)
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb
        new_positions = [p[:] for p in positions]
        new_angles = angles[:]
        
        # Randomly move one hexagon
        idx = np.random.randint(N)
        new_positions[idx][0] += np.random.normal(0, 0.5)
        new_positions[idx][1] += np.random.normal(0, 0.5)
        new_angles[idx] += np.random.normal(0, 10.0)
        
        # Occasionally move outer center
        if np.random.random() < 0.3:
            new_outer_center = [outer_center[0] + np.random.normal(0, 0.5),
                               outer_center[1] + np.random.normal(0, 0.5)]
        else:
            new_outer_center = outer_center[:]
        
        # Occasionally rotate outer
        if np.random.random() < 0.2:
            new_outer_angle = outer_angle + np.random.normal(0, 5.0)
        else:
            new_outer_angle = outer_angle
        
        new_energy = energy(new_positions, new_angles, new_outer_center, new_outer_angle)
        
        # Check collision (inner hexagons must not overlap)
        collision = False
        for i in range(N):
            if collision:
                break
            for j in range(i+1, N):
                verts_i = hex_vertices(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i])
                verts_j = hex_vertices(new_positions[j][0], new_positions[j][1], 1.0, new_angles[j])
                sep = sat_min_separation(verts_i, verts_j)
                if sep < -0.001:  # overlapping
                    collision = True
                    break
        
        if collision:
            # Reject colliding configurations
            new_energy = new_energy + 10.0  # penalty
        
        # Accept/reject
        if new_energy < best_energy:
            best_energy = new_energy
            best_positions = [p[:] for p in new_positions]
            best_angles = new_angles[:]
            best_outer_center = new_outer_center[:]
            best_outer_angle = new_outer_angle
        
        delta = new_energy - energy(positions, angles, outer_center, outer_angle)
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
            positions = new_positions
            angles = new_angles
            outer_center = new_outer_center
            outer_angle = new_outer_angle
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final polish: local optimization
    if time.time() < deadline - 10.0:
        # Gradient-free local search
        step_size = 0.1
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for idx in range(N):
                for dim in range(2):
                    for delta in [step_size, -step_size]:
                        test_pos = [p[:] for p in best_positions]
                        test_pos[idx][dim] += delta
                        test_energy = energy(test_pos, best_angles, best_outer_center, best_outer_angle)
                        
                        # Check collisions
                        col = False
                        for i in range(N):
                            if col:
                                break
                            for j in range(i+1, N):
                                vi = hex_vertices(test_pos[i][0], test_pos[i][1], 1.0, best_angles[i])
                                vj = hex_vertices(test_pos[j][0], test_pos[j][1], 1.0, best_angles[j])
                                sep = sat_min_separation(vi, vj)
                                if sep < -0.001:
                                    col = True
                                    break
                        
                        if not col and test_energy < best_energy:
                            best_energy = test_energy
                            best_positions = test_pos
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                step_size *= 0.7
                if step_size < 0.001:
                    break
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (best_outer_center[0], best_outer_center[1]), best_energy, best_outer_angle
# EVOLVE-BLOCK-END
