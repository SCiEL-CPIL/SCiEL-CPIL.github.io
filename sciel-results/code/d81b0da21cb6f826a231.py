import numpy as np
import time
from scipy.optimize import minimize
import math

N = 12
SQRT3 = math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Return 6 vertices of a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_to_segment_dist(p, a, b):
        """Distance from point p to segment ab."""
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0, min(1, t))
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def point_in_polygon(p, poly):
        """Check if point p is inside convex polygon poly."""
        # Cross product method
        sign = 0
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i + 1) % len(poly)]
            cross = np.cross(b - a, p - a)
            if cross != 0:
                if sign == 0:
                    sign = 1 if cross > 0 else -1
                elif (cross > 0 and sign == -1) or (cross < 0 and sign == 1):
                    return False
        return True

    def min_distance_polygons(poly1, poly2):
        """Compute minimum distance between two convex polygons.
        Returns negative value if they overlap (approximation of penetration depth)."""
        # First check for overlap using SAT
        # If they overlap, return a negative penalty based on vertex penetration
        # If they don't, return the positive distance
        
        # SAT Check
        separating = False
        min_gap = np.inf
        
        all_normals = []
        for poly in [poly1, poly2]:
            for i in range(len(poly)):
                v1 = poly[i]
                v2 = poly[(i + 1) % len(poly)]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10: continue
                normal /= norm
                all_normals.append(normal)
                
        for normal in all_normals:
            proj1 = poly1 @ normal
            proj2 = poly2 @ normal
            max1 = np.max(proj1)
            min1 = np.min(proj1)
            max2 = np.max(proj2)
            min2 = np.min(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                gap = min2 - max1 if max1 < min2 else min1 - max2
                if gap < min_gap:
                    min_gap = gap
                separating = True
                # We can break early if we find a large gap, but we need the MIN gap for distance
                # Actually, if they are separated, the distance is the minimum gap over all axes
                # If they overlap, there is no separating axis
        
        if separating:
            return min_gap

        # If not separating, they overlap.
        # Estimate penetration depth.
        # Find the vertex of poly1 deepest inside poly2 and vice versa.
        max_depth = 0
        for p in poly1:
            if point_in_polygon(p, poly2):
                # Distance to boundary
                min_d = np.inf
                for i in range(len(poly2)):
                    d = point_to_segment_dist(p, poly2[i], poly2[(i+1)%len(poly2)])
                    if d < min_d: min_d = d
                max_depth = max(max_depth, min_d)
        
        for p in poly2:
            if point_in_polygon(p, poly1):
                min_d = np.inf
                for i in range(len(poly1)):
                    d = point_to_segment_dist(p, poly1[i], poly1[(i+1)%len(poly1)])
                    if d < min_d: min_d = d
                max_depth = max(max_depth, min_d)
        
        return -max_depth

    def compute_outer_side(vertices, outer_angle):
        """Compute minimal side length of outer hexagon for given vertices and outer angle."""
        # Vertices is a list of arrays, or a single array of all vertices
        # We flatten all vertices into one array
        all_verts = np.vstack(vertices)
        
        max_width = 0
        for k in range(6):
            angle = outer_angle + k * np.pi / 3
            nx = np.cos(angle)
            ny = np.sin(angle)
            proj = all_verts @ np.array([nx, ny])
            width = np.max(proj) - np.min(proj)
            if width > max_width:
                max_width = width
        
        return max_width / SQRT3

    def objective(params):
        # params: x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, outer_angle
        # 36 + 1 = 37 params
        
        pos = params[:36].reshape(12, 3)
        outer_angle = params[36]
        
        # Symmetry breaking: Center of mass at 0
        # This is handled by the optimizer naturally if we don't fix it, 
        # but fixing it reduces the search space.
        # Let's not fix it in the objective, but use it in initialization.
        
        inner_hexes = []
        for i in range(12):
            verts = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, pos[i, 2])
            inner_hexes.append(verts)
        
        # Calculate outer side
        side = compute_outer_side(inner_hexes, outer_angle)
        
        # Calculate overlap penalty
        penalty = 0
        for i in range(12):
            for j in range(i + 1, 12):
                d = min_distance_polygons(inner_hexes[i], inner_hexes[j])
                if d < 0:
                    penalty += (-d) ** 2 * 1000 # Strong penalty
        
        return side + penalty

    def generate_lattice_config(rows, cols, stagger=True):
        positions = []
        angles = []
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= 12: break
                x = c * SQRT3
                y = r * 1.5
                if stagger and r % 2 == 1:
                    x += SQRT3 / 2
                positions.append([x, y])
                angles.append(0.0)
                idx += 1
            if idx >= 12: break
        
        # Center the configuration
        if positions:
            pos_arr = np.array(positions)
            mean = pos_arr.mean(axis=0)
            pos_arr -= mean
            for i in range(len(pos_arr)):
                positions[i] = pos_arr[i]
        
        # Pad if necessary (should be exactly 12)
        while len(positions) < 12:
            positions.append([0, 0])
            angles.append(0)
            
        pos_flat = []
        for i in range(12):
            pos_flat.append(positions[i][0])
            pos_flat.append(positions[i][1])
            pos_flat.append(angles[i])
        pos_flat.append(0.0) # outer angle
        return np.array(pos_flat)

    # Initial configurations
    inits = []
    
    # 1. 3x4 Lattice
    inits.append(generate_lattice_config(3, 4, stagger=True))
    inits.append(generate_lattice_config(3, 4, stagger=False))
    
    # 2. 2x6 Lattice
    inits.append(generate_lattice_config(2, 6, stagger=True))
    inits.append(generate_lattice_config(2, 6, stagger=False))
    
    # 3. 4x3 Lattice
    inits.append(generate_lattice_config(4, 3, stagger=True))
    inits.append(generate_lattice_config(4, 3, stagger=False))
    
    # 4. Random
    np.random.seed(42)
    for _ in range(5):
        pos = np.random.uniform(-3, 3, (12, 2))
        ang = np.random.uniform(0, 2*np.pi, 12)
        pos -= pos.mean(axis=0)
        params = np.zeros(37)
        for i in range(12):
            params[3*i] = pos[i, 0]
            params[3*i+1] = pos[i, 1]
            params[3*i+2] = ang[i]
        params[36] = np.random.uniform(0, np.pi/3)
        inits.append(params)

    best_score = np.inf
    best_params = None
    
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, init, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success:
                final_side = compute_outer_side(
                    [hexagon_vertices(res.x[3*i], res.x[3*i+1], 1.0, res.x[3*i+2]) for i in range(12)],
                    res.x[36]
                )
                
                # Check validity
                valid = True
                hexes = [hexagon_vertices(res.x[3*i], res.x[3*i+1], 1.0, res.x[3*i+2]) for i in range(12)]
                for i in range(12):
                    for j in range(i+1, 12):
                        d = min_distance_polygons(hexes[i], hexes[j])
                        if d < -1e-6: # Allow tiny numerical errors
                            valid = False
                            break
                    if not valid: break
                
                if valid and final_side < best_score:
                    best_score = final_side
                    best_params = res.x
        except Exception as e:
            pass

    if best_params is None:
        # Fallback: widely spaced
        params = generate_lattice_config(3, 4, stagger=True)
        # Scale up to ensure no overlap
        for i in range(12):
            params[3*i] *= 2
            params[3*i+1] *= 2
        best_params = params
        best_score = compute_outer_side(
            [hexagon_vertices(best_params[3*i], best_params[3*i+1], 1.0, best_params[3*i+2]) for i in range(12)],
            best_params[36]
        )

    # Extract results
    inner = []
    for i in range(12):
        x = best_params[3*i]
        y = best_params[3*i+1]
        angle_deg = np.degrees(best_params[3*i+2])
        inner.append([x, y, angle_deg])
    
    outer_cx = 0.0
    outer_cy = 0.0
    outer_angle_deg = np.degrees(best_params[36])
    
    return inner, (outer_cx, outer_cy), best_score, outer_angle_deg

