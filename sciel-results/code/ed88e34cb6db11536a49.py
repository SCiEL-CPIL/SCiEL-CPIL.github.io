"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

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
        """Check if two convex hexagons overlap using SAT. Returns True if overlap."""
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

    def check_all_overlaps_and_penalty(positions, angles):
        """
        Returns a penalty value. 0 if no overlap, >0 if overlap.
        positions: list of (x, y)
        angles: list of angle_deg
        """
        verts = [hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(verts[i], verts[j]):
                    # Simple penalty: large constant. 
                    # To help gradient-free methods, we can add a distance-based term if we want,
                    # but constant is often sufficient for Nelder-Mead to push apart.
                    penalty += 100.0
        return penalty

    def compute_min_side(positions, angles, outer_angle_deg):
        """
        Returns the minimum side length of outer hexagon to contain all inner hexagons.
        Outer hexagon is centered at (0,0).
        """
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
        
        max_abs_proj = 0
        for normal in normals:
            proj = np.dot(all_verts, normal)
            # The hexagon must contain all points.
            # Distance from center to side is R_in.
            # We need max(proj) <= R_in and min(proj) >= -R_in.
            # So R_in >= max(abs(proj))
            max_abs_proj = max(max_abs_proj, np.max(np.abs(proj)))
        
        # Side length S = 2 * R_in / sqrt(3)
        side = 2 * max_abs_proj / math.sqrt(3)
        return side

    def objective(params):
        """
        params: array of length 12*3 + 1
        [x1, y1, th1, ..., x12, y12, th12, outer_angle]
        """
        if time.time() > deadline:
            return float('inf')
            
        positions = []
        angles = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            th = params[i*3+2]
            positions.append((x, y))
            angles.append(th)
        
        outer_angle = params[N*3]
        
        penalty = check_all_overlaps_and_penalty(positions, angles)
        
        if penalty > 0:
            # Add penalty to side length
            # We want to minimize side length, but overlap is bad.
            # Returning a large value ensures optimizer moves away.
            side = compute_min_side(positions, angles, outer_angle)
            return side + penalty
        
        side = compute_min_side(positions, angles, outer_angle)
        return side

    # Generate initial guesses
    def make_honeycomb(rows, cols, angle_deg=0.0):
        positions = []
        angles = []
        for r in range(rows):
            for c in range(cols):
                x = c * math.sqrt(3) + (r % 2) * math.sqrt(3) / 2
                y = r * 1.5
                positions.append((x, y))
                angles.append(angle_deg)
        if not positions:
            return None
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        positions = [(x - cx, y - cy) for x, y in positions]
        return positions, angles

    def make_ring(radius, angle_deg=0.0):
        positions = []
        angles = []
        for i in range(N):
            a = 2 * math.pi * i / N
            x = radius * math.cos(a)
            y = radius * math.sin(a)
            positions.append((x, y))
            angles.append(angle_deg)
        return positions, angles

    def make_grid(rows, cols, angle_deg=0.0):
        positions = []
        angles = []
        for r in range(rows):
            for c in range(cols):
                x = c * 2.0 # Square grid, might overlap, but optimizer fixes
                y = r * 2.0
                positions.append((x, y))
                angles.append(angle_deg)
        if not positions:
            return None
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        positions = [(x - cx, y - cy) for x, y in positions]
        return positions, angles

    candidates = []
    
    # Honeycomb 3x4
    pos, ang = make_honeycomb(4, 3, 0.0)
    if pos: candidates.append((pos, ang))
    
    # Honeycomb 2x6
    pos, ang = make_honeycomb(6, 2, 0.0)
    if pos: candidates.append((pos, ang))
    
    # Honeycomb 4x3 rotated
    pos, ang = make_honeycomb(4, 3, 30.0)
    if pos: candidates.append((pos, ang))
    
    # Ring
    pos, ang = make_ring(2.5, 0.0)
    if pos: candidates.append((pos, ang))
    
    # Ring smaller
    pos, ang = make_ring(2.0, 0.0)
    if pos: candidates.append((pos, ang))
    
    # Grid 3x4
    pos, ang = make_grid(4, 3, 0.0)
    if pos: candidates.append((pos, ang))
    
    # Random starts
    np.random.seed(42)
    for _ in range(20):
        pos = []
        ang = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            pos.append((x, y))
            ang.append(np.random.uniform(0, 60))
        # Center
        cx = sum(p[0] for p in pos) / N
        cy = sum(p[1] for p in pos) / N
        pos = [(x - cx, y - cy) for x, y in pos]
        candidates.append((pos, ang))

    best_score = float('inf')
    best_params = None

    # Optimization loop
    for idx, (pos, ang) in enumerate(candidates):
        if time.time() > deadline - 5:
            break
            
        # Construct initial params
        init_params = []
        for i in range(N):
            init_params.extend([pos[i][0], pos[i][1], ang[i]])
        init_params.append(0.0) # outer angle
        
        res = None
        try:
            # Use Nelder-Mead
            # maxiter is per optimization
            # We have ~150s / 25 candidates = 6s per candidate
            # Nelder-Mead 37D might take a few seconds.
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
        except Exception:
            continue
            
        if res is None:
            continue
            
        final_params = res.x
        final_score = res.fun
        
        # Check validity explicitly
        positions = []
        angles = []
        for i in range(N):
            positions.append((final_params[i*3], final_params[i*3+1]))
            angles.append(final_params[i*3+2])
        
        penalty = check_all_overlaps_and_penalty(positions, angles)
        
        if penalty == 0:
            side = compute_min_side(positions, angles, final_params[N*3])
            if side < best_score:
                best_score = side
                best_params = final_params
        else:
            # If penalty > 0, the score reported by optimizer includes penalty.
            # We only care about valid solutions.
            pass

    # If no valid solution found from optimization, fallback to a known valid one
    if best_params is None:
        # Use the first honeycomb candidate
        pos, ang = make_honeycomb(4, 3, 0.0)
        best_params = []
        for i in range(N):
            best_params.extend([pos[i][0], pos[i][1], ang[i]])
        best_params.append(0.0)
        best_score = compute_min_side(pos, ang, 0.0)

    # Refine the best solution with a tighter optimizer if time permits
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
            if res.fun < best_score:
                # Verify validity
                positions = []
                angles = []
                for i in range(N):
                    positions.append((res.x[i*3], res.x[i*3+1]))
                    angles.append(res.x[i*3+2])
                if check_all_overlaps_and_penalty(positions, angles) == 0:
                    best_score = res.fun
                    best_params = res.x
        except Exception:
            pass

    # Construct return values
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        ang = best_params[i*3+2]
        inner.append([x, y, ang])
    
    outer_center = (0.0, 0.0)
    outer_side = best_score
    outer_angle = best_params[N*3]
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
