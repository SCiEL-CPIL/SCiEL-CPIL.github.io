import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 11

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Unit hexagon vertices (flat-top, vertex at angle 0)
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    # Normals of unit hexagon
    unit_normals = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    def get_hex_verts(x, y, angle_deg):
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = unit_hex_verts @ R.T
        verts += np.array([x, y])
        return verts
    
    def calc_side(centers, angles, outer_rotation=0.0):
        """Calculate required outer hexagon side length."""
        angle_rad = math.radians(outer_rotation)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        normals = unit_normals @ R.T
        
        max_apothem = 0.0
        for k in range(N):
            verts = get_hex_verts(centers[k, 0], centers[k, 1], angles[k])
            for v in verts:
                for n in normals:
                    proj = float(np.dot(v, n))
                    if proj > max_apothem:
                        max_apothem = proj
        return 2.0 * max_apothem / math.sqrt(3)
    
    def hexagon_overlap_test(centers, angles, eps=1e-7):
        """Check if any two hexagons overlap using vertex-in-polygon test."""
        def point_in_hexagon(px, py, cx, cy, angle_deg):
            dx = px - cx
            dy = py - cy
            angle_rad = math.radians(-angle_deg)
            c, s = math.cos(angle_rad), math.sin(angle_rad)
            lx = c * dx - s * dy
            ly = s * dx + c * dy
            for k in range(6):
                v1 = unit_hex_verts[k]
                v2 = unit_hex_verts[(k + 1) % 6]
                ex = v2[0] - v1[0]
                ey = v2[1] - v1[1]
                cross = ex * (ly - v1[1]) - ey * (lx - v1[0])
                if cross > eps:
                    return False
            return True
        
        for i in range(N):
            for j in range(i + 1, N):
                verts_i = get_hex_verts(centers[i, 0], centers[i, 1], angles[i])
                verts_j = get_hex_verts(centers[j, 0], centers[j, 1], angles[j])
                for v in verts_i:
                    if point_in_hexagon(v[0], v[1], centers[j, 0], centers[j, 1], angles[j]):
                        return False
                for v in verts_j:
                    if point_in_hexagon(v[0], v[1], centers[i, 0], centers[i, 1], angles[i]):
                        return False
        return True
    
    def check_validity(centers, angles, outer_rot, side_limit):
        """Check if configuration is valid (no overlap, fits in side_limit)."""
        if not hexagon_overlap_test(centers, angles):
            return False
        side = calc_side(centers, angles, outer_rot)
        return side <= side_limit + 1e-6
    
    def objective_direct(centers_flat, angles, outer_rotation):
        """
        Objective function for optimization.
        centers_flat: 2N array of [x0, y0, x1, y1, ...]
        angles: N array of angles
        outer_rotation: scalar
        
        Returns: side_length + overlap_penalty
        """
        centers = centers_flat.reshape(N, 2)
        
        # Calculate side length
        side = calc_side(centers, angles, outer_rotation)
        
        # Overlap penalty using center distance
        min_dist = math.sqrt(3) - 0.05
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 1000
        
        return side + penalty
    
    def nelder_mead_optimize(init_centers, init_angles, init_outer_rot, max_iters=5000):
        """Run Nelder-Mead optimization to minimize side length."""
        # Flatten parameters: [x0, y0, x1, y1, ..., angle0, angle1, ..., outer_rot]
        x0 = np.concatenate([init_centers.flatten(), init_angles, [init_outer_rot]])
        
        def obj(x):
            if time.time() > deadline - 5:
                return 1000.0
            centers = x[:2*N].reshape(N, 2)
            angles = x[2*N:3*N]
            outer_rot = x[3*N]
            return objective_direct(centers.flatten(), angles, outer_rot)
        
        try:
            result = minimize(obj, x0, method='Nelder-Mead', 
                             options={'maxiter': max_iters, 'xatol': 1e-8, 'fatol': 1e-8})
            if result.success or result.fun < 1000.0:
                centers = result.x[:2*N].reshape(N, 2)
                angles = result.x[2*N:3*N]
                outer_rot = result.x[3*N]
                return centers, angles, outer_rot, result.fun
        except Exception as e:
            pass
        
        return init_centers, init_angles, init_outer_rot, 1000.0
    
    def differential_evolution_optimize(bounds, max_iter=1000, pop_size=50):
        """Run differential evolution to find good initial configuration."""
        def obj(x):
            if time.time() > deadline - 5:
                return 1000.0
            centers = x[:2*N].reshape(N, 2)
            angles = x[2*N:3*N]
            outer_rot = x[3*N]
            return objective_direct(centers.flatten(), angles, outer_rot)
        
        try:
            result = differential_evolution(obj, bounds, maxiter=max_iter, 
                                           popsize=pop_size, seed=42, 
                                           tol=1e-6, polish=True)
            if result.success or result.fun < 1000.0:
                centers = result.x[:2*N].reshape(N, 2)
                angles = result.x[2*N:3*N]
                outer_rot = result.x[3*N]
                return centers, angles, outer_rot, result.fun
        except Exception as e:
            pass
        
        return None, None, None, 1000.0
    
    # Generate initial configurations
    np.random.seed(42)
    d = math.sqrt(3)
    
    initial_configs = []
    
    # Config 1: 7-hex cluster + 4 extensions
    centers1 = [(0, 0)]
    for k in range(6):
        angle = k * math.pi / 3
        centers1.append((d * math.cos(angle), d * math.sin(angle)))
    for k in [0, 1, 2, 3]:
        angle = k * math.pi / 3 + math.pi / 6
        x = 2 * d * math.cos(angle)
        y = 2 * d * math.sin(angle)
        centers1.append((x, y))
    centers1 = np.array(centers1[:N])
    angles1 = np.zeros(N)
    initial_configs.append((centers1, angles1))
    
    # Config 2: Two rows offset
    centers2 = []
    for i in range(6):
        centers2.append((i * d - 2.5 * d, 0))
    for i in range(5):
        centers2.append((i * d - 2.0 * d + d / 2, 1.5))
    centers2 = np.array(centers2[:N])
    angles2 = np.zeros(N)
    initial_configs.append((centers2, angles2))
    
    # Config 3: Linear
    centers3 = np.array([(i * d - (N - 1) * d / 2, 0) for i in range(N)])
    angles3 = np.zeros(N)
    initial_configs.append((centers3, angles3))
    
    # Config 4: Tighter hexagonal
    centers4 = [
        (-d / 2, 0), (d / 2, 0),
        (-d, 1.5), (0, 1.5), (d, 1.5),
        (-d / 2, 3.0), (d / 2, 3.0),
        (-d, -1.5), (0, -1.5), (d, -1.5),
        (0, -3.0)
    ]
    centers4 = np.array(centers4[:N])
    angles4 = np.zeros(N)
    initial_configs.append((centers4, angles4))
    
    # Config 5: Random
    centers5 = np.random.uniform(-3 * d, 3 * d, (N, 2))
    angles5 = np.random.uniform(-10, 10, N)
    initial_configs.append((centers5, angles5))
    
    # Config 6: 7-hex + 4 in a row
    centers6 = [(0, 0)]
    for k in range(6):
        angle = k * math.pi / 3
        centers6.append((d * math.cos(angle), d * math.sin(angle)))
    for k in range(4):
        centers6.append((3 * d + k * d, 0))
    centers6 = np.array(centers6[:N])
    angles6 = np.zeros(N)
    initial_configs.append((centers6, angles6))
    
    # Best solution tracking
    best_side = float('inf')
    best_centers = None
    best_angles = None
    best_outer_rot = 0.0
    
    # Phase 1: Run Nelder-Mead from each initial config
    for config_idx, (init_centers, init_angles) in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        for outer_rot in [0.0, 30.0]:
            if time.time() > deadline - 30:
                break
            
            result_centers, result_angles, result_outer_rot, obj_val = nelder_mead_optimize(
                init_centers, init_angles, outer_rot, max_iters=3000
            )
            
            if check_validity(result_centers, result_angles, result_outer_rot, obj_val):
                side = calc_side(result_centers, result_angles, result_outer_rot)
                if side < best_side:
                    best_side = side
                    best_centers = result_centers.copy()
                    best_angles = result_angles.copy()
                    best_outer_rot = result_outer_rot
    
    # Phase 2: If we have a good solution, refine with more iterations
    if best_centers is not None and time.time() < deadline - 20:
        for outer_rot in [best_outer_rot, 0.0, 30.0]:
            if time.time() > deadline - 20:
                break
            
            result_centers, result_angles, result_outer_rot, obj_val = nelder_mead_optimize(
                best_centers, best_angles, outer_rot, max_iters=5000
            )
            
            if check_validity(result_centers, result_angles, result_outer_rot, obj_val):
                side = calc_side(result_centers, result_angles, result_outer_rot)
                if side < best_side:
                    best_side = side
                    best_centers = result_centers.copy()
                    best_angles = result_angles.copy()
                    best_outer_rot = result_outer_rot
    
    # Phase 3: Try differential evolution if time permits
    if time.time() < deadline - 15:
        # Define bounds for DE
        bounds = [(-5, 5)] * (2 * N) + [(-30, 30)] * N + [(0, 60)]
        
        de_centers, de_angles, de_outer_rot, de_obj = differential_evolution_optimize(
            bounds, max_iter=500, pop_size=30
        )
        
        if de_centers is not None and check_validity(de_centers, de_angles, de_outer_rot, de_obj):
            side = calc_side(de_centers, de_angles, de_outer_rot)
            if side < best_side:
                best_side = side
                best_centers = de_centers.copy()
                best_angles = de_angles.copy()
                best_outer_rot = de_outer_rot
    
    # Phase 4: Final polish with Nelder-Mead
    if best_centers is not None and time.time() < deadline - 5:
        for outer_rot in [best_outer_rot]:
            if time.time() > deadline - 5:
                break
            
            result_centers, result_angles, result_outer_rot, obj_val = nelder_mead_optimize(
                best_centers, best_angles, outer_rot, max_iters=8000
            )
            
            if check_validity(result_centers, result_angles, result_outer_rot, obj_val):
                side = calc_side(result_centers, result_angles, result_outer_rot)
                if side < best_side:
                    best_side = side
                    best_centers = result_centers.copy()
                    best_angles = result_angles.copy()
                    best_outer_rot = result_outer_rot
    
    # Construct output
    if best_centers is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = []
    for i in range(N):
        x, y = best_centers[i, 0], best_centers[i, 1]
        ang = best_angles[i]
        inner.append([float(x), float(y), float(ang)])
    
    return inner, (0.0, 0.0), float(best_side), float(best_outer_rot)
