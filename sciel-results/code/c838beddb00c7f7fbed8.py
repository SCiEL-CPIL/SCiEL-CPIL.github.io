import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def project(vertices, nx, ny):
        projs = np.dot(vertices, [nx, ny])
        return np.min(projs), np.max(projs)

    def are_disjoint_sat(v1, v2, eps=1e-9):
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        
        return False

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        max_proj = 0.0
        otheta_rad = math.radians(otheta)
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def objective_full(params):
        if time.time() > deadline - 5:
            return 1e9

        centers = []
        for i in range(12):
            centers.append([params[i * 2], params[i * 2 + 1]])
        angles = [params[24 + i] for i in range(12)]
        ocx, ocy = params[36], params[37]
        otheta = params[38]
        
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        # Soft penalty for overlaps based on center distance
        min_dist = math.sqrt(3)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return side + 10.0 * penalty

    def is_valid(centers, angles):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
            
        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint_sat(verts[i], verts[j]):
                    return False
        return True

    def generate_hexagonal_cluster():
        """
        Generates a 12-hexagon cluster based on a hexagonal lattice.
        Returns list of (x, y) centers and list of angles.
        """
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
            
        ring2_candidates = []
        basis = [u, v, v - u, -u, -v, u - v]
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
                        
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
            
        angles = [30.0] * 12
        return centers, angles

    def generate_compact_cluster():
        """
        Generates a slightly compressed or different topology cluster.
        """
        # Try a rectangular-ish or different packing
        d = math.sqrt(3) + 0.01
        centers = []
        # 3 rows of 4? Or 2 rows of 6?
        # Let's try 3-4-3-2 arrangement or similar
        # Standard hex packing is best. Let's perturb the angles slightly to see if it helps?
        # For now, use the same lattice but different initial rotation
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
            
        ring2_candidates = []
        basis = [u, v, v - u, -u, -v, u - v]
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
                            
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
            
        # Perturb angles
        angles = [30.0 + random.uniform(-5, 5) for _ in range(12)]
        return centers, angles

    def rigid_body_optimize(centers, angles):
        """
        Optimizes the global rotation, translation, and outer hexagon orientation
        for a fixed set of inner centers/angles.
        """
        def obj_rigid(params):
            if time.time() > deadline - 5:
                return 1e9
            rot_deg, tx, ty, otheta = params
            rot_rad = math.radians(rot_deg)
            c = math.cos(rot_rad)
            s = math.sin(rot_rad)
            
            new_centers = []
            for cx, cy in centers:
                nx = cx * c - cy * s + tx
                ny = cx * s + cy * c + ty
                new_centers.append([nx, ny])
            
            # Outer center is fixed at 0,0 for simplicity in this sub-problem?
            # No, we can optimize outer center too, but relative position is what matters.
            # Let's fix outer center at 0,0 and optimize inner rotation/translation.
            # This is equivalent to optimizing relative position.
            side = compute_outer_side(new_centers, angles, 0.0, 0.0, otheta)
            
            # Check validity penalty
            penalty = 0.0
            min_dist = math.sqrt(3)
            for i in range(12):
                for j in range(i + 1, 12):
                    dx = new_centers[i][0] - new_centers[j][0]
                    dy = new_centers[i][1] - new_centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            
            return side + 10.0 * penalty

        # Grid search for initial rotation
        best_params = None
        best_val = float('inf')
        
        for rot in np.linspace(0, 60, 13):
            for otheta in np.linspace(0, 30, 7):
                # Optimize translation
                def obj_trans(params):
                    tx, ty = params
                    return obj_rigid([rot, tx, ty, otheta])
                
                res = minimize(obj_trans, [0.0, 0.0], method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5})
                
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = [rot, res.x[0], res.x[1], otheta]
                    
                if time.time() > deadline - 20:
                    return best_params, best_val
                    
        if best_params is None:
            return None, float('inf')
            
        # Refine
        res = minimize(obj_rigid, best_params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6})
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            
        return best_params, best_val

    # 1. Generate base configurations
    configs = []
    
    # Config 1: Standard Hex Lattice
    c1, a1 = generate_hexagonal_cluster()
    configs.append((c1, a1))
    
    # Config 2: Perturbed Angles
    c2, a2 = generate_compact_cluster()
    configs.append((c2, a2))
    
    # Config 3: Random Start
    random.seed(42)
    c3 = [[random.uniform(-3, 3), random.uniform(-3, 3)] for _ in range(12)]
    a3 = [random.uniform(0, 60) for _ in range(12)]
    configs.append((c3, a3))

    best_global_result = None
    best_global_val = float('inf')

    for centers, angles in configs:
        if time.time() > deadline - 30:
            break
            
        # Rigid Body Optimization
        params, val = rigid_body_optimize(centers, angles)
        
        if params is not None and val < best_global_val:
            # Reconstruct full parameter vector for final polish
            rot_deg, tx, ty, otheta = params
            rot_rad = math.radians(rot_deg)
            c = math.cos(rot_rad)
            s = math.sin(rot_rad)
            
            new_centers = []
            for cx, cy in centers:
                nx = cx * c - cy * s + tx
                ny = cx * s + cy * c + ty
                new_centers.append([nx, ny])
            
            # Construct full param vector: centers (24), angles (12), ocx, ocy, otheta
            full_params = []
            for cx, cy in new_centers:
                full_params.append(cx)
                full_params.append(cy)
            for a in angles:
                full_params.append(a)
            full_params.extend([0.0, 0.0, otheta]) # Outer center 0,0
            
            best_global_val = val
            best_global_result = full_params

    if best_global_result is None:
        # Fallback to simple lattice
        c1, a1 = generate_hexagonal_cluster()
        best_global_result = []
        for cx, cy in c1:
            best_global_result.append(cx)
            best_global_result.append(cy)
        for a in a1:
            best_global_result.append(a)
        best_global_result.extend([0.0, 0.0, 0.0])
        best_global_val = objective_full(best_global_result)

    # 2. Full Local Optimization (Polish)
    # Allow individual angles and positions to adjust slightly
    if time.time() < deadline - 10:
        res = minimize(objective_full, best_global_result, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7})
        
        if res.fun < best_global_val:
            best_global_val = res.fun
            best_global_result = res.x

    # Extract best parameters
    centers = []
    for i in range(12):
        centers.append([best_global_result[i * 2], best_global_result[i * 2 + 1]])
    angles = [best_global_result[24 + i] for i in range(12)]
    ocx, ocy = best_global_result[36], best_global_result[37]
    otheta = best_global_result[38]
    
    # Final Polish: Grid search on outer parameters only (fast)
    best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
    best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
    
    for da in np.linspace(-10, 10, 21):
        for dx in np.linspace(-0.5, 0.5, 11):
            for dy in np.linspace(-0.5, 0.5, 11):
                s = compute_outer_side(centers, angles, ocx + dx, ocy + dy, otheta + da)
                if s < best_side:
                    best_side = s
                    best_ocx, best_ocy, best_otheta = ocx + dx, ocy + dy, otheta + da
                    
    # Verify final validity
    if not is_valid(centers, angles):
        # Fallback to initial lattice if optimization resulted in invalid configuration
        c1, a1 = generate_hexagonal_cluster()
        centers = [[c[0], c[1]] for c in c1]
        angles = a1
        ocx, ocy, otheta = 0.0, 0.0, 0.0
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
