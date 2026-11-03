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

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        """
        Computes the side length of the outer hexagon required to contain
        the given inner hexagons.
        """
        max_proj = 0.0
        otheta_rad = math.radians(otheta_deg)
        # Normals to the sides of the outer hexagon
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            # Project all vertices of all inner hexagons
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
        # Side length = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    def is_valid_sat(centers, angles):
        """
        Exact validity check using Separating Axis Theorem.
        """
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
            
        def project(vertices, nx, ny):
            projs = np.dot(vertices, [nx, ny])
            return np.min(projs), np.max(projs)

        def are_disjoint(v1, v2, eps=1e-9):
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

        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint(verts[i], verts[j]):
                    return False
        return True

    def polish_outer(centers, angles, initial_ocx, initial_ocy, initial_otheta):
        """
        High-precision polish of outer hexagon parameters.
        """
        best_side = compute_outer_side(centers, angles, initial_ocx, initial_ocy, initial_otheta)
        best_ocx, best_ocy, best_otheta = initial_ocx, initial_ocy, initial_otheta
        
        # Multi-scale grid search
        scales = [1.0, 0.1, 0.01, 0.001, 0.0001]
        for scale in scales:
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Perturbations
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da_deg in [-scale*10, 0, scale*10]:
                            new_otheta = best_otheta + da_deg
                            side = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, new_otheta)
                            if side < best_side - 1e-12:
                                best_side = side
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = new_otheta
                                improved = True
        return best_ocx, best_ocy, best_otheta, best_side

    def optimize_slsqp(x0):
        """
        SLSQP optimization with hard constraints for non-overlap.
        Variables: x1, y1, a1, ..., x12, y12, a12, ocx, ocy, otheta
        """
        n = 12
        def objective(x):
            if time.time() > deadline - 30:
                return 1e10
            centers = [[x[i*3], x[i*3+1]] for i in range(n)]
            angles = [x[i*3+2] for i in range(n)]
            ocx, ocy, otheta = x[3*n], x[3*n+1], x[3*n+2]
            return compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        def constraint_fun(x, i, j):
            xi = x[i*3]
            yi = x[i*3+1]
            xj = x[j*3]
            yj = x[j*3+1]
            return (xi - xj)**2 + (yi - yj)**2 - 3.0 # dist^2 - 3 >= 0
        
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: constraint_fun(x, i, j)})
        
        bounds = []
        for i in range(n):
            bounds.extend([(-5, 5), (-5, 5), (0, 60)])
        bounds.extend([(-2, 2), (-2, 2), (0, 60)])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 100, 'ftol': 1e-10})
            if res.success:
                return res.x
        except Exception:
            pass
        return x0

    # Generate starts
    starts = []
    
    # 1. Hexagonal Lattice
    d = math.sqrt(3) + 0.01
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
    
    centers_init = [np.array([0.0, 0.0])]
    ring1_vectors = [
        u, v, v - u, -u, -v, u - v
    ]
    for vec in ring1_vectors:
        centers_init.append(vec.copy())
        
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
        centers_init.append(ring2_candidates[i].copy())
        
    x0 = []
    for c in centers_init:
        x0.append(c[0])
        x0.append(c[1])
    for i in range(12):
        x0.append(30.0)
    x0.extend([0.0, 0.0, 0.0])
    starts.append(x0)
    
    # 2. Random Starts
    random.seed(42)
    for _ in range(3):
        centers_rand = []
        for _ in range(12):
            centers_rand.append([random.uniform(-3, 3), random.uniform(-3, 3)])
        x_rand = []
        for c in centers_rand:
            x_rand.append(c[0])
            x_rand.append(c[1])
        for i in range(12):
            x_rand.append(random.uniform(0, 60))
        x_rand.extend([0.0, 0.0, 0.0])
        starts.append(x_rand)

    best_result = None
    best_val = float('inf')

    for x0 in starts:
        if time.time() > deadline - 20:
            break
        
        # Run SLSQP
        res_x = optimize_slsqp(x0)
        
        # Extract parameters
        centers = [[res_x[i*3], res_x[i*3+1]] for i in range(12)]
        angles = [res_x[i*3+2] for i in range(12)]
        ocx, ocy = res_x[36], res_x[37]
        otheta = res_x[38]
        
        # Polish outer hexagon
        best_ocx, best_ocy, best_otheta, best_side = polish_outer(
            centers, angles, ocx, ocy, otheta
        )
        
        # Validate
        if is_valid_sat(centers, angles):
            if best_side < best_val:
                best_val = best_side
                best_result = (centers, angles, best_ocx, best_ocy, best_otheta, best_side)

    if best_result is None:
        # Fallback to initial lattice
        centers = [[c[0], c[1]] for c in centers_init]
        angles = [30.0] * 12
        ocx, ocy, otheta = 0.0, 0.0, 0.0
        best_ocx, best_ocy, best_otheta, best_side = polish_outer(
            centers, angles, ocx, ocy, otheta
        )
        best_result = (centers, angles, best_ocx, best_ocy, best_otheta, best_side)

    centers, angles, ocx, ocy, otheta, side = best_result
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (ocx, ocy)
    outer_side = side
    outer_angle = otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
