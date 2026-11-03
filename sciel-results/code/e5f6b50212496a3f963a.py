import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    deadline = time.time() + 140.0
    random.seed(42)

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def project_range(vertices, nx, ny):
        projs = np.dot(vertices, [nx, ny])
        return np.min(projs), np.max(projs)

    def compute_outer_side_smooth(centers, angles, ocx, ocy, otheta, beta=50.0):
        """Soft-max version of outer side computation."""
        otheta_rad = math.radians(otheta)
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            # Collect all projections for this normal
            projs = []
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    projs.append(proj)
            projs = np.array(projs)
            # Soft max
            m = np.max(projs)
            soft_max = m + np.log(np.sum(np.exp((projs - m) * beta)) / len(projs)) / beta
            if soft_max > max_proj:
                max_proj = soft_max
        return 2.0 * max_proj / math.sqrt(3)

    def compute_outer_side_exact(centers, angles, ocx, ocy, otheta):
        """Exact (nonsmooth) outer side computation."""
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

    def compute_separation_penalty(centers, angles, beta=30.0):
        """Smooth penalty for overlapping hexagons using soft-min over SAT axes."""
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        
        total_penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                vi, vj = verts[i], verts[j]
                # Collect separations on all 12 SAT axes
                seps = []
                for k in range(6):
                    p1 = vi[k]
                    p2 = vi[(k + 1) % 6]
                    edge = p2 - p1
                    nx, ny = -edge[1], edge[0]
                    min1, max1 = project_range(vi, nx, ny)
                    min2, max2 = project_range(vj, nx, ny)
                    sep = min(max1 - min2, max2 - min1)
                    seps.append(sep)
                for k in range(6):
                    p1 = vj[k]
                    p2 = vj[(k + 1) % 6]
                    edge = p2 - p1
                    nx, ny = -edge[1], edge[0]
                    min1, max1 = project_range(vi, nx, ny)
                    min2, max2 = project_range(vj, nx, ny)
                    sep = min(max1 - min2, max2 - min1)
                    seps.append(sep)
                
                seps = np.array(seps)
                # Soft min: if any sep is negative (overlap), penalize
                m = np.min(seps)
                # Soft min approximation
                soft_min = m - np.log(np.sum(np.exp((m - seps) * beta)) / len(seps)) / beta
                if soft_min < 0:
                    total_penalty += soft_min ** 2
        return total_penalty

    def is_valid(centers, angles, eps=1e-6):
        """Check if configuration is valid (no overlaps)."""
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                vi, vj = verts[i], verts[j]
                min_sep = float('inf')
                for k in range(6):
                    p1 = vi[k]
                    p2 = vi[(k + 1) % 6]
                    edge = p2 - p1
                    nx, ny = -edge[1], edge[0]
                    min1, max1 = project_range(vi, nx, ny)
                    min2, max2 = project_range(vj, nx, ny)
                    sep = min(max1 - min2, max2 - min1)
                    if sep < min_sep:
                        min_sep = sep
                for k in range(6):
                    p1 = vj[k]
                    p2 = vj[(k + 1) % 6]
                    edge = p2 - p1
                    nx, ny = -edge[1], edge[0]
                    min1, max1 = project_range(vi, nx, ny)
                    min2, max2 = project_range(vj, nx, ny)
                    sep = min(max1 - min2, max2 - min1)
                    if sep < min_sep:
                        min_sep = sep
                if min_sep < -eps:
                    return False
        return True

    def generate_hexagonal_cluster():
        d = math.sqrt(3) + 0.005
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

    def generate_packed_cluster():
        """Tighter initial packing with smaller spacing."""
        d = math.sqrt(3) + 0.001
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

    def params_to_state(params):
        centers = [[params[i * 2], params[i * 2 + 1]] for i in range(12)]
        angles = [params[24 + i] for i in range(12)]
        otheta = params[36]
        return centers, angles, otheta

    def state_to_params(centers, angles, otheta):
        params = []
        for cx, cy in centers:
            params.append(cx)
            params.append(cy)
        for a in angles:
            params.append(a)
        params.append(otheta)
        return params

    def objective_smooth(params, beta_outer=50.0, beta_sep=30.0, penalty_weight=100.0):
        if time.time() > deadline - 5:
            return 1e9
        centers, angles, otheta = params_to_state(params)
        side = compute_outer_side_smooth(centers, angles, 0.0, 0.0, otheta, beta_outer)
        penalty = compute_separation_penalty(centers, angles, beta_sep)
        return side + penalty_weight * penalty

    def objective_exact(params):
        if time.time() > deadline - 5:
            return 1e9
        centers, angles, otheta = params_to_state(params)
        side = compute_outer_side_exact(centers, angles, 0.0, 0.0, otheta)
        penalty = compute_separation_penalty(centers, angles, 30.0)
        return side + 100.0 * penalty

    # Generate initial configurations
    configs = []
    
    # Config 1: Standard hex lattice with small gap
    c1, a1 = generate_hexagonal_cluster()
    params1 = state_to_params(c1, a1, 0.0)
    configs.append(params1)
    
    # Config 2: Tighter packing
    c2, a2 = generate_packed_cluster()
    params2 = state_to_params(c2, a2, 0.0)
    configs.append(params2)
    
    # Config 3: Perturbed angles
    c3, a3 = c1, [30.0 + random.uniform(-5, 5) for _ in range(12)]
    params3 = state_to_params(c3, a3, 15.0)
    configs.append(params3)
    
    # Config 4: Different outer angle
    c4, a4 = c1, [30.0] * 12
    params4 = state_to_params(c4, a4, 30.0)
    configs.append(params4)

    best_params = None
    best_val = float('inf')

    for init_idx, init_params in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        # Stage 1: Coarse SLSQP with smooth objective
        def obj_coarse(params):
            return objective_smooth(params, beta_outer=30.0, beta_sep=20.0, penalty_weight=50.0)
        
        try:
            res1 = minimize(obj_coarse, init_params, method='SLSQP',
                          options={'maxiter': 200, 'ftol': 1e-6, 'disp': False})
            p1 = res1.x
            v1 = objective_exact(p1)
        except Exception:
            p1 = init_params
            v1 = objective_exact(init_params)
        
        if v1 < best_val:
            best_val = v1
            best_params = p1.copy()
        
        # Stage 2: Fine SLSQP
        if time.time() > deadline - 20:
            continue
            
        def obj_fine(params):
            return objective_smooth(params, beta_outer=80.0, beta_sep=50.0, penalty_weight=200.0)
        
        try:
            res2 = minimize(obj_fine, p1, method='SLSQP',
                          options={'maxiter': 300, 'ftol': 1e-8, 'disp': False})
            p2 = res2.x
            v2 = objective_exact(p2)
        except Exception:
            p2 = p1
            v2 = v1
        
        if v2 < best_val:
            best_val = v2
            best_params = p2.copy()
        
        # Stage 3: Nelder-Mead polish
        if time.time() > deadline - 10:
            continue
            
        def obj_nm(params):
            return objective_smooth(params, beta_outer=100.0, beta_sep=60.0, penalty_weight=500.0)
        
        try:
            res3 = minimize(obj_nm, p2, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9})
            p3 = res3.x
            v3 = objective_exact(p3)
        except Exception:
            p3 = p2
            v3 = v2
        
        if v3 < best_val:
            best_val = v3
            best_params = p3.copy()

    if best_params is None:
        best_params = configs[0]
        best_val = objective_exact(best_params)

    # Final: grid search on outer angle with fine resolution
    centers = [[best_params[i*2], best_params[i*2+1]] for i in range(12)]
    angles = [best_params[24+i] for i in range(12)]
    best_side = compute_outer_side_exact(centers, angles, 0.0, 0.0, best_params[36])
    best_otheta = best_params[36]
    
    # Coarse grid
    for da in np.linspace(-30, 30, 61):
        s = compute_outer_side_exact(centers, angles, 0.0, 0.0, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da
    
    # Fine grid around best
    for da in np.linspace(-2, 2, 41):
        s = compute_outer_side_exact(centers, angles, 0.0, 0.0, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da
    
    # Even finer
    for da in np.linspace(-0.5, 0.5, 21):
        s = compute_outer_side_exact(centers, angles, 0.0, 0.0, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    # Verify validity
    if not is_valid(centers, angles):
        # Try to fix by using a slightly larger outer hexagon
        # or fall back to a valid configuration
        for da in np.linspace(-45, 45, 91):
            s = compute_outer_side_exact(centers, angles, 0.0, 0.0, da)
            if is_valid(centers, angles):
                best_side = s
                best_otheta = da
                break

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
