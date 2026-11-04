import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
SQRT3 = math.sqrt(3)
MIN_DIST = SQRT3 - 1e-6

def construct():
    deadline = time.time() + 145.0

    def hex_verts(cx, cy, angle):
        thetas = angle + np.arange(6) * (np.pi / 3)
        return np.column_stack([cx + np.cos(thetas), cy + np.sin(thetas)])

    def sat_disjoint(verts1, verts2, eps=1e-7):
        edges1 = np.roll(verts1, -1, axis=0) - verts1
        edges2 = np.roll(verts2, -1, axis=0) - verts2
        norms1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
        norms2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])
        norms = np.vstack([norms1, norms2])
        norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-15)
        proj1 = verts1 @ norms.T
        proj2 = verts2 @ norms.T
        min1 = proj1.min(axis=0)
        max1 = proj1.max(axis=0)
        min2 = proj2.min(axis=0)
        max2 = proj2.max(axis=0)
        return bool(np.any(max1 < min2 - eps) or np.any(max2 < min1 - eps))

    def all_disjoint(centers, angles, eps=1e-7):
        verts_list = [hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(verts_list[i], verts_list[j], eps):
                    return False
        return True

    def outer_side(centers, angles, ocx, ocy, otheta_deg):
        otheta = math.radians(otheta_deg)
        max_proj = 0.0
        for k in range(6):
            na = otheta + math.pi / 6 + k * np.pi / 3
            nx, ny = math.cos(na), math.sin(na)
            for i in range(N):
                verts = hex_verts(centers[i][0], centers[i][1], angles[i])
                proj = (verts[:, 0] - ocx) * nx + (verts[:, 1] - ocy) * ny
                if proj.max() > max_proj:
                    max_proj = proj.max()
        return 2.0 * max_proj / SQRT3

    def unpack(x):
        centers = np.array([[x[3 * i], x[3 * i + 1]] for i in range(N)])
        angles = np.array([x[3 * i + 2] for i in range(N)])
        ocx, ocy, otheta = x[36], x[37], x[38]
        return centers, angles, ocx, ocy, otheta

    def pack(x):
        centers, angles, ocx, ocy, otheta = unpack(x)
        return centers, angles, ocx, ocy, otheta

    def generate_ring_config():
        """Standard 1+6+5 ring configuration."""
        d = SQRT3 + 0.001
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * SQRT3 / 2.0])
        ring1 = [u, v, v - u, -u, -v, u - v]
        
        # Ring 2: pick 5 from ring1 + basis
        basis = [u, v, v - u, -u, -v, u - v]
        ring2_candidates = []
        for i in range(6):
            p1 = ring1[i]
            for b in basis:
                p2 = p1 + b
                if np.linalg.norm(p2) < 0.01:
                    continue
                if any(np.linalg.norm(p2 - r1) < 0.01 for r1 in ring1):
                    continue
                if any(np.linalg.norm(p2 - rc) < 0.01 for rc in ring2_candidates):
                    continue
                ring2_candidates.append(p2)
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        ring2 = ring2_candidates[:5]
        
        centers = [[0.0, 0.0]] + [list(r) for r in ring1] + [list(r) for r in ring2]
        angles = [30.0] * N
        return centers, angles

    def generate_lattice_config():
        """Hexagonal lattice configuration."""
        d = SQRT3 + 0.001
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * SQRT3 / 2.0])
        
        # Positions in hex lattice
        positions = [
            np.array([0, 0]),
            u, v, v - u, -u, -v, u - v,
            2*u, u+v, 2*v, v-u+u, -u+u-v, 2*(v-u)
        ]
        # Remove duplicates and center
        seen = []
        for p in positions:
            if not any(np.linalg.norm(p - s) < 0.01 for s in seen):
                seen.append(p)
        
        centers = [list(p) for p in seen[:N]]
        angles = [30.0] * N
        return centers, angles

    def generate_chain_config():
        """Linear chain configuration."""
        d = SQRT3 + 0.001
        centers = []
        for i in range(N):
            centers.append([i * d, 0.0])
        angles = [30.0] * N
        return centers, angles

    def slsqp_optimize(centers, angles, max_iters=200):
        n = N
        x0 = []
        for i in range(n):
            x0.extend([centers[i][0], centers[i][1], angles[i]])
        x0.extend([0.0, 0.0, 0.0])
        
        def objective(x):
            if time.time() > deadline - 10.0:
                return 1e10
            c = [[x[3 * i], x[3 * i + 1]] for i in range(n)]
            a = [x[3 * i + 2] for i in range(n)]
            ocx, ocy, otheta = x[3 * n], x[3 * n + 1], x[3 * n + 2]
            side = outer_side(np.array(c), np.array(a), ocx, ocy, otheta)
            return side
        
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                def dist_constraint(x, i=i, j=j):
                    xi = x[3 * i]
                    yi = x[3 * i + 1]
                    xj = x[3 * j]
                    yj = x[3 * j + 1]
                    return (xi - xj) ** 2 + (yi - yj) ** 2 - MIN_DIST ** 2
                constraints.append({'type': 'ineq', 'fun': dist_constraint})
        
        bounds = []
        for i in range(n):
            bounds.extend([(-5, 5), (-5, 5), (-90, 90)])
        bounds.extend([(-3, 3), (-3, 3), (-30, 30)])
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                             constraints=constraints,
                             options={'maxiter': max_iters, 'ftol': 1e-10, 'disp': False})
            if result.fun < 1e8:
                c = [[result.x[3 * i], result.x[3 * i + 1]] for i in range(n)]
                a = [result.x[3 * i + 2] for i in range(n)]
                ocx, ocy, otheta = result.x[3 * n], result.x[3 * n + 1], result.x[3 * n + 2]
                if all_disjoint(np.array(c), np.array(a)):
                    side = outer_side(np.array(c), np.array(a), ocx, ocy, otheta)
                    return c, a, ocx, ocy, otheta, side
        except Exception:
            pass
        return None

    def nm_polish(x_start, max_iter=500):
        def nm_objective(x):
            centers, angles, ocx, ocy, otheta = unpack(x)
            if not all_disjoint(centers, angles):
                return 1e8
            return outer_side(centers, angles, ocx, ocy, otheta)
        try:
            res = minimize(nm_objective, x_start, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < 1e7:
                return res.x, res.fun
        except Exception:
            pass
        return x_start, nm_objective(x_start)

    def polish_outer(centers, angles, ocx, ocy, otheta):
        best_side = outer_side(centers, angles, ocx, ocy, otheta)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
        scales = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        for scale in scales:
            if time.time() > deadline - 5.0:
                break
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da in [-scale * 5, 0, scale * 5]:
                            s = outer_side(centers, angles, best_ocx + dx, best_ocy + dy, best_otheta + da)
                            if s < best_side - 1e-14:
                                best_side = s
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta += da
                                improved = True
        return best_ocx, best_ocy, best_otheta, best_side

    # Generate initial configurations
    configs = []
    centers0, angles0 = generate_ring_config()
    configs.append((centers0, angles0))
    
    centers1, angles1 = generate_lattice_config()
    configs.append((centers1, angles1))
    
    centers2, angles2 = generate_chain_config()
    configs.append((centers2, angles2))
    
    # Random perturbations of ring config
    random.seed(42)
    for trial in range(3):
        c = [list(p) for p in centers0]
        a = list(angles0)
        for i in range(N):
            c[i][0] += random.uniform(-0.2, 0.2)
            c[i][1] += random.uniform(-0.2, 0.2)
            a[i] = random.uniform(-30, 30)
        # Check validity
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                dx = c[i][0] - c[j][0]
                dy = c[i][1] - c[j][1]
                if math.sqrt(dx * dx + dy * dy) < MIN_DIST:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            configs.append((c, a))

    best_x = None
    best_side = float('inf')

    # Try SLSQP from each initial config
    for centers, angles in configs:
        if time.time() > deadline - 40.0:
            break
        result = slsqp_optimize(centers, angles, max_iters=150)
        if result is not None:
            c, a, ocx, ocy, otheta, side = result
            if side < best_side:
                best_side = side
                x = []
                for i in range(N):
                    x.extend([c[i][0], c[i][1], a[i]])
                x.extend([ocx, ocy, otheta])
                best_x = np.array(x)

    if best_x is None:
        centers, angles = configs[0]
        x0 = []
        for i in range(N):
            x0.extend([centers[i][0], centers[i][1], angles[i]])
        x0.extend([0.0, 0.0, 0.0])
        best_x = np.array(x0)
        best_side = outer_side(np.array(centers), np.array(angles), 0.0, 0.0, 0.0)

    # Nelder-Mead polish
    if time.time() < deadline - 25.0:
        best_x, best_side = nm_polish(best_x, max_iter=500)

    # Additional Nelder-Mead restarts
    if time.time() < deadline - 30.0:
        for trial in range(4):
            if time.time() > deadline - 30.0:
                break
            x_trial = list(best_x)
            for i in range(N):
                x_trial[3 * i + 2] = random.uniform(-45, 45)
                x_trial[3 * i] += random.uniform(-0.15, 0.15)
                x_trial[3 * i + 1] += random.uniform(-0.15, 0.15)
            x_trial = np.array(x_trial)
            x_polished, side_polished = nm_polish(x_trial, max_iter=400)
            if side_polished < best_side:
                best_side = side_polished
                best_x = x_polished

    # Final high-precision outer hexagon polish
    centers, angles, ocx, ocy, otheta = unpack(best_x)
    if time.time() < deadline - 5.0:
        ocx, ocy, otheta, best_side = polish_outer(centers, angles, ocx, ocy, otheta)

    # Build return value
    inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
    outer_center = (ocx, ocy)
    outer_side = best_side
    outer_angle = otheta

    return inner, outer_center, outer_side, outer_angle
