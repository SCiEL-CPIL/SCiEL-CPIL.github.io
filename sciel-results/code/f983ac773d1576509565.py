import numpy as np
import time
import math
import random

N = 12
SQRT3 = math.sqrt(3.0)

def _hex_verts(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    a = math.radians(angle_deg)
    xs, ys = [], []
    for k in range(6):
        ang = a + k * math.pi / 3.0
        xs.append(cx + side * math.cos(ang))
        ys.append(cy + side * math.sin(ang))
    return np.array(list(zip(xs, ys)))

def _outer_side_from_points(pts):
    """Given an array of (x,y) points, return the minimum side length of a regular
    hexagon (centered at origin, angle 0) that contains all points."""
    x = pts[:, 0]
    y = pts[:, 1]
    s3 = SQRT3
    L = np.stack([
        x + y / s3,
        2.0 * y / s3,
        -x + y / s3,
        -x - y / s3,
        -2.0 * y / s3,
        x - y / s3,
    ], axis=1)
    return np.max(L)

def _pairwise_overlap_penalty(verts_list, positions):
    """
    Fast approximate overlap penalty using center distances.
    For two unit hexagons with angle 0, the minimum center distance to avoid overlap
    depends on relative orientation. We use a conservative bound:
    - If both have same angle: min dist = SQRT3 (flat-to-flat)
    - General: use bounding-circle radius = 1.0 (circumradius), so if dist < 2.0,
      they *might* overlap. We use a smooth penalty that kicks in below 2.0.
    """
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 2.0:
                # Smooth quadratic penalty
                penalty += 0.5 * (2.0 - dist) ** 2
    return penalty

def _exact_overlap_check(positions, angles, tol=1e-7):
    """
    Exact SAT-based check for pairwise disjointness of all hexagons.
    Returns True if all pairs are disjoint.
    """
    verts_list = []
    for i in range(N):
        verts_list.append(_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i]))
    
    for i in range(N):
        for j in range(i + 1, N):
            p1 = verts_list[i]
            p2 = verts_list[j]
            disjoint = False
            # Check all edges of p1 against p2
            for k in range(6):
                k2 = (k + 1) % 6
                edge = p1[k2] - p1[k]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                proj1 = p1 @ normal
                proj2 = p2 @ normal
                if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                    disjoint = True
                    break
            if not disjoint:
                for k in range(6):
                    k2 = (k + 1) % 6
                    edge = p2[k2] - p2[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-12:
                        continue
                    normal /= norm
                    proj1 = p1 @ normal
                    proj2 = p2 @ normal
                    if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                        disjoint = True
                        break
            if not disjoint:
                return False
    return True

def _all_vertices(positions, angles):
    """Get all 72 vertices."""
    verts = np.zeros((N * 6, 2))
    for i in range(N):
        v = _hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i])
        verts[i*6:(i+1)*6] = v
    return verts

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    
    def elapsed():
        return time.time() - start_time
    
    # --- Initial configurations ---
    def make_hex_lattice():
        """Standard hexagonal lattice arrangement."""
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        pos = np.zeros((N, 2))
        for i, (q, r) in enumerate(coords):
            pos[i, 0] = SQRT3 * (q + r / 2.0)
            pos[i, 1] = 1.5 * r
        ang = np.zeros(N)
        return pos, ang
    
    def make_circular():
        pos = np.zeros((N, 2))
        for i in range(N):
            a = 2 * math.pi * i / N
            pos[i, 0] = 2.5 * math.cos(a)
            pos[i, 1] = 2.5 * math.sin(a)
        ang = np.zeros(N)
        return pos, ang
    
    def make_flower():
        """1 center + 6 ring + 5 outer."""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # Center
        pos[0] = [0, 0]
        # Ring of 6
        for k in range(6):
            a = k * math.pi / 3.0
            pos[k+1, 0] = SQRT3 * math.cos(a)
            pos[k+1, 1] = SQRT3 * math.sin(a)
        # 5 more spread out
        for k in range(5):
            a = 2 * math.pi * k / 5.0 + math.pi / 5.0
            pos[k+7, 0] = 2.5 * SQRT3 * math.cos(a)
            pos[k+7, 1] = 2.5 * SQRT3 * math.sin(a)
        return pos, ang
    
    def make_random(seed):
        rng = random.Random(seed)
        pos = np.array([[rng.uniform(-6, 6), rng.uniform(-6, 6)] for _ in range(N)])
        ang = np.array([rng.choice([0, 30, 60, 90, 120, 150]) for _ in range(N)])
        return pos, ang
    
    inits = [
        make_hex_lattice(),
        make_circular(),
        make_flower(),
        make_random(42),
        make_random(123),
        make_random(777),
    ]
    
    # --- Simulated Annealing ---
    def sa_optimize(pos0, ang0, n_steps=3000, T0=2.0, T_end=0.01):
        """Run SA to minimize outer side + overlap penalty."""
        pos = pos0.copy()
        ang = ang0.copy()
        
        # Compute initial objective
        verts = _all_vertices(pos, ang)
        side = _outer_side_from_points(verts)
        pen = _pairwise_overlap_penalty(verts, pos)
        obj = side + 50.0 * pen
        
        best_obj = obj
        best_pos = pos.copy()
        best_ang = ang.copy()
        
        T = T0
        cooling = (T_end / T0) ** (1.0 / max(n_steps, 1))
        
        step_size = 0.3
        
        for step in range(n_steps):
            if elapsed() > deadline:
                break
            
            # Random perturbation
            idx = random.randint(0, N - 1)
            if random.random() < 0.8:
                # Move position
                dx = random.gauss(0, step_size)
                dy = random.gauss(0, step_size)
                old_px, old_py = pos[idx, 0], pos[idx, 1]
                pos[idx, 0] += dx
                pos[idx, 1] += dy
            else:
                # Rotate
                da = random.gauss(0, 10.0)
                old_a = ang[idx]
                ang[idx] = (old_a + da) % 360.0
            
            verts_new = _all_vertices(pos, ang)
            side_new = _outer_side_from_points(verts_new)
            pen_new = _pairwise_overlap_penalty(verts_new, pos)
            obj_new = side_new + 50.0 * pen_new
            
            delta = obj_new - obj
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                obj = obj_new
                if obj < best_obj:
                    best_obj = obj
                    best_pos = pos.copy()
                    best_ang = ang.copy()
            else:
                # Revert
                if random.random() < 0.8:
                    pos[idx, 0] = old_px
                    pos[idx, 1] = old_py
                else:
                    ang[idx] = old_a
            
            T *= cooling
            # Anneal step size
            step_size = max(0.05, 0.3 * (0.999 ** step))
        
        return best_pos, best_ang, best_obj
    
    # --- Multi-start SA ---
    for init_idx, (pos0, ang0) in enumerate(inits):
        if elapsed() > deadline:
            break
        
        # Ensure initial validity; if not, scale up
        if not _exact_overlap_check(pos0, ang0):
            for scale in [1.3, 1.6, 2.0, 2.5, 3.0]:
                if _exact_overlap_check(pos0 * scale, ang0):
                    pos0 = pos0 * scale
                    break
            else:
                # Spread linearly
                pos0 = np.array([[i * 3.0, 0.0] for i in range(N)])
        
        # Run SA with different parameters
        for trial in range(2):
            if elapsed() > deadline:
                break
            n_steps = 2500
            T0 = 3.0 if trial == 0 else 1.5
            pos_res, ang_res, obj_res = sa_optimize(pos0, ang0, n_steps, T0)
            
            # Validate
            if _exact_overlap_check(pos_res, ang_res):
                verts = _all_vertices(pos_res, ang_res)
                side = _outer_side_from_points(verts)
                if side < best_side:
                    best_side = side
                    best_positions = pos_res
                    best_angles = ang_res
    
    # --- Local polish with scipy ---
    if best_positions is not None and elapsed() < deadline - 10:
        try:
            from scipy.optimize import minimize
            
            def objective(params):
                p = params[:2*N].reshape(N, 2)
                a = params[2*N:]
                verts = _all_vertices(p, a)
                side = _outer_side_from_points(verts)
                pen = _pairwise_overlap_penalty(verts, p)
                return side + 50.0 * pen
            
            x0 = np.concatenate([best_positions.flatten(), best_angles])
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_side + 0.01:
                p_res = res.x[:2*N].reshape(N, 2)
                a_res = res.x[2*N:]
                if _exact_overlap_check(p_res, a_res):
                    verts = _all_vertices(p_res, a_res)
                    side = _outer_side_from_points(verts)
                    if side < best_side:
                        best_side = side
                        best_positions = p_res
                        best_angles = a_res
        except Exception:
            pass
    
    # --- Fallback ---
    if best_positions is None:
        pos0, ang0 = make_hex_lattice()
        for scale in [1.0, 1.2, 1.5, 2.0]:
            if _exact_overlap_check(pos0 * scale, ang0):
                best_positions = pos0 * scale
                best_angles = ang0
                best_side = _outer_side_from_points(_all_vertices(best_positions, best_angles))
                break
        else:
            best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_angles = np.zeros(N)
            best_side = _outer_side_from_points(_all_vertices(best_positions, best_angles))
    
    # Final validation
    if not _exact_overlap_check(best_positions, best_angles):
        # Last resort: spread out
        best_positions = np.array([[i * 3.5, 0.0] for i in range(N)])
        best_angles = np.zeros(N)
        best_side = _outer_side_from_points(_all_vertices(best_positions, best_angles))
    
    # Compute final exact side
    verts_final = _all_vertices(best_positions, best_angles)
    final_side = float(_outer_side_from_points(verts_final))
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    return inner, (0.0, 0.0), final_side, 0.0
