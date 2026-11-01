import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    N = 12

    def hexagon_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def calc_min_outer_side_fast(all_verts, theta):
        angles = theta + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        widths = dots.max(axis=0) - dots.min(axis=0)
        max_width = np.max(widths)
        apothem = max_width / 2.0
        side = 2 * apothem / np.sqrt(3)
        return side

    def find_best_outer_orientation_fast(all_verts):
        thetas = np.linspace(0, np.pi / 3, 72)
        best_side = np.inf
        best_theta = 0
        for th in thetas:
            s = calc_min_outer_side_fast(all_verts, th)
            if s < best_side:
                best_side = s
                best_theta = th
        if best_side < np.inf:
            try:
                res = minimize_scalar(lambda th: calc_min_outer_side_fast(all_verts, th),
                                      bounds=(max(0, best_theta - np.pi / 90), min(np.pi / 3, best_theta + np.pi / 90)),
                                      method='bounded', options={'xatol': 1e-6})
                if res.fun < best_side:
                    best_side = res.fun
                    best_theta = res.x
            except Exception:
                pass
        return best_side, best_theta

    def get_overlap_penalty(positions, angles, threshold=0.0):
        penalty = 0.0
        n = len(positions)
        for i in range(n):
            vi = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i + 1, n):
                vj = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                axes = []
                for v in [vi, vj]:
                    for k in range(len(v)):
                        m = (k + 1) % len(v)
                        edge = v[m] - v[k]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-12:
                            normal /= norm
                            axes.append(normal)
                min_overlap = float('inf')
                for axis in axes:
                    proj1 = vi @ axis
                    proj2 = vj @ axis
                    min1, max1 = proj1.min(), proj1.max()
                    min2, max2 = proj2.min(), proj2.max()
                    if max1 < min2 or max2 < min1:
                        min_overlap = 0.0
                        break
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < min_overlap:
                        min_overlap = overlap
                if min_overlap > 0:
                    penalty += min_overlap * min_overlap
        return penalty

    def hexagons_disjoint(positions, angles, tol=1e-7):
        n = len(positions)
        for i in range(n):
            vi = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i + 1, n):
                vj = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                axes = []
                for v in [vi, vj]:
                    for k in range(len(v)):
                        m = (k + 1) % len(v)
                        edge = v[m] - v[k]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-12:
                            normal /= norm
                            axes.append(normal)
                separated = False
                for axis in axes:
                    proj1 = vi @ axis
                    proj2 = vj @ axis
                    min1, max1 = proj1.min(), proj1.max()
                    min2, max2 = proj2.min(), proj2.max()
                    if max1 < min2 - tol or max2 < min1 - tol:
                        separated = True
                        break
                if not separated:
                    return False
        return True

    def approx_outer_side(positions, angles):
        all_verts = []
        for i in range(len(positions)):
            v = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        min_side = float('inf')
        for theta in [0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3]:
            side = calc_min_outer_side_fast(all_verts, theta)
            if side < min_side:
                min_side = side
        return min_side

    def make_honeycomb_config(ring1_dist, ring2_dist, ring2_angles, center_angle=0.0):
        positions = [[0.0, 0.0]]
        angles = [center_angle]
        for k in range(6):
            ang = np.radians(k * 60)
            positions.append([ring1_dist * np.cos(ang), ring1_dist * np.sin(ang)])
            angles.append(0.0)
        for k in range(5):
            ang = np.radians(ring2_angles[k])
            positions.append([ring2_dist * np.cos(ang), ring2_dist * np.sin(ang)])
            angles.append(0.0)
        return np.array(positions), np.array(angles)

    def make_linear_config(n_per_row=4, row_spacing=2.0, col_spacing=2.0):
        positions = []
        angles = []
        for r in range(3):
            for c in range(n_per_row):
                x = c * col_spacing
                y = r * row_spacing
                positions.append([x, y])
                angles.append(0.0)
        return np.array(positions), np.array(angles)

    # Generate diverse initial configurations
    configs = []
    
    # Honeycomb 7+5
    d1 = np.sqrt(3)
    d2 = 2 * np.sqrt(3)
    p1, a1 = make_honeycomb_config(d1, d2, [30, 90, 150, 210, 270])
    configs.append((p1, a1))
    
    p2, a2 = make_honeycomb_config(d1, d2, [0, 60, 120, 180, 240])
    configs.append((p2, a2))
    
    # Tighter honeycomb
    p3, a3 = make_honeycomb_config(d1 * 0.95, d2 * 0.95, [30, 90, 150, 210, 270])
    configs.append((p3, a3))
    
    # Rotated honeycomb
    p4, a4 = make_honeycomb_config(d1, d2, [30, 90, 150, 210, 270], center_angle=np.pi/6)
    configs.append((p4, a4))
    
    # Linear grid 4x3
    p5, a5 = make_linear_config(4, 2.0, 2.0)
    configs.append((p5, a5))
    
    # Staggered grid
    positions = []
    angles = []
    for r in range(3):
        offset = (r % 2) * 1.0
        for c in range(4):
            positions.append([c * 2.0 + offset, r * 1.8])
            angles.append(0.0)
    configs.append((np.array(positions), np.array(angles)))
    
    # Ring of 12
    positions = []
    angles = []
    for i in range(12):
        ang = 2 * np.pi * i / 12
        positions.append([3.0 * np.cos(ang), 3.0 * np.sin(ang)])
        angles.append(np.pi / 2)  # point outward
    configs.append((np.array(positions), np.array(angles)))

    # Validate initial configs
    valid_configs = []
    for p, a in configs:
        if hexagons_disjoint(p, a):
            valid_configs.append((p, a))
    
    if not valid_configs:
        # Fallback: large circle
        p_fb = np.zeros((12, 2))
        a_fb = np.zeros(12)
        for i in range(12):
            ang = 2 * np.pi * i / 12
            p_fb[i] = [6.0 * np.cos(ang), 6.0 * np.sin(ang)]
        valid_configs.append((p_fb, a_fb))

    best_valid_side = 1e9
    best_valid_params = None

    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, np.pi / 3)] * 12

    def local_obj(params):
        if time.time() > deadline:
            return 1e10
        n = len(params) // 3
        positions = np.array([[params[i * 3], params[i * 3 + 1]] for i in range(n)])
        angles = np.array([params[i * 3 + 2] for i in range(n)])
        penalty = get_overlap_penalty(positions, angles)
        if penalty > 0.001:
            return 10.0 + 1000.0 * penalty
        return approx_outer_side(positions, angles)

    # Run local optimization from each valid config
    for idx, (p_init, a_init) in enumerate(valid_configs):
        if time.time() > deadline - 15:
            break
        init_params = np.concatenate([p_init.flatten(), a_init])
        
        try:
            res = minimize(local_obj, init_params, method='Powell',
                           options={'maxiter': 300, 'xtol': 1e-5, 'ftol': 1e-6})
            if res.fun < 10.0:
                pos = np.array([[res.x[i * 3], res.x[i * 3 + 2]] for i in range(12)])
                ang = np.array([res.x[i * 3 + 2] for i in range(12)])
                if hexagons_disjoint(pos, ang):
                    all_verts = []
                    for i in range(12):
                        v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                        all_verts.append(v)
                    all_verts = np.vstack(all_verts)
                    side, _ = find_best_outer_orientation_fast(all_verts)
                    if side < best_valid_side:
                        best_valid_side = side
                        best_valid_params = res.x.copy()
        except Exception as e:
            pass

    # If we have time, try Nelder-Mead from best so far
    if best_valid_params is not None and time.time() < deadline - 30:
        try:
            res = minimize(local_obj, best_valid_params, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6})
            if res.fun < 10.0:
                pos = np.array([[res.x[i * 3], res.x[i * 3 + 1]] for i in range(12)])
                ang = np.array([res.x[i * 3 + 2] for i in range(12)])
                if hexagons_disjoint(pos, ang):
                    all_verts = []
                    for i in range(12):
                        v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                        all_verts.append(v)
                    all_verts = np.vstack(all_verts)
                    side, _ = find_best_outer_orientation_fast(all_verts)
                    if side < best_valid_side:
                        best_valid_side = side
                        best_valid_params = res.x.copy()
        except Exception:
            pass

    # If we have time, try a few more local restarts with small perturbations
    if best_valid_params is not None and time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            pert = np.random.normal(0, 0.1, size=best_valid_params.shape)
            try:
                res = minimize(local_obj, best_valid_params + pert, method='Powell',
                               options={'maxiter': 150, 'xtol': 1e-5, 'ftol': 1e-6})
                if res.fun < 10.0:
                    pos = np.array([[res.x[i * 3], res.x[i * 3 + 1]] for i in range(12)])
                    ang = np.array([res.x[i * 3 + 2] for i in range(12)])
                    if hexagons_disjoint(pos, ang):
                        all_verts = []
                        for i in range(12):
                            v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                            all_verts.append(v)
                        all_verts = np.vstack(all_verts)
                        side, _ = find_best_outer_orientation_fast(all_verts)
                        if side < best_valid_side:
                            best_valid_side = side
                            best_valid_params = res.x.copy()
            except Exception:
                pass

    if best_valid_side < 1e8 and best_valid_params is not None:
        best_positions = np.array([[best_valid_params[i * 3], best_valid_params[i * 3 + 1]] for i in range(12)])
        best_angles = np.array([best_valid_params[i * 3 + 2] for i in range(12)])
        all_verts = []
        for i in range(12):
            v = hexagon_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        outer_side, outer_angle_rad = find_best_outer_orientation_fast(all_verts)
        angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        mins = dots.min(axis=0)
        maxs = dots.max(axis=0)
        mids = (mins + maxs) / 2.0
        c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
        outer_center = tuple(c)
        outer_angle_deg = np.degrees(outer_angle_rad)
    else:
        outer_center = (0.0, 0.0)
        outer_side = 10.0
        outer_angle_deg = 0.0
        best_positions = valid_configs[0][0]
        best_angles = valid_configs[0][1]

    inner = [[best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])] for i in range(12)]
    return inner, outer_center, outer_side, outer_angle_deg
# EVOLVE-BLOCK-END
