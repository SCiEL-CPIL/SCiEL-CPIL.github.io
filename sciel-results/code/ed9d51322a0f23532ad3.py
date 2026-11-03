import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 143.0
SQRT3 = math.sqrt(3)

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = SQRT3
    
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def get_all_normals(angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _enx[None, :] * c[:, None] - _eny[None, :] * s[:, None]
        ry = _enx[None, :] * s[:, None] + _eny[None, :] * c[:, None]
        return np.stack((rx, ry), axis=-1)

    def check_config_valid_fast(positions, angles_rad):
        """
        Precompute vertices and normals for all hexagons, then check pairs.
        """
        all_verts = get_all_vertices(positions, angles_rad)
        all_norms = get_all_normals(angles_rad)
        
        for i in range(N):
            vi = all_verts[i]
            ni = all_norms[i]
            for j in range(i+1, N):
                vj = all_verts[j]
                nj = all_norms[j]
                
                for k in range(6):
                    n = ni[k]
                    proj_i = vi @ n
                    proj_j = vj @ n
                    if proj_i.max() < proj_j.min() or proj_j.max() < proj_i.min():
                        break
                else:
                    for k in range(6):
                        n = nj[k]
                        proj_i = vi @ n
                        proj_j = vj @ n
                        if proj_i.max() < proj_j.min() or proj_j.max() < proj_i.min():
                            break
                    else:
                        return False
        return True

    def compute_min_outer_side_fast(positions, angles_deg):
        """
        Fast calculation of min outer side length.
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side_per_phi = np.max(side_vals, axis=1)

        best_idx = np.argmin(side_per_phi)
        best_side = side_per_phi[best_idx]
        best_phi = float(best_idx)

        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            na = phi + np.pi/6 + np.arange(6) * np.pi/3
            nxx = np.cos(na)
            nyy = np.sin(na)
            pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
            return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()

        gr = (math.sqrt(5) - 1) / 2
        a = max(0.0, best_phi - 1.0)
        b = min(60.0, best_phi + 1.0)
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = eval_phi(c)
        fd = eval_phi(d)
        for _ in range(10):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = eval_phi(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = eval_phi(d)
        
        best_phi = (a + b) / 2
        best_side = eval_phi(best_phi)

        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        return check_config_valid_fast(positions, angles_rad)

    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])

    def make_honeycomb():
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
        return pos, ang

    def make_ring_config():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            theta = i * np.pi / 3
            pos[i] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(6):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 6] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    def make_two_rows():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            pos[i] = [i * 2 * sqrt3, 0]
            pos[i + 6] = [i * 2 * sqrt3 + sqrt3, 3.0]
        return pos, ang

    def make_compact_hex():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        for i in range(6):
            theta = i * np.pi / 3
            pos[i + 1] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 7] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    def make_discrete_orientations(base_pos, base_ang):
        """
        Try discrete orientations (0, 30, 60) for each hexagon and keep the best valid config.
        """
        best = None
        best_side = float('inf')
        
        # Try a few combinations of discrete angles
        angle_choices = [0.0, 30.0, 60.0]
        
        # For efficiency, try all hexagons at same angle first
        for base_angle in angle_choices:
            new_pos = base_pos.copy()
            new_ang = base_ang.copy()
            for i in range(N):
                new_ang[i] = base_angle
            
            if is_valid_configuration(new_pos, new_ang):
                s, phi, cx, cy = compute_min_outer_side_fast(new_pos, new_ang)
                if s < best_side:
                    best_side = s
                    best = (new_pos.copy(), new_ang.copy())
        
        # Then try mixed orientations: each hexagon independently
        random.seed(42)
        for trial in range(50):
            new_pos = base_pos.copy()
            new_ang = base_ang.copy()
            for i in range(N):
                new_ang[i] = random.choice(angle_choices)
            
            if is_valid_configuration(new_pos, new_ang):
                s, phi, cx, cy = compute_min_outer_side_fast(new_pos, new_ang)
                if s < best_side:
                    best_side = s
                    best = (new_pos.copy(), new_ang.copy())
        
        return best

    init_configs = [make_honeycomb(), make_ring_config(), make_two_rows(), make_compact_hex()]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for pos0, ang0 in init_configs:
        if is_valid_configuration(pos0, ang0):
            s, phi, cx, cy = compute_min_outer_side_fast(pos0, ang0)
            if s < best_side:
                best_side = s
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy
        
        # Also try discrete orientations from this config
        discrete_best = make_discrete_orientations(pos0, ang0)
        if discrete_best is not None:
            dp, da = discrete_best
            s, phi, cx, cy = compute_min_outer_side_fast(dp, da)
            if s < best_side:
                best_side = s
                best_positions = dp.copy()
                best_angles = da.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy

    if best_positions is None:
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side_fast(best_positions, best_angles)

    # Phase 1: Simulated Annealing with periodic discrete orientation restarts
    sa_time_budget = DEADLINE * 0.6
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side_fast(current_positions, current_angles)

    T = 0.3
    T_min = 0.0001
    cooling = 0.9999
    step_size = 0.25
    step_decay = 0.99998

    step_count = 0
    max_steps = 500000
    accept_count = 0
    total_count = 0
    no_improve_count = 0
    last_best_side = best_side
    discrete_restarts_done = 0
    max_discrete_restarts = 5

    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1

        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()

        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 4.0)

        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da

        angles_rad_new = np.radians(new_angles)
        valid = check_config_valid_fast(new_positions, angles_rad_new)

        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.003: step_size = 0.003
            continue

        new_side, new_phi, new_cx, new_cy = compute_min_outer_side_fast(new_positions, new_angles)

        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            accept_count += 1
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        if total_count % 300 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.08:
                step_size *= 0.8
                if step_size < 0.003: step_size = 0.003
            elif acc_rate > 0.4:
                step_size *= 1.15
                if step_size > 0.8: step_size = 0.8
            accept_count = 0
            total_count = 0

        # Periodic discrete orientation restart if stuck
        if no_improve_count > 5000 and discrete_restarts_done < max_discrete_restarts:
            discrete_restarts_done += 1
            discrete_best = make_discrete_orientations(current_positions, current_angles)
            if discrete_best is not None:
                dp, da = discrete_best
                s, phi, cx, cy = compute_min_outer_side_fast(dp, da)
                if s < current_side:
                    current_positions = dp.copy()
                    current_angles = da.copy()
                    current_side = s
                    current_phi = phi
                    current_cx = cx
                    current_cy = cy
                    if s < best_side:
                        best_side = s
                        best_positions = dp.copy()
                        best_angles = da.copy()
                        best_phi = phi
                        best_cx = cx
                        best_cy = cy
                        no_improve_count = 0
            # Reset some SA parameters
            T = max(T, 0.1)
            step_size = max(step_size, 0.1)
            no_improve_count = 0

        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.003: step_size = 0.003

    # Phase 2: Polish using L-BFGS-B with soft-max
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 5.0:
        def smooth_outer_side(params):
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3*i]
                pos[i, 1] = params[3*i+1]
                ang[i] = params[3*i+2]
            
            angles_rad = np.radians(ang)
            if not check_config_valid_fast(pos, angles_rad):
                return 1e6 + random.random() * 10
            
            all_verts = get_all_vertices(pos, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            
            phi = np.radians(30)
            na = phi + np.pi/6 + np.arange(6) * np.pi/3
            nxx = np.cos(na)
            nyy = np.sin(na)
            pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
            abs_pr = np.abs(pr)
            
            beta = 50.0
            max_val = np.max(abs_pr, axis=0)
            smooth_max = max_val + np.log(np.mean(np.exp((abs_pr - max_val[:, None]) * beta), axis=0)) / beta
            
            return (2.0 / sqrt3) * np.max(smooth_max)

        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = best_positions[i, 0]
            init_params[3*i+1] = best_positions[i, 1]
            init_params[3*i+2] = best_angles[i]
        
        try:
            res = minimize(smooth_outer_side, init_params, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_side + 0.5:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side_fast(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception:
            pass
        
        # Final fine-tuning of phi
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 2.0:
            def eval_phi_fine(phi_deg):
                phi = np.radians(phi_deg)
                angles_rad = np.radians(best_angles)
                all_verts = get_all_vertices(best_positions, angles_rad)
                verts_flat = all_verts.reshape(-1, 2)
                center = np.mean(verts_flat, axis=0)
                offsets = verts_flat - center
                na = phi + np.pi/6 + np.arange(6) * np.pi/3
                nxx = np.cos(na)
                nyy = np.sin(na)
                pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
                return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()
            
            best_phi_fine = best_phi
            best_side_fine = eval_phi_fine(best_phi)
            
            for start_phi in [best_phi - 2.0, best_phi, best_phi + 2.0]:
                lo = max(0.0, start_phi - 1.0)
                hi = min(60.0, start_phi + 1.0)
                try:
                    res = minimize_scalar(eval_phi_fine, bounds=(lo, hi), method='bounded',
                                          options={'xatol': 1e-10, 'maxiter': 200})
                    if res.success and res.fun < best_side_fine:
                        best_side_fine = res.fun
                        best_phi_fine = res.x
                except Exception:
                    pass
            
            if best_side_fine < best_side - 1e-10:
                best_side = best_side_fine
                best_phi = best_phi_fine
                _, _, best_cx, best_cy = compute_min_outer_side_fast(best_positions, best_angles)

    # Phase 3: Final discrete orientation polish
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 3.0:
        discrete_best = make_discrete_orientations(best_positions, best_angles)
        if discrete_best is not None:
            dp, da = discrete_best
            s, phi, cx, cy = compute_min_outer_side_fast(dp, da)
            if s < best_side:
                best_side = s
                best_positions = dp.copy()
                best_angles = da.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy

    if not is_valid_configuration(best_positions, best_angles):
        for pos0, ang0 in init_configs:
            if is_valid_configuration(pos0, ang0):
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side_fast(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side_fast(best_positions, best_angles)
    
    try:
        angles_rad_final = np.radians(best_angles)
        all_verts_final = get_all_vertices(best_positions, angles_rad_final)
        verts_flat_final = all_verts_final.reshape(-1, 2)
        center_final = np.mean(verts_flat_final, axis=0)
        offsets_final = verts_flat_final - center_final
        
        def exact_side_for_phi(phi_deg):
            phi = np.radians(phi_deg)
            na = phi + np.pi/6 + np.arange(6) * np.pi/3
            nxx = np.cos(na)
            nyy = np.sin(na)
            pr = offsets_final[:, 0:1] * nxx[None, :] + offsets_final[:, 1:2] * nyy[None, :]
            return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()
        
        phi_grid = np.linspace(max(0, best_phi - 0.5), min(60, best_phi + 0.5), 101)
        sides_grid = np.array([exact_side_for_phi(p) for p in phi_grid])
        best_idx = np.argmin(sides_grid)
        if sides_grid[best_idx] < best_side - 1e-12:
            best_phi = float(phi_grid[best_idx])
            best_side = float(sides_grid[best_idx])
            best_cx = float(center_final[0])
            best_cy = float(center_final[1])
    except Exception:
        pass
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
