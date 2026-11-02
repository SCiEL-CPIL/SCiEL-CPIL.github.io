import numpy as np
import time
import math

N = 12
SQRT3 = math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    # --- Geometry helpers ---
    def hex_vertices(cx, cy, s, ang_rad):
        a = ang_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([cx + s * np.cos(a), cy + s * np.sin(a)])

    def outer_side_from_positions(positions, angles, outer_cx=0.0, outer_cy=0.0, outer_ang_rad=0.0):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        all_verts = []
        for i in range(N):
            all_verts.append(hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
        all_verts = np.vstack(all_verts)
        max_req = 0.0
        for k in range(6):
            ea = outer_ang_rad + np.radians(k * 60)
            nx, ny = np.cos(ea), np.sin(ea)
            proj = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            proj_c = proj - (outer_cx * nx + outer_cy * ny)
            mp = np.max(proj_c)
            if mp > 0:
                req = mp * 2.0 / SQRT3
                if req > max_req:
                    max_req = req
        return max_req

    def sat_disjoint(p1, p2, tol=1e-7):
        for poly in [p1, p2]:
            for i in range(6):
                v1, v2 = poly[i], poly[(i + 1) % 6]
                edge = v2 - v1
                nrm = np.array([-edge[1], edge[0]])
                nn = np.linalg.norm(nrm)
                if nn < 1e-10:
                    continue
                nrm = nrm / nn
                pr1 = p1 @ nrm
                pr2 = p2 @ nrm
                if np.max(pr1) < np.min(pr2) - tol or np.max(pr2) < np.min(pr1) - tol:
                    return True
        return False

    def check_validity(positions, angles):
        verts = [hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(verts[i], verts[j]):
                    return False
        return True

    def polygon_distance(p1, p2):
        min_d = np.inf
        for poly in [p1, p2]:
            for i in range(6):
                v1, v2 = poly[i], poly[(i + 1) % 6]
                edge = v2 - v1
                nrm = np.array([-edge[1], edge[0]])
                nn = np.linalg.norm(nrm)
                if nn < 1e-10:
                    continue
                nrm = nrm / nn
                pr1 = p1 @ nrm
                pr2 = p2 @ nrm
                d = max(np.min(pr2) - np.max(pr1), np.min(pr1) - np.max(pr2))
                if d < min_d:
                    min_d = d
                if min_d <= 0:
                    return min_d
        return min_d

    # --- Simulated Annealing (structural idea from inspiration 1) ---
    def simulated_annealing(init_pos, init_ang, max_iters=80000, T_start=3.0, T_end=0.005):
        current_pos = init_pos.copy()
        current_ang = init_ang.copy()
        current_side = outer_side_from_positions(current_pos, current_ang)
        if not check_validity(current_pos, current_ang):
            for scale in [1.1, 1.3, 1.5, 2.0, 2.5, 3.0]:
                sp = current_pos * scale
                if check_validity(sp, current_ang):
                    current_pos = sp
                    current_side = outer_side_from_positions(current_pos, current_ang)
                    break
            else:
                return init_pos, init_ang, current_side

        best_pos = current_pos.copy()
        best_ang = current_ang.copy()
        best_side = current_side

        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break

            T = T_start * (T_end / T_start) ** (it / max_iters)
            idx = np.random.randint(N)

            # Perturb position of one hexagon
            new_pos = current_pos.copy()
            new_pos[idx] += np.random.normal(0, 0.3, 2)

            # Occasionally perturb angle
            new_ang = current_ang.copy()
            if np.random.random() < 0.15:
                new_ang[idx] += np.random.normal(0, 0.1)

            if not check_validity(new_pos, new_ang):
                continue

            new_side = outer_side_from_positions(new_pos, new_ang)
            delta = new_side - current_side

            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-9)):
                current_pos = new_pos
                current_ang = new_ang
                current_side = new_side
                if current_side < best_side:
                    best_pos = current_pos.copy()
                    best_ang = current_ang.copy()
                    best_side = current_side

        return best_pos, best_ang, best_side

    # --- Refinement via local optimization (from current program) ---
    def refine(positions, angles, outer_cx=0.0, outer_cy=0.0, outer_ang=0.0):
        from scipy.optimize import minimize

        def objective(params):
            pos = params[:2 * N].reshape(N, 2)
            ang = params[2 * N:3 * N]
            verts = [hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i]) for i in range(N)]
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = polygon_distance(verts[i], verts[j])
                    if d < 0:
                        penalty += (-d) ** 2 * 50.0
                    elif d < 0.05:
                        penalty += (0.05 - d) ** 2 * 5.0
            side = outer_side_from_positions(pos, ang, outer_cx, outer_cy, outer_ang)
            return side + penalty

        x0 = np.concatenate([positions.flatten(), angles])
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           bounds=[(-10, 10)] * 2 * N + [(-np.pi, np.pi)] * N,
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            pos = res.x[:2 * N].reshape(N, 2)
            ang = res.x[2 * N:3 * N]
            if check_validity(pos, ang):
                side = outer_side_from_positions(pos, ang, outer_cx, outer_cy, outer_ang)
                return pos, ang, side
        except Exception:
            pass
        return positions, angles, outer_side_from_positions(positions, angles, outer_cx, outer_cy, outer_ang)

    # --- Generate diverse initial configurations ---
    np.random.seed(42)
    configs = []

    # 1. Honeycomb lattice (from inspiration 1)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y])
    init1 = np.array(init1)
    init1 -= init1.mean(axis=0)
    configs.append((init1, np.zeros(N)))

    # 2. Compact ring: 1 center + 6 inner + 5 outer
    def ring_config(ir, or_):
        pos = [[0.0, 0.0]]
        ang = [0.0]
        for i in range(6):
            a = 2 * np.pi * i / 6
            pos.append([ir * np.cos(a), ir * np.sin(a)])
            ang.append(0.0)
        for i in range(5):
            a = 2 * np.pi * (i + 0.5) / 6
            pos.append([or_ * np.cos(a), or_ * np.sin(a)])
            ang.append(0.0)
        pos = np.array(pos)
        ang = np.array(ang)
        pos -= pos.mean(axis=0)
        return pos, ang

    for ir in [1.7, 1.75, 1.8, 1.85]:
        for or_ in [2.8, 3.0, 3.2]:
            configs.append(ring_config(ir, or_))

    # 3. Circular arrangements
    for r in [2.0, 2.2, 2.4, 2.6]:
        for off in [0, np.pi / 12, np.pi / 6]:
            pos = np.array([[r * np.cos(2 * np.pi * i / N + off),
                             r * np.sin(2 * np.pi * i / N + off)] for i in range(N)])
            pos -= pos.mean(axis=0)
            configs.append((pos, np.zeros(N)))

    # 4. Random
    for _ in range(15):
        pos = np.random.uniform(-3, 3, (N, 2))
        pos -= pos.mean(axis=0)
        ang = np.random.uniform(0, 2 * np.pi, N)
        configs.append((pos, ang))

    # --- Main search loop ---
    best_pos = None
    best_ang = None
    best_side = np.inf

    for pos, ang in configs:
        if time.time() > deadline - 10.0:
            break
        res_pos, res_ang, res_side = simulated_annealing(pos, ang)
        if res_side < best_side:
            best_side = res_side
            best_pos = res_pos.copy()
            best_ang = res_ang.copy()

    # Refine best configuration
    if best_pos is not None and time.time() < deadline - 15.0:
        rpos, rang, rside = refine(best_pos, best_ang)
        if rside < best_side:
            best_pos = rpos
            best_ang = rang
            best_side = rside

    # Try different outer hexagon rotations
    if best_pos is not None and time.time() < deadline - 5.0:
        for oa_deg in np.linspace(-30, 30, 13):
            oa = np.radians(oa_deg)
            s = outer_side_from_positions(best_pos, best_ang, 0.0, 0.0, oa)
            if s < best_side:
                best_side = s
                best_outer_ang = oa
        else:
            best_outer_ang = 0.0

    # Fallback
    if best_pos is None:
        best_pos = np.array([[r * np.cos(2 * np.pi * i / N), r * np.sin(2 * np.pi * i / N)]
                             for i in range(N)])
        best_pos -= best_pos.mean(axis=0)
        best_ang = np.zeros(N)
        best_side = outer_side_from_positions(best_pos, best_ang)
        best_outer_ang = 0.0

    # Final validation
    if not check_validity(best_pos, best_ang):
        for scale in [1.2, 1.5, 2.0, 2.5, 3.0]:
            sp = best_pos * scale
            if check_validity(sp, best_ang):
                best_pos = sp
                best_side = outer_side_from_positions(best_pos, best_ang)
                break
        else:
            best_pos = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_ang = np.zeros(N)
            best_side = outer_side_from_positions(best_pos, best_ang)

    exact_side = outer_side_from_positions(best_pos, best_ang, 0.0, 0.0, best_outer_ang)

    inner = [[float(best_pos[i, 0]), float(best_pos[i, 1]), float(np.degrees(best_ang[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(exact_side), float(np.degrees(best_outer_ang))
