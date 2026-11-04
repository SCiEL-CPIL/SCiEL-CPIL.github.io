import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)

        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i

        return max_proj * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        min_gap = float('inf')
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9:
                continue
            axis = normal1 / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap

        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9:
                continue
            axis = normal2 / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap

        return min_gap

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def compute_penalty(params, penalty_weight=100.0):
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            positions[i][0] = params[3*i]
            positions[i][1] = params[3*i+1]
            angles[i] = params[3*i+2]

        side_len = calc_min_outer_side(positions, angles, 0.0)

        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += penalty_weight * gap**2

        return side_len + penalty

    def best_side_over_angles(positions, angles, coarse):
        """Best outer side length over a set of outer rotations."""
        best = float('inf')
        if coarse:
            outer_angles = [math.degrees(k) for k in range(0, 360, 6)]
        else:
            outer_angles = [math.degrees(k) for k in range(0, 360, 1)]
        for oa in outer_angles:
            s = calc_min_outer_side(positions, angles, oa)
            if s < best:
                best = s
        return best

    # ---- Phase 1: generate good initial configurations from honeycomb lattice ----
    sqrt3 = math.sqrt(3)
    a1 = np.array([1.5, sqrt3/2])
    a2 = np.array([0.0, sqrt3])
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            lattice_points.append(i * a1 + j * a2)
    lattice_points.sort(key=lambda p: float(np.dot(p, p)))
    init_points = np.array(lattice_points[:12])
    init_points -= np.mean(init_points, axis=0)

    best_params = None
    best_score = float('inf')

    for scale in [1.0, 1.01, 1.02, 1.05, 1.1]:
        for rot_deg in [0, 30, 60, 90, 120, 150]:
            rot_rad = math.radians(rot_deg)
            rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                                [math.sin(rot_rad), math.cos(rot_rad)]])
            rotated = (init_points * scale) @ rot_mat.T
            angles = np.full(N, rot_deg)
            if is_valid(rotated, angles, tol=1e-5):
                side = calc_min_outer_side(rotated, angles, 0.0)
                if side < best_score:
                    best_score = side
                    params = np.zeros(3*N)
                    for i in range(N):
                        params[3*i] = rotated[i][0]
                        params[3*i+1] = rotated[i][1]
                        params[3*i+2] = angles[i]
                    best_params = params
        if best_params is not None:
            break

    # Guaranteed-valid fallback
    if best_params is None:
        fallback = init_points * 1.3
        fallback -= np.mean(fallback, axis=0)
        best_params = np.zeros(3*N)
        for i in range(N):
            best_params[3*i] = fallback[i][0]
            best_params[3*i+1] = fallback[i][1]
            best_params[3*i+2] = 0.0
        best_score = calc_min_outer_side(fallback, np.zeros(N), 0.0)

    x0 = best_params.copy()

    # ---- Phase 2: simulated annealing to escape local minima ----
    current_params = x0.copy()
    current_score = best_score
    best_params = x0.copy()
    best_score = current_score

    T = 0.5
    cooling_rate = 0.9995
    step_size = 0.3
    max_iter = 50000

    np.random.seed(42)
    random.seed(42)

    for iteration in range(max_iter):
        if time.time() > deadline - 55.0:
            break
        new_params = current_params.copy()
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            idx = random.randint(0, N-1)
            new_params[3*idx] += np.random.randn() * step_size
            new_params[3*idx+1] += np.random.randn() * step_size
            new_params[3*idx+2] += np.random.randn() * step_size * 30

        new_score = compute_penalty(new_params)

        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        else:
            delta = new_score - current_score
            if T > 0:
                if random.random() < math.exp(-delta / T):
                    current_params = new_params
                    current_score = new_score

        T *= cooling_rate
        if iteration % 500 == 0:
            step_size *= 0.95

    # ---- Phase 3: hard-constraint SLSQP (Minkowski-difference SAT distance) ----
    def objective(params):
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            positions[i][0] = params[3*i]
            positions[i][1] = params[3*i+1]
            angles[i] = params[3*i+2]
        return calc_min_outer_side(positions, angles, 0.0)

    def make_constraint(i, j):
        def constraint(params):
            pi = get_hex_verts(params[3*i], params[3*i+1], params[3*i+2])
            pj = get_hex_verts(params[3*j], params[3*j+1], params[3*j+2])
            return sat_gap(pi, pj)
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    def unpack(params):
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            positions[i][0] = params[3*i]
            positions[i][1] = params[3*i+1]
            angles[i] = params[3*i+2]
        return positions, angles

    def try_slsqp(start, maxiter, ftol):
        nonlocal best_params, best_score
        try:
            result = minimize(
                objective, start, method='SLSQP',
                bounds=None, constraints=constraints,
                options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
            )
        except Exception:
            return
        if result.x is None:
            return
        pos, ang = unpack(result.x)
        if is_valid(pos, ang, tol=1e-5):
            side = calc_min_outer_side(pos, ang, 0.0)
            if side < best_score:
                best_score = side
                best_params = result.x.copy()

    if time.time() < deadline - 40.0:
        try_slsqp(best_params, 200, 1e-10)

    # ---- Phase 4: coarse outer-angle search, then fine polish ----
    pos, ang = unpack(best_params)
    coarse_best = float('inf')
    coarse_best_ang = 0.0
    for oa in [math.degrees(k) for k in range(0, 360, 6)]:
        s = calc_min_outer_side(pos, ang, oa)
        if s < coarse_best:
            coarse_best = s
            coarse_best_ang = oa

    if time.time() < deadline - 25.0:
        # fine outer-angle polish around coarse best
        fine_best = coarse_best
        fine_best_ang = coarse_best_ang
        for oa in [coarse_best_ang + math.degrees(k) for k in range(-12, 13)]:
            s = calc_min_outer_side(pos, ang, oa)
            if s < fine_best:
                fine_best = s
                fine_best_ang = oa

        # tight-tolerance inner polish holding outer angle fixed
        def objective_fixed(params):
            p, a = unpack(params)
            return calc_min_outer_side(p, a, fine_best_ang)

        try:
            result = minimize(
                objective_fixed, best_params, method='SLSQP',
                constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-12, 'disp': False}
            )
            if result.x is not None:
                p2, a2 = unpack(result.x)
                if is_valid(p2, a2, tol=1e-6):
                    s2 = calc_min_outer_side(p2, a2, fine_best_ang)
                    if s2 < fine_best:
                        best_params = result.x.copy()
                        best_score = s2
                        fine_best = s2
                        pos, ang = p2, a2
        except Exception:
            pass
        best_score = min(best_score, fine_best)

    # ---- Final: report best over a fine outer-angle scan of the stored config ----
    pos, ang = unpack(best_params)
    final_side = float('inf')
    for oa in [math.degrees(k) for k in range(0, 360, 1)]:
        s = calc_min_outer_side(pos, ang, oa)
        if s < final_side:
            final_side = s

    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], ang[i]])

    return inner, (0.0, 0.0), final_side, 0.0
# EVOLVE-BLOCK-END
