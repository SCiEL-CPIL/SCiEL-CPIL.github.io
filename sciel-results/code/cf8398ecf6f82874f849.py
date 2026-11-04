"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 140.0  # safety margin below 150s

# Pre-compute base hexagon vertices (side=1, angle=0)
_BASE_ANGLES = np.deg2rad(np.arange(0, 360, 60))
_BASE_VERTS = np.column_stack((np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)))

def _hex_vertices_array(cx, cy, angle_deg):
    """Returns (6, 2) array of vertices for a hexagon."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return (R @ _BASE_VERTS.T).T + np.array([cx, cy])

def _check_pair_collision(verts1, verts2):
    """Fast SAT check for two convex hexagons represented as (6,2) arrays.
    Returns True if they collide (overlap)."""
    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack((-edges[:, 1], edges[:, 0]))
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        return normals / lengths

    n1 = get_normals(verts1)
    n2 = get_normals(verts2)
    all_normals = np.vstack((n1, n2))
    proj1 = verts1 @ all_normals.T
    proj2 = verts2 @ all_normals.T
    min1 = np.min(proj1, axis=1)
    max1 = np.max(proj1, axis=1)
    min2 = np.min(proj2, axis=1)
    max2 = np.max(proj2, axis=1)
    separates = (max1 < min2) | (max2 < min1)
    if np.any(separates):
        return False
    return True

def _is_valid_fast(cfg):
    """cfg: list of [x, y, angle_deg]. Returns True if all pairs are disjoint."""
    verts_list = []
    for p in cfg:
        verts_list.append(_hex_vertices_array(p[0], p[1], p[2]))
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, _ = cfg[i]
            xj, yj, _ = cfg[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            if dist_sq > 4.0:
                continue
            if _check_pair_collision(verts_list[i], verts_list[j]):
                return False
    return True

def _outer_hex_side_fast(cfg, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = np.deg2rad(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals_angles = a + np.deg2rad(90 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(normals_angles), np.sin(normals_angles)))
    cx, cy = center
    max_need = 0.0
    for p in cfg:
        verts = _hex_vertices_array(p[0], p[1], p[2])
        rel_verts = verts - np.array([cx, cy])
        projections = rel_verts @ normals.T
        max_proj = np.max(projections)
        if max_proj > max_need:
            max_need = max_proj
    return max_need / apothem_unit

def _lattice_topologies():
    """Generate hexagonal lattice topologies for 12 unit hexagons."""
    topologies = []
    sqrt3 = math.sqrt(3)

    # Topology A: 4-3-4-1
    offsets_A = [
        (0, 0), (3, 0), (6, 0), (9, 0),
        (1.5, sqrt3/2), (4.5, sqrt3/2), (7.5, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (4.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_A, [0.0] * 12))

    # Topology B: 2-3-4-3
    offsets_B = [
        (1.5, 0), (4.5, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (1.5, 3*sqrt3/2), (4.5, 3*sqrt3/2), (7.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_B, [0.0] * 12))

    # Topology C: 3-4-3-2
    offsets_C = [
        (0, 0), (3, 0), (6, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2), (9, sqrt3/2),
        (1.5, sqrt3), (4.5, sqrt3), (7.5, sqrt3),
        (3, 3*sqrt3/2), (6, 3*sqrt3/2),
    ]
    topologies.append((offsets_C, [0.0] * 12))

    # Topology D: 2-4-3-3
    offsets_D = [
        (1.5, 0), (4.5, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2), (9, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3),
        (1.5, 3*sqrt3/2), (4.5, 3*sqrt3/2), (7.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_D, [0.0] * 12))

    # Topology E: 3-3-4-2
    offsets_E = [
        (0, 0), (3, 0), (6, 0),
        (1.5, sqrt3/2), (4.5, sqrt3/2), (7.5, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (3, 3*sqrt3/2), (6, 3*sqrt3/2),
    ]
    topologies.append((offsets_E, [0.0] * 12))

    # Topology F: 4-2-4-2
    offsets_F = [
        (0, 0), (3, 0), (6, 0), (9, 0),
        (1.5, sqrt3/2), (4.5, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (1.5, 3*sqrt3/2), (4.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_F, [0.0] * 12))

    # Topology G: 2-4-2-4
    offsets_G = [
        (1.5, 0), (4.5, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2), (9, sqrt3/2),
        (1.5, sqrt3), (4.5, sqrt3),
        (0, 3*sqrt3/2), (3, 3*sqrt3/2), (6, 3*sqrt3/2), (9, 3*sqrt3/2),
    ]
    topologies.append((offsets_G, [0.0] * 12))

    # Topology H: 3-4-2-3
    offsets_H = [
        (0, 0), (3, 0), (6, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2), (9, sqrt3/2),
        (1.5, sqrt3), (4.5, sqrt3),
        (0, 3*sqrt3/2), (3, 3*sqrt3/2), (6, 3*sqrt3/2),
    ]
    topologies.append((offsets_H, [0.0] * 12))

    # Topology I: 4-3-2-3
    offsets_I = [
        (0, 0), (3, 0), (6, 0), (9, 0),
        (1.5, sqrt3/2), (4.5, sqrt3/2), (7.5, sqrt3/2),
        (3, sqrt3), (6, sqrt3),
        (0, 3*sqrt3/2), (3, 3*sqrt3/2), (6, 3*sqrt3/2),
    ]
    topologies.append((offsets_I, [0.0] * 12))

    # Topology J: 3-2-4-3
    offsets_J = [
        (0, 0), (3, 0), (6, 0),
        (1.5, sqrt3/2), (4.5, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (0, 3*sqrt3/2), (3, 3*sqrt3/2), (6, 3*sqrt3/2),
    ]
    topologies.append((offsets_J, [0.0] * 12))

    # Topology K: 2-3-3-4
    offsets_K = [
        (1.5, 0), (4.5, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2),
        (1.5, sqrt3), (4.5, sqrt3), (7.5, sqrt3),
        (0, 3*sqrt3/2), (3, 3*sqrt3/2), (6, 3*sqrt3/2), (9, 3*sqrt3/2),
    ]
    topologies.append((offsets_K, [0.0] * 12))

    # Topology L: 3-3-3-3
    offsets_L = [
        (0, 0), (3, 0), (6, 0),
        (1.5, sqrt3/2), (4.5, sqrt3/2), (7.5, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3),
        (1.5, 3*sqrt3/2), (4.5, 3*sqrt3/2), (7.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_L, [0.0] * 12))

    # Topology M: 1-4-4-3
    offsets_M = [
        (4.5, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2), (9, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (1.5, 3*sqrt3/2), (4.5, 3*sqrt3/2), (7.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_M, [0.0] * 12))

    # Topology N: 3-4-4-1
    offsets_N = [
        (0, 0), (3, 0), (6, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2), (9, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (4.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_N, [0.0] * 12))

    return topologies

def _topology_score(offsets, angles, global_rotation_deg):
    """Compute the outer hexagon side length for a given topology with global rotation."""
    a_rad = np.deg2rad(global_rotation_deg)
    ca, sa = math.cos(a_rad), math.sin(a_rad)
    cfg = []
    for i, (dx, dy) in enumerate(offsets):
        rx = ca * dx - sa * dy
        ry = sa * dx + ca * dy
        ang = angles[i] + global_rotation_deg
        cfg.append([rx, ry, ang])
    if not _is_valid_fast(cfg):
        return None
    return _outer_hex_side_fast(cfg, (0.0, 0.0), global_rotation_deg)

def _find_best_rotation(offsets, angles, t0):
    """Find best global rotation for a topology using multi-start local search."""
    best_score = float('inf')
    best_theta = 0.0
    sqrt3 = math.sqrt(3)

    # Multi-start: try several initial rotations
    for start_theta in np.linspace(0, 60, 7):
        if time.time() - t0 > TIME_LIMIT - 5.0:
            break
        theta = start_theta
        # Local refinement using gradient-free hill climbing
        step = 5.0
        for _ in range(30):
            if time.time() - t0 > TIME_LIMIT - 5.0:
                break
            best_local = None
            best_local_theta = None
            for delta in [-step, -step/3, step/3, step]:
                trial = (theta + delta) % 60
                s = _topology_score(offsets, angles, trial)
                if s is not None and (best_local is None or s < best_local):
                    best_local = s
                    best_local_theta = trial
            if best_local is None:
                break
            if best_local < best_score:
                best_score = best_local
                best_theta = best_local_theta
            if abs(best_local_theta - theta) < 1e-4:
                break
            theta = best_local_theta
            step *= 0.7
    return best_theta, best_score

def construct():
    t0 = time.time()
    random.seed(42)

    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0

    # Get all lattice topologies
    topologies = _lattice_topologies()

    # For each topology, find best global rotation
    for offsets, angles in topologies:
        if time.time() - t0 > TIME_LIMIT - 10.0:
            break
        theta_opt, score_opt = _find_best_rotation(offsets, angles, t0)
        if score_opt < best_score:
            best_score = score_opt
            a_rad = np.deg2rad(theta_opt)
            ca, sa = math.cos(a_rad), math.sin(a_rad)
            best_cfg = []
            for i, (dx, dy) in enumerate(offsets):
                rx = ca * dx - sa * dy
                ry = sa * dx + ca * dy
                ang = angles[i] + theta_opt
                best_cfg.append([rx, ry, ang])
            best_center = [0.0, 0.0]
            best_angle = theta_opt

    # If no valid topology found, fallback
    if best_cfg is None:
        cfg = []
        for i in range(N):
            theta = 2 * math.pi * i / N
            r = 10.0
            cfg.append([r * math.cos(theta), r * math.sin(theta), 0.0])
        best_cfg = cfg
        best_score = 20.0
        best_center = [0.0, 0.0]
        best_angle = 0.0

    # Simulated Annealing polish
    T = 0.5
    T_end = 0.003
    alpha_sa = 0.9997
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    nm_time_budget = TIME_LIMIT - 5.0

    while time.time() - t0 < nm_time_budget:
        r = random.random()
        if r < 0.45:
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.1 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.65:
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 3.0 + 2.0 * T)
        elif r < 0.75:
            old_cx, old_cy = center[0], center[1]
            step = 0.05 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.85:
            old_angle = angle
            angle += random.gauss(0, 1.0 + 1.0 * T)
        else:
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.08 * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)

        if not _is_valid_fast(cfg):
            if r < 0.45:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.65:
                cfg[idx][2] = old_ang
            elif r < 0.75:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.85:
                angle = old_angle
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
            T = max(T_end, T * alpha_sa)
            continue

        new_score = _outer_hex_side_fast(cfg, tuple(center), angle)
        if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_cfg = [row[:] for row in cfg]
                best_center = center[:]
                best_angle = angle
        else:
            if r < 0.45:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.65:
                cfg[idx][2] = old_ang
            elif r < 0.75:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.85:
                angle = old_angle
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        T = max(T_end, T * alpha_sa)

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
