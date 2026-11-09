# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Packs 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length (maximizing 1/side).

    Returns
        inner_hex_data: np.ndarray of shape (11,3) -> (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,)  -> (x, y, angle_deg)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 560  # safety margin

    SQRT3 = np.sqrt(3.0)
    hex_angle_offsets = np.radians(60 * np.arange(6))
    hex_cos = np.cos(hex_angle_offsets)
    hex_sin = np.sin(hex_angle_offsets)

    def hex_vertices(cx, cy, s, a_rad):
        angles = a_rad + hex_angle_offsets
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def all_vertices(inner):
        # inner: (11,3) -> (66,2)
        cx = inner[:, 0]
        cy = inner[:, 1]
        a = np.radians(inner[:, 2])
        ca = np.cos(a)[:, None]
        sa = np.sin(a)[:, None]
        vx = cx[:, None] + ca * hex_cos[None, :]
        vy = cy[:, None] + sa * hex_sin[None, :]
        return np.column_stack([vx.ravel(), vy.ravel()])

    def hexagons_separated(v1, v2):
        # v1, v2: (6,2). True if separated by > 1e-6 on at least one of 12 normals.
        for i in range(6):
            e1 = v1[(i + 1) % 6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm = np.linalg.norm(n1)
            if norm > 1e-15:
                n1 = n1 / norm
            if v1 @ n1 > v2 @ n1:
                p1, p2 = v1 @ n1, v2 @ n1
            else:
                p1, p2 = v2 @ n1, v1 @ n1
            if p1.max() < p2.min() - 1e-6:
                return True
        for i in range(6):
            e2 = v2[(i + 1) % 6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm = np.linalg.norm(n2)
            if norm > 1e-15:
                n2 = n2 / norm
            if v1 @ n2 > v2 @ n2:
                p1, p2 = v1 @ n2, v2 @ n2
            else:
                p1, p2 = v2 @ n2, v1 @ n2
            if p1.max() < p2.min() - 1e-6:
                return True
        return False

    def enclosing_hex_side(allv):
        # allv: (N,2) already centered at origin (outer center). Returns minimal side.
        s = 0.0
        for i in range(6):
            a_i = np.radians(60 * i)
            a_ip1 = np.radians(60 * (i + 1))
            e = np.array([np.cos(a_ip1) - np.cos(a_i), np.sin(a_ip1) - np.sin(a_i)])
            crosses = allv[:, 0] * e[1] - allv[:, 1] * e[0]
            min_cross = np.min(crosses)
            required_s = -2.0 * min_cross / SQRT3
            if required_s > s:
                s = required_s
        return max(s, 0.0)

    # ---- Symmetry reduction ----
    # Outer hexagon fixed: center (0,0), angle 0.
    # 6-fold dihedral symmetry: flower packing.
    # 7 unique inner hexagons:
    #   h0 : center, angle a0
    #   h1 : (d1, 0), angle a1, replicated at 60*i
    #   h2 : (d2*cos30, d2*sin30), angle a2, replicated at 60*i+30
    #   h3 : (d3*cos30, d3*sin30), angle a3, replicated at 60*i+30
    # Total: 1 + 6 + 6 + 6 = 19? No: 1 + 6 + 6 = 13, need 11.
    # Use: center(1) + ring1(6) + ring2(4) = 11.
    #   h1 : (d1,0), angle a1, at 60*i for i=0..5  -> 6 hexagons
    #   h2 : (d2*cos60, d2*sin60), angle a2, at 60*i+60 for i=0..3 -> 4 hexagons
    # 1 + 6 + 4 = 11.

    def expand(inner_sym):
        # inner_sym: (7,3) rows [h0, h1, h2] where:
        #   h0 = (cx, cy, a0)
        #   h1 = (d1, 0, a1)   -> 6 copies at angle 60*i
        #   h2 = (d2, 60, a2)  -> 4 copies at angle 60*i+60, i=0..3
        h0 = inner_sym[0]
        h1 = inner_sym[1]
        h2 = inner_sym[2]
        rows = [h0]
        d1, a1 = h1[0], h1[2]
        for i in range(6):
            ang = 60.0 * i
            ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            rows.append([d1 * ca, d1 * sa, a1 + ang])
        d2, a2 = h2[0], h2[2]
        for i in range(4):
            ang = 60.0 * i + 60.0
            ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            rows.append([d2 * ca, d2 * sa, a2 + ang])
        return np.array(rows)

    def objective(inner_sym):
        if not np.all(np.isfinite(inner_sym)):
            return 1e10
        inner = expand(inner_sym)
        verts = all_vertices(inner)  # (66,2), already centered at origin
        s = enclosing_hex_side(verts)

        # Separation penalty over all pairs
        pen = 0.0
        vlist = [verts[6 * i:6 * (i + 1)] for i in range(11)]
        for i in range(11):
            for j in range(i + 1, 11):
                if not hexagons_separated(vlist[i], vlist[j]):
                    # compute min gap over 12 normals
                    min_gap = np.inf
                    vi, vj = vlist[i], vlist[j]
                    for vi_, vj_ in [(vi, vj), (vj, vi)]:
                        for k in range(6):
                            e = vi_[(k + 1) % 6] - vi_[k]
                            n = np.array([-e[1], e[0]])
                            nn = np.linalg.norm(n)
                            if nn > 1e-15:
                                n = n / nn
                            p1 = vi @ n
                            p2 = vj @ n
                            gap = max(p1.min() - p2.max(), p2.min() - p1.max())
                            if gap < min_gap:
                                min_gap = gap
                    if min_gap < 1e-6:
                        pen += (1e-6 - min_gap) ** 2
        return s + 1000.0 * pen

    # Initial symmetric guess: flower + 4
    sqrt3 = np.sqrt(3.0)
    x0 = np.array([
        [0.0, 0.0, 0.0],      # h0 center
        [2.0, 0.0, 0.0],      # h1 ring1
        [2.5, 60.0, 0.0],     # h2 ring2
    ])

    best_params = x0.copy()
    best_val = objective(x0)

    # Multi-start local optimization
    rng = np.random.default_rng(42)
    num_starts = 12
    for start_idx in range(num_starts):
        if time.time() > deadline - 40:
            break
        if start_idx == 0:
            x_start = x0.copy()
        else:
            x_start = x0.copy()
            scale = 0.4 * (start_idx + 1)
            x_start[0, :2] += rng.normal(0, 0.3, 2)
            x_start[0, 2] += rng.normal(0, 15)
            x_start[1, 0] += rng.normal(0, scale)
            x_start[1, 2] += rng.normal(0, 30)
            x_start[2, 0] += rng.normal(0, scale)
            x_start[2, 2] += rng.normal(0, 30)

        try:
            res = minimize(objective, x_start, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # High-precision polish
    if time.time() < deadline - 10:
        try:
            res2 = minimize(objective, best_params, method='COBYLA',
                            options={'maxiter': 2000, 'rhobeg': 1e-7, 'tol': 1e-12})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res.x.copy() if False else res2.x.copy()
        except Exception:
            pass

    # Extract
    inner = expand(best_params)
    verts = all_vertices(inner)
    s = enclosing_hex_side(verts)
    s = max(s, np.sqrt(11.0))

    inner_hex_data = inner  # (11,3)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return inner_hex_data, outer_hex_data, float(s)

# EVOLVE-BLOCK-END
