import numpy as np
from scipy.optimize import least_squares
import time

N = 12

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    # A regular hexagon (flat-top) with center c, side s, rotation theta:
    # vertices at c + s * R(theta) @ u_k, u_k = (cos(k*pi/3), sin(k*pi/3))
    # For two such hexagons, the signed distance (penetration) between them is
    # the minimum over the 6 support directions of (support_i - support_j).
    # We use a smooth log-sum-exp approximation of the min.

    def hex_supports(cx, cy, s, th):
        # 6 support values along the hexagon's own edge normals (flat-top):
        # edge normal directions are at angles (k-0.5)*pi/3 for k=0..5
        k = np.arange(6)
        n_ang = (k - 0.5) * np.pi / 3.0
        # outward unit normals in local frame
        nx = np.cos(n_ang)
        ny = np.sin(n_ang)
        # rotate by theta
        ct, st = np.cos(th), np.sin(th)
        rx = nx * ct - ny * st
        ry = nx * st + ny * ct
        # support = dot(center, n) + s * (apothem) ... for regular hexagon
        # support along outward normal n: center dot n + s * (cos(pi/6)) = s*sqrt(3)/2
        apothem = s * np.sqrt(3.0) / 2.0
        sup = cx * rx + cy * ry + apothem
        return sup, rx, ry

    def penetration(s_i, th_i, s_j, th_j, cx_i, cy_i, cx_j, cy_j):
        # min over directions of (sup_i - sup_j) ; if positive -> overlap
        sup_i, ri_x, ri_y = hex_supports(cx_i, cy_i, s_i, th_i)
        sup_j, rj_x, rj_y = hex_supports(cx_j, cy_j, s_j, th_j)
        diff = sup_i - sup_j
        # soft min
        m = np.min(diff)
        # log-sum-exp style soft min: -logsumexp(-diff)
        softmin = -np.log(np.sum(np.exp(-(diff - m)))) - m
        return softmin

    def outer_penetration(cx_i, cy_i, s_i, th_i, cx_o, cy_o, s_o, th_o):
        # inner must be inside outer: for all 6 outer edge normals,
        # support_outer >= support_inner  ->  (sup_o - sup_i) >= 0
        sup_o, ro_x, ro_y = hex_supports(cx_o, cy_o, s_o, th_o)
        sup_i, ri_x, ri_y = hex_supports(cx_i, cy_i, s_i, th_i)
        # we need sup_o - sup_i >= 0 in each of the 6 directions
        diff = sup_o - sup_i
        # violation = max(0, -min(diff)) ; use soft max of (-diff)
        neg = -diff
        m = np.max(neg)
        softmax = np.log(np.sum(np.exp(neg - m))) + m
        return softmax

    def objective(v):
        # v layout:
        # [0:2N]   inner positions (x,y)
        # [2N:3N]  inner angles
        # [3N:3N+2] outer center
        # [3N+2:3N+3] outer side
        # [3N+3:3N+4] outer angle
        pos = v[0:2*N].reshape(N, 2)
        ang = v[2*N:3*N]
        ocx, ocy = v[3*N:3*N+2]
        os = v[3*N+2]
        oth = v[3*N+3]

        res = []
        # pair penetration (want <= 0)
        for i in range(N):
            for j in range(i+1, N):
                p = penetration(1.0, ang[i], 1.0, ang[j],
                                pos[i,0], pos[i,1], pos[j,0], pos[j,1])
                # penalize positive penetration
                res.append(max(0.0, p))
        # outer containment (want violation <= 0)
        for i in range(N):
            p = outer_penetration(pos[i,0], pos[i,1], 1.0, ang[i],
                                  ocx, ocy, os, oth)
            res.append(max(0.0, p))
        # objective: minimize outer side length (plus small penalty for violations)
        res.append(os)
        return np.array(res)

    def objective_smooth(v):
        # fully smooth version (no hard max) for gradient-based solver
        pos = v[0:2*N].reshape(N, 2)
        ang = v[2*N:3*N]
        ocx, ocy = v[3*N:3*N+2]
        os = v[3*N+2]
        oth = v[3*N+3]

        res = []
        for i in range(N):
            for j in range(i+1, N):
                p = penetration(1.0, ang[i], 1.0, ang[j],
                                pos[i,0], pos[i,1], pos[j,0], pos[j,1])
                # smooth softplus of p
                res.append(np.log1p(np.exp(p)))
        for i in range(N):
            p = outer_penetration(pos[i,0], pos[i,1], 1.0, ang[i],
                                  ocx, ocy, os, oth)
            res.append(np.log1p(np.exp(p)))
        res.append(os)
        return np.array(res)

    # Build initial guess: dense hexagonal lattice of 12 unit hexagons
    def hex_lattice_init():
        # 12 hexagons in a compact hex arrangement: 3-4-3-? Let's do rows.
        # Use axial coords for a hex cluster of 12.
        pts = []
        # hex cluster: center + ring(6) + 5 more
        # simpler: place in 4-4-4 grid rows with hex offset
        rows = [4, 4, 4]
        dy = np.sqrt(3.0)  # vertical spacing between rows for touching flat-top hexes
        for r, cnt in enumerate(rows):
            y = r * dy
            x0 = -(cnt-1) * 2.0 / 2.0
            for c in range(cnt):
                x = x0 + c * 2.0
                pts.append((x, y))
        return pts

    pts = hex_lattice_init()
    v0 = np.zeros(3*N + 4)
    for i in range(N):
        v0[2*i] = pts[i][0]
        v0[2*i+1] = pts[i][1]
    v0[2*N:3*N] = 0.0
    v0[3*N] = 0.0
    v0[3*N+1] = 0.0
    v0[3*N+2] = 10.0  # initial outer side
    v0[3*N+3] = 0.0

    best_v = v0.copy()
    best_score = objective(v0)[-1] + 100*np.sum(np.maximum(0, objective(v0)[:-1]))

    # multi-start
    starts = [v0]
    # a couple of perturbed starts
    rng = np.random.default_rng(0)
    for _ in range(2):
        v = v0.copy()
        v[0:2*N] += rng.normal(0, 0.3, size=2*N)
        v[2*N:3*N] += rng.normal(0, 0.1, size=N)
        starts.append(v)

    for vstart in starts:
        if time.time() > DEADLINE:
            break
        try:
            sol = least_squares(objective_smooth, vstart, method='trf',
                                max_nfev=4000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        except Exception:
            continue
        if sol.success or True:
            v = sol.x
            # compute actual (hard) score
            obj = objective(v)
            viol = np.sum(np.maximum(0.0, obj[:-1]))
            score = v[3*N+2] + 100.0 * viol
            if score < best_score:
                best_score = score
                best_v = v.copy()
        if time.time() > DEADLINE:
            break

    v = best_v
    pos = v[0:2*N].reshape(N, 2)
    ang = v[2*N:3*N]
    ocx, ocy = v[3*N], v[3*N+1]
    os = v[3*N+2]
    oth = v[3*N+3]

    inner = [[float(pos[i,0]), float(pos[i,1]), float(np.degrees(ang[i]))] for i in range(N)]
    return inner, (float(ocx), float(ocy)), float(os), float(np.degrees(oth))
