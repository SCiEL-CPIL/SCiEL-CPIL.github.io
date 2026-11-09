"""Unit regular hexagons inside a regular hexagon – compact packing via hex lattice + local search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 145.0

    # --- Geometry helpers ---
    def hex_verts(theta, scale=1.0):
        angles = theta + np.arange(6) * np.pi / 3.0
        return np.column_stack([np.cos(angles), np.sin(angles)]) * scale

    def hex_support(theta, p):
        v = hex_verts(theta)
        return np.max(v @ p)

    def separation_sat(pos1, th1, pos2, th2):
        """Returns 0 if disjoint, >0 if overlapping (penetration depth)."""
        d = pos2 - pos1
        max_overlap = 0.0
        # Normals from hex1
        for k in range(6):
            a = th1 + np.pi / 6 + k * np.pi / 3.0
            n = np.array([np.cos(a), np.sin(a)])
            h1 = hex_support(th1, n)
            h2 = hex_support(th2, -n)
            d_proj = d @ n
            overlap = h1 + h2 - d_proj
            if overlap > max_overlap:
                max_overlap = overlap
        # Normals from hex2
        for k in range(6):
            a = th2 + np.pi / 6 + k * np.pi / 3.0
            n = np.array([np.cos(a), np.sin(a)])
            h1 = hex_support(th1, n)
            h2 = hex_support(th2, -n)
            d_proj = d @ n
            overlap = h1 + h2 - d_proj
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap

    def required_side(pos, thetas, rot=0.0):
        """Min side of outer hexagon centered at origin containing all inner hexagons."""
        sqrt3_2 = np.sqrt(3.0) / 2.0
        max_s = 0.0
        for k in range(6):
            a = rot + k * np.pi / 3.0
            n = np.array([np.cos(a), np.sin(a)])
            for i in range(N):
                v = hex_verts(thetas[i])
                s_inner = np.max(v @ n)
                d = pos[i] @ n
                s = (d + s_inner) / sqrt3_2
                s2 = (-d - s_inner) / sqrt3_2
                if s > max_s:
                    max_s = s
                if s2 > max_s:
                    max_s = s2
        return max_s

    # --- Honeycomb initial placement ---
    def initial_honeycomb():
        positions = []
        angles = []
        a1 = np.array([2.0, 0.0])
        a2 = np.array([1.0, np.sqrt(3.0)])
        # Diamond cluster
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if abs(dx) + abs(dy) + abs(dx + dy) <= 6:
                    p = dx * a1 + dy * a2
                    positions.append(p)
                    angles.append(0.0)
        # Keep N closest to origin
        if len(positions) > N:
            idx = np.argsort([np.linalg.norm(p) for p in positions])[:N]
            positions = [positions[i] for i in idx]
            angles = [angles[i] for i in idx]
        return np.array(positions), np.array(angles)

    pos0, ang0 = initial_honeycomb()

    # --- Objective for optimizer ---
    def objective(x):
        pos = x[:2 * N].reshape(N, 2)
        thetas = x[2 * N:]
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                ov = separation_sat(pos[i], thetas[i], pos[j], thetas[j])
                if ov > 0:
                    pen += ov ** 2
        side = required_side(pos, thetas, 0.0)
        return side + 200.0 * pen

    # --- Multi-start local optimization ---
    best_side = np.inf
    best_x = None
    rng = np.random.default_rng(123)
    n_starts = 50

    for s in range(n_starts):
        if time.time() > DEADLINE - 10:
            break
        if s == 0:
            pos_s = pos0.copy()
            ang_s = ang0.copy()
        elif s == 1:
            # Slightly different initial: shift angles
            pos_s = pos0.copy()
            ang_s = ang0 + rng.uniform(-0.1, 0.1, N)
        else:
            pos_s = pos0 + rng.normal(0, 0.5, (N, 2))
            ang_s = ang0 + rng.uniform(-0.3, 0.3, N)

        x0 = np.concatenate([pos_s.ravel(), ang_s])
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            obj_val = objective(res.x)
            # Check validity
            pos = res.x[:2 * N].reshape(N, 2)
            thetas = res.x[2 * N:]
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if separation_sat(pos[i], thetas[i], pos[j], thetas[j]) > 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                side = required_side(pos, thetas, 0.0)
                if side < best_side:
                    best_side = side
                    best_x = res.x.copy()
        except Exception:
            pass

    # --- Polish pass ---
    if best_x is not None and time.time() < DEADLINE - 20:
        try:
            res2 = minimize(objective, best_x, method='L-BFGS-B',
                            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            pos = res2.x[:2 * N].reshape(N, 2)
            thetas = res2.x[2 * N:]
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if separation_sat(pos[i], thetas[i], pos[j], thetas[j]) > 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                side = required_side(pos, thetas, 0.0)
                if side < best_side:
                    best_side = side
                    best_x = res2.x.copy()
        except Exception:
            pass

    # --- Fallback if no valid solution found ---
    if best_x is None:
        side = required_side(pos0, ang0, 0.0) + 0.1
        inner = [[pos0[i][0], pos0[i][1], np.degrees(ang0[i])] for i in range(N)]
        return inner, (0.0, 0.0), side, 0.0

    pos = best_x[:2 * N].reshape(N, 2)
    thetas = best_x[2 * N:]
    side = required_side(pos, thetas, 0.0) + 0.005

    inner = [[pos[i][0], pos[i][1], np.degrees(thetas[i])] for i in range(N)]
    return inner, (0.0, 0.0), side, 0.0

# EVOLVE-BLOCK-END
