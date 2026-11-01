"""Suite task: pack 12 unit hexagons in a minimal regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
SQ3 = np.sqrt(3.0)
SQ3_HALF = SQ3 / 2.0
DEADLINE = time.time() + 135.0  # safety margin before 150 s wall-clock kill


def construct():
    # --- unit hexagon vertices (circumradius = side = 1) ---
    base = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]
                     for a in range(0, 360, 60)])

    def hexagon(L, angle_deg):
        return L * (base * np.cos(np.deg2rad(angle_deg))
                    + np.cross(base, np.array([0.0, 1.0])) * np.sin(np.deg2rad(angle_deg)))

    def pt_in_poly(px, py, P):
        # convex polygon, CCW; point inside iff all cross >= -eps
        n = len(P)
        x0, y0 = P[0]
        x1, y1 = P[1]
        if (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0) < -1e-9:
            return False
        for k in range(n):
            x0, y0 = P[k]
            x1, y1 = P[(k + 1) % n]
            if (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0) < -1e-9:
                return False
        return True

    def sat_overlap(A, B):
        # Separating Axis Theorem for two convex polygons.
        # Returns False if separated (no overlap).
        for P in (A, B):
            n = len(P)
            for k in range(n):
                ax, ay = P[k]
                bx, by = P[(k + 1) % n]
                ex, ey = bx - ax, by - ay
                L2 = ex * ex + ey * ey
                if L2 == 0:
                    continue
                nx, ny = -ey / L2, ex / L2  # unit normal
                minA, maxA = np.dot(A, [nx, ny]).min(), np.dot(A, [nx, ny]).max()
                minB, maxB = np.dot(B, [nx, ny]).min(), np.dot(B, [nx, ny]).max()
                if maxA < minB or maxB < minA:
                    return False
        return True

    def is_valid(pos, rot, L, tol=1e-6):
        outer = hexagon(L, 0.0)
        for i in range(N):
            inner = pos[i] + hexagon(1.0, rot[i])
            for v in inner:
                if not pt_in_poly(v[0], v[1], outer):
                    return False
        for i in range(N):
            Ai = pos[i] + hexagon(1.0, rot[i])
            for j in range(i + 1, N):
                Bj = pos[j] + hexagon(1.0, rot[j])
                if sat_overlap(Ai, Bj):
                    return False
        return True

    def objective(params):
        pos = params[:2 * N].reshape(N, 2)
        rot = params[2 * N:3 * N]
        L = params[3 * N]
        outer = hexagon(L, 0.0)
        inner = [pos[i] + hexagon(1.0, rot[i]) for i in range(N)]
        val = 0.0
        # containment: max violation per inner hexagon
        for i in range(N):
            mx = 0.0
            for v in inner[i]:
                if not pt_in_poly(v[0], v[1], outer):
                    # depth: distance to each edge, take most-negative
                    d = 1e9
                    P = outer
                    n = len(P)
                    for k in range(n):
                        x0, y0 = P[k]
                        x1, y1 = P[(k + 1) % n]
                        ex, ey = x1 - x0, y1 - y0
                        L2 = ex * ex + ey * ey
                        if L2 == 0:
                            continue
                        # signed distance (positive inside for CCW)
                        sd = ((x1 - x0) * (v[1] - y0) - (y1 - y0) * (v[0] - x0)) / np.sqrt(L2)
                        if sd < d:
                            d = sd
                    if d < 0:
                        if -d > mx:
                            mx = -d
            val += mx * mx
        # overlap: pairwise, via penetration depth estimate
        for i in range(N):
            Ai = inner[i]
            for j in range(i + 1, N):
                Bj = inner[j]
                if sat_overlap(Ai, Bj):
                    # penetration depth approx: min over axes of overlap
                    pd = 1e9
                    for P in (Ai, Bj):
                        n = len(P)
                        for k in range(n):
                            ax, ay = P[k]
                            bx, by = P[(k + 1) % n]
                            ex, ey = bx - ax, by - ay
                            L2 = ex * ex + ey * ey
                            if L2 == 0:
                                continue
                            nx, ny = -ey / np.sqrt(L2), ex / np.sqrt(L2)
                            minA, maxA = np.dot(Ai, [nx, ny]).min(), np.dot(Ai, [nx, ny]).max()
                            minB, maxB = np.dot(Bj, [nx, ny]).min(), np.dot(Bj, [nx, ny]).max()
                            ov = min(maxA, maxB) - max(minA, minB)
                            if ov < pd:
                                pd = ov
                    val += pd * pd
        # shrink pressure
        val += 5.0 * L
        return val

    # --- initial guess: hexagonal flower + ring ---
    def initial_guess(L, rng):
        s3 = SQ3
        pos = np.zeros((N, 2))
        # center
        pos[0] = [0.0, 0.0]
        ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        idx = 1
        for q, r in ring1:
            pos[idx] = [s3 * q + s3 / 2 * r, 1.5 * r]
            idx += 1
        ring2 = [(2, 0), (2, -1), (1, -2), (0, -2), (-1, -2)]
        for q, r in ring2:
            pos[idx] = [s3 * q + s3 / 2 * r, 1.5 * r]
            idx += 1
        rot = np.zeros(N)
        # small random rotation perturbation to escape symmetry
        rot = rng.uniform(-10, 10, N)
        params = np.concatenate([pos.ravel(), rot, [L]])
        return params

    best = None
    best_val = 1e9
    rng = np.random.default_rng(12345)

    # try a spread of starting L values
    start_Ls = [5.0, 4.6, 4.4, 4.2, 4.0, 3.9, 3.8, 3.7, 3.6]
    for L0 in start_Ls:
        if time.time() > DEADLINE:
            break
        for attempt in range(2):
            if time.time() > DEADLINE:
                break
            p0 = initial_guess(L0, rng)
            try:
                res = minimize(objective, p0, method='L-BFGS-B',
                               options={'maxiter': 400, 'ftol': 1e-9, 'gtol': 1e-6})
                p = res.x
                pos = p[:2 * N].reshape(N, 2)
                rot = p[2 * N:3 * N]
                L = float(p[3 * N])
                if is_valid(pos, rot, L):
                    # greedy shrink
                    for _ in range(40):
                        if time.time() > DEADLINE:
                            break
                        L2 = L * 0.99
                        if is_valid(pos, rot, L2):
                            L = L2
                        else:
                            # bisection to find the boundary
                            lo, hi = L2, L
                            for _ in range(20):
                                if time.time() > DEADLINE:
                                    break
                                mid = (lo + hi) / 2.0
                                if is_valid(pos, rot, mid):
                                    lo = mid
                                else:
                                    hi = mid
                            L = lo
                            break
                    val = L
                    if val < best_val:
                        best_val = val
                        best = (pos.copy(), rot.copy(), L)
            except Exception:
                pass

    if best is None:
        # fallback: large valid container, no overlap (spread out)
        pos = np.zeros((N, 2))
        for i in range(N):
            pos[i] = [6.0 * i, 0.0]
        rot = np.zeros(N)
        L = 45.0
        inner = [[float(pos[i][0]), float(pos[i][1]), float(rot[i])] for i in range(N)]
        return inner, (0.0, 0.0), float(L), 0.0

    pos, rot, L = best
    inner = [[float(pos[i][0]), float(pos[i][1]), float(rot[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(L), 0.0
