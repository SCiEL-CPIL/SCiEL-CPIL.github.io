"""Hexagon packing: 12 unit hexagons in a minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 150.0
T_START = time.time()


def _time_left():
    return max(0.0, DEADLINE - (time.time() - T_START) - 3.0)


def _corners(x, y, s, a_deg):
    a = math.radians(a_deg)
    c, sn = math.cos(a), math.sin(a)
    pts = []
    for k in range(6):
        th = math.radians(60.0 * k)
        px, py = s * math.cos(th), s * math.sin(th)
        rx, ry = c * px - sn * py, sn * px + c * py
        pts.append((x + rx, y + ry))
    return pts


def _signed_dist(p, poly):
    # poly = list of (x,y) vertices, CCW; returns signed distance (positive inside)
    px, py = p
    n = len(poly)
    best = 1e18
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        dx, dy = px - ax, py - ay
        L2 = ex * ex + ey * ey
        t = (dx * ex + dy * ey) / L2 if L2 > 1e-14 else 0.0
        t = max(0.0, min(1.0, t))
        cx, cy = ax + t * ex, ay + t * ey
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        d = math.sqrt(d2)
        # sign via cross product of edge and (p - a)
        cross = ex * (py - ay) - ey * (px - ax)
        s = d if cross >= 0 else -d
        if s < best:
            best = s
    return best


def _objective(theta):
    xs = theta[0:N]
    ys = theta[N:2 * N]
    angs = theta[2 * N:3 * N]
    cx, cy = theta[3 * N], theta[3 * N + 1]
    S = theta[3 * N + 2]
    adeg = theta[3 * N + 3]
    if S < 0.5:
        return 1e6
    outer = _corners(cx, cy, S, adeg)
    max_viol = 0.0
    for i in range(N):
        ci = _corners(xs[i], ys[i], 1.0, angs[i])
        for (px, py) in ci:
            d = _signed_dist((px, py), outer)
            if d < max_viol:
                max_viol = d
    # soft-max over violations (log-sum-exp)
    m = max_viol
    if m > 0.0:
        logsum = math.log(math.exp(20.0 * m)) + 20.0 * m
        viol = m + logsum / 20.0
    else:
        viol = m
    return S + 5.0 * viol


def _regular_grid():
    # 3-4-3-2 rows centered
    rows = [3, 4, 3, 2]
    row_y = [1.5, 0.0, -1.5, -3.0]
    pts = []
    for r, cnt in enumerate(rows):
        y = row_y[r]
        if r % 2 == 1:
            x0 = -(cnt - 1) / 2.0
        else:
            x0 = -(cnt - 1) / 2.0
        for j in range(cnt):
            x = x0 + j
            pts.append([x, y, 0.0])
    return pts


def _compute_outer(inner):
    # find minimal side S of a regular hexagon (center + rotation) covering all inner corners
    best = (1e9, 0.0, 0.0, 0.0)
    for adeg in range(0, 60, 3):
        a = math.radians(adeg)
        c, sn = math.cos(a), math.sin(a)
        # candidate center = mean of all points
        allpts = []
        for (x, y, aa) in inner:
            allpts.extend(_corners(x, y, 1.0, aa))
        allpts = np.array(allpts)
        cx = float(allpts[:, 0].mean())
        cy = float(allpts[:, 1].mean())
        outer = _corners(cx, cy, 1.0, adeg)
        # expand S until all points inside
        S = 1.0
        while S < 100:
            outer = _corners(cx, cy, S, adeg)
            ok = True
            for (px, py) in allpts:
                if _signed_dist((px, py), outer) < 0:
                    ok = False
                    break
            if ok:
                # try to shrink a bit
                break
            S *= 1.05
        # binary shrink
        lo, hi = 0.5, S
        for _ in range(40):
            mid = (lo + hi) / 2.0
            outer = _corners(cx, cy, mid, adeg)
            ok = True
            for (px, py) in allpts:
                if _signed_dist((px, py), outer) < 0:
                    ok = False
                    break
            if ok:
                hi = mid
            else:
                lo = mid
        S = hi
        if S < best[0]:
            best = (S, cx, cy, adeg)
    return best


def construct():
    # initial: regular grid
    inner = _regular_grid()
    S0, cx0, cy0, a0 = _compute_outer(inner)
    x0 = []
    for (x, y, a) in inner:
        x0 += [x, y, a]
    x0 += [cx0, cy0, S0, a0]
    x0 = np.array(x0, dtype=float)

    best = list(x0)
    best_val = _objective(best)
    if best_val < 0:
        pass

    t_end = _time_left()
    if t_end > 5.0:
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-7, 'fatol': 1e-7,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best = list(res.x)
        except Exception:
            pass

    t_end = _time_left()
    if t_end > 4.0:
        try:
            res = minimize(_objective, np.array(best), method='Nelder-Mead',
                           options={'maxiter': 6000, 'xatol': 1e-9, 'fatol': 1e-9,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best = list(res.x)
        except Exception:
            pass

    # build inner from best
    inner_out = []
    for i in range(N):
        inner_out.append([best[0 * N + i], best[1 * N + i], best[2 * N + i]])
    cx, cy = best[3 * N], best[3 * N + 1]
    S = best[3 * N + 2]
    adeg = best[3 * N + 3]
    return inner_out, (cx, cy), S, adeg


if __name__ == '__main__':
    inner, (cx, cy), S, a = construct()
    print(S)
