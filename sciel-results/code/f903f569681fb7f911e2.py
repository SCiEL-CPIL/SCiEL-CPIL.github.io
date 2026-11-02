import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY = 5.0
EPS = 1e-9

_HEX = np.array([[1.0, 0.0], [0.5, math.sqrt(3) / 2.0],
                 [-0.5, math.sqrt(3) / 2.0],
                 [-1.0, 0.0], [-0.5, -math.sqrt(3) / 2.0],
                 [0.5, -math.sqrt(3) / 2.0]])

_DIRS = np.array([[0.0, 1.0], [0.5, math.sqrt(3) / 2.0], [1.0, 0.0],
                  [0.0, -1.0], [-0.5, -math.sqrt(3) / 2.0], [-1.0, 0.0]])


def _hex_halfwidths(theta):
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    v = _HEX @ R.T
    return v @ _DIRS.T


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _exact_S(centers, thetas, beta):
    hw = np.array([_hex_halfwidths(th) for th in thetas])
    R = _rot(beta)
    D = 0.0
    for k in range(6):
        n = R[:, k]
        for i in range(N):
            ext = centers[i] @ n + hw[i, k]
            D = max(D, abs(ext))
    return (D + APOTHEM) / APOTHEM


def _overlap(centers, thetas):
    hw = np.array([_hex_halfwidths(th) for th in thetas])
    worst = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = centers[i] - centers[j]
            for k in range(6):
                worst = max(worst, abs(d @ _DIRS[k]) - (hw[i, k] + hw[j, k]))
    return worst


def _init_seeds():
    seeds = []
    axial = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, 0), (-2, 0), (0, 2), (0, -2)]
    pts = np.array([[math.sqrt(3) * (q + r / 2.0), 1.5 * r] for q, r in axial])
    pts = pts - pts.mean(axis=0)
    seeds.append((pts, np.zeros(N)))
    ang = np.linspace(0, 2 * math.pi, N, endpoint=False)
    seeds.append((2.3 * np.column_stack([np.cos(ang), np.sin(ang)]),
                  np.zeros(N)))
    a1 = np.linspace(0, 2 * math.pi, 6, endpoint=False)
    a2 = np.linspace(0, 2 * math.pi, 5, endpoint=False) + 0.3
    r1 = np.full(6, 1.4)
    r2 = np.full(5, 2.6)
    ring = np.column_stack([np.concatenate([r1 * np.cos(a1), r2 * np.cos(a2)]),
                            np.concatenate([r1 * np.sin(a1), r2 * np.sin(a2)])])
    seeds.append((ring, np.zeros(N)))
    return seeds


def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    t0 = time.time()
    best_S = float('inf')
    best_p = None
    best_beta = 0.0

    for seed_pts, seed_th in _init_seeds():
        if time.time() - t0 > TIME_LIMIT - SAFETY:
            break
        cur_p = np.concatenate([seed_pts.ravel(), seed_th])
        cur_S = _exact_S(seed_pts, seed_th, 0.0) * 1.1
        temps = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        for temp in temps:
            if time.time() - t0 > TIME_LIMIT - SAFETY:
                break

            def obj(p, temp=temp, cur_S=cur_S):
                if time.time() - t0 > 6.0:
                    return 1e9
                centers = p[:2 * N].reshape(N, 2)
                thetas = p[2 * N:]
                hw = np.array([_hex_halfwidths(th) for th in thetas])
                g = np.zeros((N * (N - 1) // 2, 6))
                gc = np.zeros(N * 6)
                idx = 0
                for i in range(N):
                    for j in range(i + 1, N):
                        d = centers[i] - centers[j]
                        for k in range(6):
                            g[idx, k] = abs(d @ _DIRS[k]) - (hw[i, k] + hw[j, k])
                        idx += 1
                R = _rot(0.0)
                D = cur_S * APOTHEM
                idx = 0
                for i in range(N):
                    for k in range(6):
                        gc[idx] = abs(centers[i] @ _DIRS[k]) - (D - hw[i, k])
                        idx += 1
                allg = np.concatenate([g.ravel(), gc.ravel()])
                m = np.max(allg)
                if m > 1.0:
                    soft = temp * np.log(np.sum(np.exp(allg / temp)))
                elif m < -30.0:
                    soft = 0.0
                else:
                    soft = temp * np.log(np.sum(np.exp(allg / temp)))
                return cur_S + 10.0 * soft

            try:
                res = minimize(obj, cur_p, method='SLSQP',
                               options={'maxiter': 120, 'ftol': 1e-8})
                if res.x is not None:
                    cur_p = res.x
                    cur_S *= 0.96
            except Exception:
                pass

        centers = cur_p[:2 * N].reshape(N, 2)
        thetas = cur_p[2 * N:]
        if _overlap(centers, thetas) >= -EPS:
            # gradient push-apart
            for _ in range(300):
                if time.time() - t0 > TIME_LIMIT - SAFETY:
                    break
                grad = np.zeros_like(centers)
                hw = np.array([_hex_halfwidths(th) for th in thetas])
                for i in range(N):
                    for j in range(i + 1, N):
                        d = centers[i] - centers[j]
                        for k in range(6):
                            v = abs(d @ _DIRS[k]) - (hw[i, k] + hw[j, k])
                            if v > 0:
                                sign = 1.0 if (d @ _DIRS[k]) >= 0 else -1.0
                                grad[i] += sign * _DIRS[k]
                                grad[j] -= sign * _DIRS[k]
                centers -= 0.02 * grad
                centers = centers - centers.mean(axis=0)

        for beta in [0.0, math.pi / 6, math.pi / 12]:
            S = _exact_S(centers, thetas, beta)
            if _overlap(centers, thetas) < -EPS and S < best_S:
                best_S = S
                best_p = np.concatenate([centers.ravel(), thetas])
                best_beta = beta

    if best_p is None:
        axial = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                 (2, 0), (-2, 0), (0, 2), (0, -2)]
        pts = np.array([[math.sqrt(3) * (q + r / 2.0), 1.5 * r] for q, r in axial])
        pts = pts - pts.mean(axis=0)
        thetas = np.zeros(N)
        best_beta = 0.0
        best_S = _exact_S(pts, thetas, 0.0)
        best_p = np.concatenate([pts.ravel(), thetas])

    centers = best_p[:2 * N].reshape(N, 2)
    thetas = best_p[2 * N:]
    cx, cy = centers.mean(axis=0)
    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0]), float(centers[i, 1]),
                      float(math.degrees(thetas[i]))])
    return inner, [float(cx), float(cy)], float(best_S), float(math.degrees(best_beta))
