"""Rewritten program: barycentric + soft-min L-BFGS-B for Heilbronn 11-point problem."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
T_LIMIT = 110.0  # safety margin before 120s kill

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
VERTS = np.array([V0, V1, V2])

# Area of the reference triangle (for normalization of barycentric area)
_REF_AREA = 0.5 * np.abs(np.cross(V1 - V0, V2 - V0))


def bary_to_cart(bary):
    """bary: (n,3) with rows summing to 1 -> (n,2) cartesian in triangle."""
    bary = np.asarray(bary)
    # map from 2 free barycentric coords (drop third) to cartesian
    # x = v0 + (v1-v0)*a1 + (v2-v0)*a2 ; a3 = 1-a1-a2
    a1 = bary[:, 1]
    a2 = bary[:, 2]
    x = V0[0] + a1 * (V1[0] - V0[0]) + a2 * (V2[0] - V0[0])
    y = V0[1] + a1 * (V1[1] - V0[1]) + a2 * (V2[1] - V0[1])
    return np.column_stack([x, y])


def min_triangle_area_normalized(bary):
    """Return the minimum (normalized by ref area) triangle area among all triples.
    bary: (n,3). Returns scalar in [0,1]."""
    n = bary.shape[0]
    best = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # area of triangle formed by 3 barycentric points, normalized by ref area
                # = |det of matrix with rows bary_i, bary_j, bary_k|
                m = np.array([bary[i], bary[j], bary[k]])
                a = abs(np.linalg.det(m))
                if a < best:
                    best = a
    return best


def soft_min_logsumexp(vals, beta):
    """Smooth approximation of min(vals) using log-sum-exp (negative logsumexp)."""
    vals = np.asarray(vals)
    return -np.log(np.sum(np.exp(-beta * vals))) / beta


def objective_soft(x, beta):
    """x: flattened (n*3) barycentric coords. Returns -soft_min of areas (maximize)."""
    bary = x.reshape(N, 3)
    n = bary.shape[0]
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                m = np.array([bary[i], bary[j], bary[k]])
                areas[idx] = abs(np.linalg.det(m))
                idx += 1
    # maximize the min -> minimize negative soft-min
    sm = soft_min_logsumexp(areas, beta)
    return -sm


def objective_true(x):
    """x: flattened (n*3). Returns -min area (maximize min)."""
    bary = x.reshape(N, 3)
    return -min_triangle_area_normalized(bary)


def random_bary_init(rng):
    """Generate random barycentric points (n,3) with small margins."""
    pts = []
    while len(pts) < N:
        u, v = rng.random(), rng.random()
        # margins to stay strictly inside
        if u + v < 0.96 and u > 0.02 and v > 0.02:
            a1 = u
            a2 = v
            a0 = 1.0 - a1 - a2
            pts.append([a0, a1, a2])
    return np.array(pts)


def polish(bary, deadline):
    """Polish a barycentric configuration using true min-area objective."""
    x0 = bary.reshape(-1)
    res = minimize(
        objective_true,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10, "adaptive": True},
    )
    if res.x is not None:
        cand = res.x.reshape(N, 3)
        # enforce feasibility (non-negative, sum to 1)
        cand = np.clip(cand, 0.0, None)
        cand = cand / cand.sum(axis=1, keepdims=True)
        return cand
    return bary


def construct():
    start = time.time()
    import random
    rng = random.Random(12345)

    best_bary = None
    best_val = -1.0

    # Stage 1: soft-min L-BFGS-B from several random starts
    beta = 8.0
    n_starts = 6
    for s in range(n_starts):
        if time.time() - start > T_LIMIT - 30:
            break
        bary = random_bary_init(rng)
        x0 = bary.reshape(-1)
        try:
            res = minimize(
                objective_soft,
                x0,
                args=(beta,),
                method="L-BFGS-B",
                options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if res.x is not None:
                cand = res.x.reshape(N, 3)
                cand = np.clip(cand, 0.0, None)
                cand = cand / cand.sum(axis=1, keepdims=True)
                val = min_triangle_area_normalized(cand)
                if val > best_val:
                    best_val = val
                    best_bary = cand
        except Exception:
            pass

    # Stage 2: polish the best with the true objective
    if best_bary is not None:
        if time.time() - start < T_LIMIT - 10:
            try:
                best_bary = polish(best_bary, T_LIMIT)
                best_val = min_triangle_area_normalized(best_bary)
            except Exception:
                pass

    # Final: convert to cartesian and return
    if best_bary is None:
        best_bary = random_bary_init(rng)
    cart = bary_to_cart(best_bary)
    return cart


if __name__ == "__main__":
    pts = construct()
    print(pts.shape)
    print("min area:", min_triangle_area_normalized(pts.reshape(N, 3) if pts.shape[0] == N else pts))
