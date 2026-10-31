"""Heilbronn triangles: 11 points in unit-area equilateral triangle."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
# Equilateral triangle vertices (unit area)
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3.0) / 2.0])
VERTS = np.array([V0, V1, V2])

# Barycentric coordinate transform: (b0,b1,b2) with sum=1, bi>=0
# We use 2 params per point (b1, b2), b0 = 1 - b1 - b2
# Point = b1*V1 + b2*V2  (since V0 is origin)


def bary_to_cart(b):
    """b: (n,2) array of (b1, b2). Returns (n,2) cartesian."""
    return b[:, 0:1] * V1 + b[:, 1:2] * V2


def min_area(points):
    """Smallest area of any triangle formed by 3 points."""
    n = len(points)
    best = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if area < best:
                    best = area
    return best


def soft_min_area(b, beta=50.0):
    """Smooth surrogate: -log-sum-exp(-beta * areas) approximates min area."""
    pts = bary_to_cart(b)
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                bb = pts[j]
                c = pts[k]
                area = 0.5 * abs((bb[0] - a[0]) * (c[1] - a[1]) - (bb[1] - a[1]) * (c[0] - a[0]))
                areas.append(area)
    areas = np.array(areas)
    # log-sum-exp trick
    m = areas.min()
    return m + np.log(np.sum(np.exp(-(areas - m) * beta))) / beta


def soft_min_neg(b, beta=50.0):
    return -soft_min_area(b, beta)


def construct():
    start = time.time()
    deadline = 135.0  # safety margin before 150s kill

    # Bounded box for barycentric params (b1, b2) in [0,1]
    # Constraint: b1 + b2 <= 1 (b0 >= 0). We'll handle via penalty or clip.
    bounds = [(0.0, 1.0)] * (2 * N)

    def penalty(b):
        """Penalty for b1+b2 > 1 (outside triangle)."""
        pts = b
        over = 0.0
        for i in range(N):
            s = pts[2 * i] + pts[2 * i + 1]
            if s > 1.0:
                over += (s - 1.0) ** 2
        return over

    def objective_de(b):
        """Objective for DE: maximize min area (minimize negative) + penalty."""
        # Clip to feasible region for evaluation
        bc = b.copy()
        for i in range(N):
            s = bc[2 * i] + bc[2 * i + 1]
            if s > 1.0:
                scale = 1.0 / s
                bc[2 * i] *= scale
                bc[2 * i + 1] *= scale
        val = soft_min_area(bc, beta=30.0)
        return -val + 100.0 * penalty(bc)

    # Phase 1: Differential Evolution (global)
    best_b = None
    best_val = -np.inf
    try:
        res = differential_evolution(
            objective_de,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-4,
            mutation=(0.4, 1.0),
            recombination=0.9,
            polish=False,
            init='sobol',
            updating='deferred',
            seed=42,
            workers=1,
        )
        if res.success or res.fun < 1e6:
            best_b = res.x
            # Evaluate actual soft-min
            best_val = soft_min_area(best_b, beta=50.0)
    except Exception:
        pass

    if best_b is None:
        # Fallback: random feasible points
        best_b = np.random.RandomState(7).rand(2 * N)
        for i in range(N):
            s = best_b[2 * i] + best_b[2 * i + 1]
            if s > 1.0:
                best_b[2 * i] /= s
                best_b[2 * i + 1] /= s

    # Phase 2: SLSQP with smooth soft-min surrogate (higher beta = sharper)
    elapsed = time.time() - start
    if elapsed < deadline - 30:
        try:
            for beta in [50.0, 100.0, 200.0]:
                if time.time() - start > deadline - 15:
                    break
                res2 = minimize(
                    soft_min_neg,
                    best_b,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (2 * N),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
                )
                # Clip to feasible
                cb = res2.x.copy()
                for i in range(N):
                    s = cb[2 * i] + cb[2 * i + 1]
                    if s > 1.0:
                        cb[2 * i] /= s
                        cb[2 * i + 1] /= s
                v = soft_min_area(cb, beta=200.0)
                if v > best_val:
                    best_val = v
                    best_b = cb
        except Exception:
            pass

    # Phase 3: Final polish with hard min (exact objective) via L-BFGS-B
    elapsed = time.time() - start
    if elapsed < deadline - 10:
        def hard_min_neg(b):
            pts = bary_to_cart(b)
            return -min_area(pts)

        try:
            res3 = minimize(
                hard_min_neg,
                best_b,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            cb = res3.x.copy()
            for i in range(N):
                s = cb[2 * i] + cb[2 * i + 1]
                if s > 1.0:
                    cb[2 * i] /= s
                    cb[2 * i + 1] /= s
            v = min_area(bary_to_cart(cb))
            pts_final = bary_to_cart(best_b)
            v_orig = min_area(pts_final)
            if v > v_orig:
                best_b = cb
        except Exception:
            pass

    # Convert to cartesian
    points = bary_to_cart(best_b)

    # Ensure points are inside triangle (clip tiny numerical errors)
    for i in range(N):
        b1 = best_b[2 * i]
        b2 = best_b[2 * i + 1]
        s = b1 + b2
        if s > 1.0:
            best_b[2 * i] /= s
            best_b[2 * i + 1] /= s
    points = bary_to_cart(best_b)

    return points
