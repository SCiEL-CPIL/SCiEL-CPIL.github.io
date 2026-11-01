"""Rewritten circle-packing program for n=32 in the unit square."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 32

# EVOLVE-BLOCK-START
def _soft_penalty(v, k=80.0):
    """Smooth one-sided penalty: 0 for v<=0, ~v for large v>0."""
    return (np.log1p(np.exp(k * v)) - np.log(2.0)) / k

def _objective_grad(z, contact_pairs, k=80.0):
    """
    Objective: minimize -sum(r) + boundary penalties + contact penalties.
    z = [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Returns (value, gradient).
    """
    n = N
    pos = z[:2 * n].reshape(n, 2)
    radii = z[2 * n:]

    val = -np.sum(radii)
    grad = np.zeros_like(z)

    # --- Boundary terms (smooth) ---
    # For each circle i:
    #   left:   x_i - r_i >= 0  -> penalty(v) with v = r_i - x_i
    #   right:  x_i + r_i <= 1  -> v = x_i + r_i - 1
    #   bottom: y_i - r_i >= 0  -> v = r_i - y_i
    #   top:    y_i + r_i <= 1  -> v = y_i + r_i - 1
    for i in range(n):
        x, y, r = pos[i, 0], pos[i, 1], radii[i]
        # left
        v = r - x
        p = _soft_penalty(v, k)
        val += p
        dp = k * (1.0 / (1.0 + np.exp(-k * v)))
        grad[2 * n + i] += dp          # dr
        grad[2 * i]     -= dp          # dx

        # right
        v = x + r - 1.0
        p = _soft_penalty(v, k)
        val += p
        dp = k * (1.0 / (1.0 + np.exp(-k * v)))
        grad[2 * n + i] += dp
        grad[2 * i]     += dp

        # bottom
        v = r - y
        p = _soft_penalty(v, k)
        val += p
        dp = k * (1.0 / (1.0 + np.exp(-k * v)))
        grad[2 * n + i] += dp
        grad[2 * i + 1] -= dp

        # top
        v = y + r - 1.0
        p = _soft_penalty(v, k)
        val += p
        dp = k * (1.0 / (1.0 + np.exp(-k * v)))
        grad[2 * n + i] += dp
        grad[2 * i + 1] += dp

    # --- Contact penalties (smooth) on Delaunay edges ---
    for (i, j) in contact_pairs:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        d2 = dx * dx + dy * dy
        d = np.sqrt(max(d2, 1e-20))
        v = radii[i] + radii[j] - d
        p = _soft_penalty(v, k)
        val += p
        dp = k * (1.0 / (1.0 + np.exp(-k * v)))
        if d > 1e-12:
            inv_d = 1.0 / d
            # grad w.r.t. x_i, y_i
            grad[2 * i]     += dp * dx * inv_d
            grad[2 * i + 1] += dp * dy * inv_d
            # grad w.r.t. x_j, y_j
            grad[2 * j]     -= dp * dx * inv_d
            grad[2 * j + 1] -= dp * dy * inv_d
        # grad w.r.t. radii
        grad[2 * n + i] += dp
        grad[2 * n + j] += dp

    return val, grad

def _hex_init():
    """
    Generate a hexagonal close-packing arrangement of N circles in the unit square.
    Rows offset by half a spacing.
    """
    # Try to fit N circles in a hexagonal lattice
    # Approximate: for n circles in hex packing, ~sqrt(n) rows
    n_rows = int(np.ceil(np.sqrt(n * 2.0 / np.sqrt(3.0))))
    pts = []
    r_base = 0.01
    # Start with a very small radius and let the optimizer grow them
    for row in range(n_rows):
        n_in_row = int(np.floor(np.sqrt(n * 3.0 / (2.0 * row + 1)))) if row > 0 else int(np.ceil(np.sqrt(n)))
        n_in_row = max(1, min(n_in_row, n - len(pts)))
        # x positions
        dx = 1.0 / (n_in_row + 1)
        for col in range(n_in_row):
            x = (col + 1) * dx
            y = (row + 0.5) / (n_rows + 1)
            # Hex offset
            if row % 2 == 1 and n_in_row > 1:
                x += dx * 0.5
            if len(pts) < n:
                pts.append([x, y, r_base])
    # If we didn't get enough, add remaining in a grid
    while len(pts) < n:
        idx = len(pts)
        side = int(np.ceil(np.sqrt(idx + 1)))
        x = (idx % side + 0.5) / side
        y = (idx // side + 0.5) / side
        pts.append([x, y, r_base])
    return np.array(pts[:n])

def _delaunay_pairs(pos):
    """
    Compute Delaunay edges from positions.
    Returns list of (i, j) pairs.
    """
    try:
        tri = Delaunay(pos)
        pairs = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    pairs.add((min(i, j), max(i, j)))
        return list(pairs)
    except Exception:
        # Fallback: all pairs (dense but safe)
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _validate(arr):
    """Check validity: inside unit square, non-overlapping, non-negative radii."""
    n = len(arr)
    for i in range(n):
        x, y, r = arr[i]
        if r < -1e-10:
            return False
        if x - r < -1e-7 or x + r > 1 + 1e-7 or y - r < -1e-7 or y + r > 1 + 1e-7:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = arr[i]
            xj, yj, rj = arr[j]
            dist = np.hypot(xi - xj, yi - yj)
            if dist < ri + rj - 1e-6:
                return False
    return True

def _fallback_grid(n=32):
    """Fallback: uniform grid of small circles."""
    side = 6
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.005
            cells.append([x, y, r])
    return np.array(cells[:n])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 150s total, leave 40s margin

    best_arr = _hex_init()
    best_score = np.sum(best_arr[:, 2])

    if not _validate(best_arr):
        best_arr = _fallback_grid()
        best_score = np.sum(best_arr[:, 2])

    n_restarts = 0
    while time.time() < deadline - 5.0:
        n_restarts += 1

        # Initialize from best so far with perturbation
        init = best_arr.copy()
        for i in range(N):
            init[i, 0] += random.uniform(-0.01, 0.01)
            init[i, 1] += random.uniform(-0.01, 0.01)
            init[i, 2] = max(0.005, init[i, 2] * random.uniform(0.9, 1.1))
        # Clip positions to stay near [0,1]
        init[:, 0] = np.clip(init[:, 0], 0.0, 1.0)
        init[:, 1] = np.clip(init[:, 1], 0.0, 1.0)
        init[:, 2] = np.clip(init[:, 2], 0.005, 0.5)

        # Build contact graph from current positions
        contact_pairs = _delaunay_pairs(init[:, :2])

        # Vector for L-BFGS-B: [x, y, r]
        z0 = np.concatenate([init[:, 0], init[:, 1], init[:, 2]])

        bounds = []
        for _ in range(2 * N):
            bounds.append((0.0, 1.0))
        for _ in range(N):
            bounds.append((0.001, 0.5))

        def obj(z):
            val, _ = _objective_grad(z, contact_pairs)
            return val

        def grad(z):
            _, g = _objective_grad(z, contact_pairs)
            return g

        try:
            res = minimize(obj, z0, jac=grad, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-10,
                                    'gtol': 1e-7, 'maxls': 50})
        except Exception:
            continue

        # Reconstruct
        pos = res.x[:2 * N].reshape(N, 2)
        radii = res.x[2 * N:]
        cand = np.column_stack([pos, radii])

        # Ensure radii non-negative
        cand[:, 2] = np.maximum(cand[:, 2], 0.0)

        if _validate(cand):
            score = np.sum(cand[:, 2])
            if score > best_score:
                best_score = score
                best_arr = cand

        # --- Polish pass: recompute Delaunay on improved positions and re-optimize ---
        if time.time() < deadline - 3.0:
            contact_pairs2 = _delaunay_pairs(best_arr[:, :2])
            z0b = np.concatenate([best_arr[:, 0], best_arr[:, 1], best_arr[:, 2]])

            def obj2(z):
                val, _ = _objective_grad(z, contact_pairs2)
                return val

            def grad2(z):
                _, g = _objective_grad(z, contact_pairs2)
                return g

            try:
                res2 = minimize(obj2, z0b, jac=grad2, method='L-BFGS-B',
                                bounds=bounds,
                                options={'maxiter': 300, 'ftol': 1e-12,
                                         'gtol': 1e-8})
                pos2 = res2.x[:2 * N].reshape(N, 2)
                radii2 = res2.x[2 * N:]
                cand2 = np.column_stack([pos2, radii2])
                cand2[:, 2] = np.maximum(cand2[:, 2], 0.0)
                if _validate(cand2):
                    score2 = np.sum(cand2[:, 2])
                    if score2 > best_score:
                        best_score = score2
                        best_arr = cand2
            except Exception:
                pass

    # Final safety
    if not _validate(best_arr):
        best_arr = _fallback_grid()

    return best_arr
# EVOLVE-BLOCK-END
