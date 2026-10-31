"""Circle packing in perimeter-4 rectangle: contact-graph-guided SLSQP + multi-start."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 21
START = time.time()
DEADLINE = START + 135.0  # safety margin before 150s kill


def _rect_from_width(W):
    H = 2.0 - W
    return W, H


def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-8 < r[i] + r[j]:
                return None
    return float(np.sum(r))


def _build_contact_graph(x, W, H, threshold=1.05):
    """
    Build a contact graph from a layout.
    Returns adjacency matrix (n x n) where A[i,j]=1 means circles i,j are in contact.
    Contact = distance <= threshold * (r_i + r_j)
    """
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d <= threshold * (r[i] + r[j]):
                A[i, j] = 1
                A[j, i] = 1
    return A


def _perturb_contact_graph(A, n, n_perturb=3, rng=None):
    """
    Perturb a contact graph by adding/removing a few edges.
    This explores nearby topologies.
    """
    if rng is None:
        rng = random
    A_new = A.copy()
    # Remove some edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A_new[i, j]]
    for _ in range(n_perturb):
        if edges:
            i, j = rng.choice(edges)
            A_new[i, j] = 0
            A_new[j, i] = 0
            edges.remove((i, j))
    # Add some edges (prefer non-adjacent or distant pairs)
    non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not A_new[i, j]]
    for _ in range(n_perturb):
        if non_edges:
            i, j = rng.choice(non_edges)
            A_new[i, j] = 1
            A_new[j, i] = 1
            non_edges.remove((i, j))
    return A_new


def _objective_contact(x, W, H, A, mu=1e3, mu_contact=5e3):
    """
    Objective with contact-graph awareness.
    - Standard penalty for boundary/overlap violations
    - Additional penalty to ENFORCE contact edges: push touching pairs to exactly r_i+r_j
    - Anti-contact penalty for non-edges: keep non-contact pairs separated
    """
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)

    cx = np.clip(cx, -1.0, W + 1.0)
    cy = np.clip(cy, -1.0, H + 1.0)
    r = np.clip(r, -1.0, 2.0)

    val = -np.sum(r)
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0

    # Boundary penalty: r <= min(cx, cy, W-cx, H-cy)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    viol_b = np.maximum(r - lo, 0.0)
    val += mu * np.sum(viol_b ** 2)
    for i in range(n):
        if viol_b[i] <= 0:
            continue
        d_left = cx[i]
        d_bottom = cy[i]
        d_right = W - cx[i]
        d_top = H - cy[i]
        min_d = lo[i]
        g_cx = 0.0
        g_cy = 0.0
        eps = 1e-9
        cnt = 0
        if d_left <= min_d + eps:
            g_cx += 1.0
            cnt += 1
        if d_right <= min_d + eps:
            g_cx += -1.0
            cnt += 1
        if d_bottom <= min_d + eps:
            g_cy += 1.0
            cnt += 1
        if d_top <= min_d + eps:
            g_cy += -1.0
            cnt += 1
        if cnt > 0:
            g_cx /= cnt
            g_cy /= cnt
        dv = 2.0 * mu * viol_b[i]
        grad[i, 0] += dv * (-g_cx)
        grad[i, 1] += dv * (-g_cy)
        grad[i, 2] += dv * 1.0

    # r > 0
    viol_r = np.maximum(-r, 0.0)
    val += mu * np.sum(viol_r ** 2)
    for i in range(n):
        if viol_r[i] > 0:
            grad[i, 2] += 2.0 * mu * viol_r[i] * (-1.0)

    # Centers in box
    mu_c = mu * 0.1
    viol_cx_lo = np.maximum(-cx, 0.0)
    viol_cx_hi = np.maximum(cx - W, 0.0)
    viol_cy_lo = np.maximum(-cy, 0.0)
    viol_cy_hi = np.maximum(cy - H, 0.0)
    val += mu_c * (np.sum(viol_cx_lo**2) + np.sum(viol_cx_hi**2) +
                   np.sum(viol_cy_lo**2) + np.sum(viol_cy_hi**2))
    for i in range(n):
        if viol_cx_lo[i] > 0:
            grad[i, 0] += 2.0 * mu_c * viol_cx_lo[i] * (-1.0)
        if viol_cx_hi[i] > 0:
            grad[i, 0] += 2.0 * mu_c * viol_cx_hi[i] * 1.0
        if viol_cy_lo[i] > 0:
            grad[i, 1] += 2.0 * mu_c * viol_cy_lo[i] * (-1.0)
        if viol_cy_hi[i] > 0:
            grad[i, 1] += 2.0 * mu_c * viol_cy_hi[i] * 1.0

    # Non-overlap penalty for ALL pairs
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            req = r[i] + r[j]
            viol = req - d
            if viol <= 0:
                continue
            val += mu * viol ** 2
            if d < 1e-12:
                grad[i, 0] += 2.0 * mu * viol * 1.0
                grad[j, 0] += 2.0 * mu * viol * (-1.0)
                grad[i, 2] += 2.0 * mu * viol * 1.0
                grad[j, 2] += 2.0 * mu * viol * 1.0
                continue
            dv = 2.0 * mu * viol
            grad[i, 0] += dv * (-dx / d)
            grad[i, 1] += dv * (-dy / d)
            grad[j, 0] += dv * (dx / d)
            grad[j, 1] += dv * (dy / d)
            grad[i, 2] += dv * 1.0
            grad[j, 2] += dv * 1.0

    # Contact enforcement: for edges in A, push distance toward r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                continue
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            target = r[i] + r[j]
            # We want d ≈ target. Penalty = (d - target)^2
            diff = d - target
            if abs(diff) < 1e-10:
                continue
            val += mu_contact * diff ** 2
            if d < 1e-12:
                # degenerate
                grad[i, 0] += 2.0 * mu_contact * diff * 1.0
                grad[j, 0] += 2.0 * mu_contact * diff * (-1.0)
                grad[i, 2] += 2.0 * mu_contact * diff * (-1.0)
                grad[j, 2] += 2.0 * mu_contact * diff * (-1.0)
                continue
            # d(diff)/d(cx_i) = dx/d, d(diff)/d(cx_j) = -dx/d
            # d(diff)/d(r_i) = -1, d(diff)/d(r_j) = -1
            dv = 2.0 * mu_contact * diff
            grad[i, 0] += dv * (dx / d)
            grad[i, 1] += dv * (dy / d)
            grad[j, 0] += dv * (-dx / d)
            grad[j, 1] += dv * (-dy / d)
            grad[i, 2] += dv * (-1.0)
            grad[j, 2] += dv * (-1.0)

    return val, grad


def _grid_layout(kx, ky, W, H, shrink=0.999):
    """Build a kx*ky grid layout (pad/truncate to N)."""
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W - xx, H - yy) * shrink
            pts.append([xx, yy, max(r, 1e-4)])
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]


def _random_layout(W, H, seed=None):
    """Random feasible-ish layout."""
    rng = random.Random(seed)
    pts = []
    for _ in range(N):
        cx = rng.uniform(0.05 * W, 0.95 * W)
        cy = rng.uniform(0.05 * H, 0.95 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.3
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)


def _force_precondition(x, W, H, steps=100, lr=0.005):
    """
    Quick force-directed relaxation to get a reasonable starting point.
    Pushes overlapping circles apart and grows radii.
    """
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)

    for step in range(steps):
        fcx = np.zeros(n)
        fcy = np.zeros(n)

        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.hypot(dx, dy)
                req = r[i] + r[j]
                if d < req and d > 1e-12:
                    f = (req - d) / d
                    fcx[i] += f * dx
                    fcy[i] += f * dy
                    fcx[j] -= f * dx
                    fcy[j] -= f * dy
                elif d <= 1e-12:
                    angle = random.uniform(0, 2 * np.pi)
                    fcx[i] += 0.1 * np.cos(angle)
                    fcy[i] += 0.1 * np.sin(angle)
                    fcx[j] -= 0.1 * np.cos(angle)
                    fcy[j] -= 0.1 * np.sin(angle)

        cx += lr * fcx
        cy += lr * fcy
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)

        lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
        r = np.minimum(r * 1.0005 + 0.00005, lo)
        r = np.maximum(r, 1e-6)
        lr *= 0.995

    return np.column_stack([cx, cy, r])


def _optimize_from(x0, W, H, A=None, max_iter=200):
    """
    Run SLSQP with contact-graph-aware objective.
    If A is None, build it from x0.
    """
    best_x = x0.copy()
    best_sum = _evaluate(x0, W, H)
    if best_sum is None:
        best_sum = 0.0

    if A is None:
        A = _build_contact_graph(x0, W, H, threshold=1.02)

    def fun(x):
        v, _ = _objective_contact(x, W, H, A)
        return v

    def grad(x):
        _, g = _objective_contact(x, W, H, A)
        return g

    # First SLSQP pass
    try:
        res = minimize(fun, x0, jac=grad, method='SLSQP',
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': 0})
        if res.x is not None:
            s = _evaluate(res.x, W, H)
            if s is not None and s > best_sum:
                best_sum = s
                best_x = res.x
    except Exception:
        pass

    # Rebuild contact graph from improved solution
    if time.time() < DEADLINE - 8:
        A2 = _build_contact_graph(best_x, W, H, threshold=1.02)

        def fun2(x):
            v, _ = _objective_contact(x, W, H, A2)
            return v

        def grad2(x):
            _, g = _objective_contact(x, W, H, A2)
            return g

        try:
            res2 = minimize(fun2, best_x, jac=grad2, method='SLSQP',
                            options={'maxiter': 150, 'ftol': 1e-12, 'disp': 0})
            if res2.x is not None:
                s = _evaluate(res2.x, W, H)
                if s is not None and s > best_sum:
                    best_sum = s
                    best_x = res2.x
        except Exception:
            pass

    return best_x, best_sum


def _topology_search(x0, W, H, n_topologies=4, max_iter=150, rng=None):
    """
    Try multiple contact topologies by perturbing the base graph.
    Returns the best (x, sum) found.
    """
    if rng is None:
        rng = random

    base_A = _build_contact_graph(x0, W, H, threshold=1.02)
    best_x = x0.copy()
    best_sum = _evaluate(x0, W, H)
    if best_sum is None:
        best_sum = 0.0

    # Try the base topology
    x_opt, s_opt = _optimize_from(x0, W, H, A=base_A, max_iter=max_iter)
    if s_opt > best_sum:
        best_sum = s_opt
        best_x = x_opt

    # Try perturbed topologies
    for _ in range(n_topologies):
        if time.time() > DEADLINE - 10:
            break
        A_pert = _perturb_contact_graph(base_A, N, n_perturb=2, rng=rng)
        x_opt, s_opt = _optimize_from(x0, W, H, A=A_pert, max_iter=max_iter)
        if s_opt > best_sum:
            best_sum = s_opt
            best_x = x_opt

    return best_x, best_sum


def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0

    def time_ok():
        return time.time() < DEADLINE

    random.seed(42)
    np.random.seed(42)

    candidates = []

    # Aspect ratios to try
    Ws = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    grid_sizes = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5), (5, 4), (4, 5), (6, 3), (3, 6)]

    for W in Ws:
        H = 2.0 - W
        for kx, ky in grid_sizes:
            if kx * ky < N:
                continue
            g = _grid_layout(kx, ky, W, H, shrink=0.98)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))

    # Add random starts
    for W in Ws:
        H = 2.0 - W
        for seed in [1, 2, 3]:
            g = _random_layout(W, H, seed=seed)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))

    # Sort by initial score, best first
    candidates.sort(key=lambda t: -t[0])

    # Optimize top candidates with topology search
    top_k = min(12, len(candidates))
    for idx in range(top_k):
        if not time_ok():
            break
        s0, W, H, g0 = candidates[idx]
        # Force pre-conditioning
        if time.time() < DEADLINE - 15:
            g0 = _force_precondition(g0, W, H, steps=80, lr=0.005)
            s0 = _evaluate(g0, W, H)
            if s0 is None:
                s0 = 0.0

        x_opt, s_opt = _topology_search(g0, W, H, n_topologies=3, max_iter=120)
        if s_opt > best_sum:
            best_sum = s_opt
            best_arr = x_opt
            best_W = W
            best_H = H

    # If we still have time, do a few more random restarts from the best found
    if time_ok() and best_arr is not None:
        remaining = DEADLINE - time.time()
        n_extra = min(5, int(remaining // 12))
        for _ in range(n_extra):
            if not time_ok():
                break
            # Perturb best
            x0 = best_arr + np.random.randn(*best_arr.shape) * 0.02
            x0 = np.clip(x0, 0.0, None)
            x0[:, 0] = np.clip(x0[:, 0], 0.0, best_W)
            x0[:, 1] = np.clip(x0[:, 1], 0.0, best_H)
            x0[:, 2] = np.maximum(x0[:, 2], 1e-4)
            x_opt, s_opt = _topology_search(x0, best_W, best_H, n_topologies=2, max_iter=100)
            if s_opt > best_sum:
                best_sum = s_opt
                best_arr = x_opt

    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.98)
        best_arr = g
        best_W, best_H = W, H

    # Final clamping to ensure validity
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])

    # Final validation: if invalid, fall back to a known-good grid
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        for s0, W, H, g0 in candidates[:5]:
            cx2 = np.clip(g0[:, 0], 0.0, W)
            cy2 = np.clip(g0[:, 1], 0.0, H)
            r2 = np.maximum(g0[:, 2], 1e-6)
            lo2 = np.minimum(np.minimum(cx2, cy2), np.minimum(W - cx2, H - cy2))
            r2 = np.minimum(r2, lo2)
            r2 = np.maximum(r2, 1e-6)
            out2 = np.column_stack([cx2, cy2, r2])
            if _evaluate(out2, W, H) is not None:
                return out2
        # Ultimate fallback: 5x5 grid in unit square
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return out
