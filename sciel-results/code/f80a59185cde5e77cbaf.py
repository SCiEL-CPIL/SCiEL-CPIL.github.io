"""Circle packing in perimeter-4 rectangle, 21 circles.
Structural model: Delaunay contact-graph tension + radii-only (SDP-like) solve,
refined by softplus-penalty gradient polish with multi-phase beta + SA aspect ratio.
"""
import numpy as np
import time
import random
from scipy.spatial import Delaunay

N = 21
DEADLINE = 150.0 - 8.0  # wall-clock safety margin


# ---------------- softplus machinery (from inspiration) ----------------
def _softplus(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0, x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)


def _softplus_grad(x, beta=100.0):
    return 1.0 / (1.0 + np.exp(-beta * x))


def _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.05):
    c = x.reshape(-1, 3)
    xs, ys, rs = c[:, 0], c[:, 1], c[:, 2]
    obj = -np.sum(rs)
    grad = np.zeros_like(x)
    if eq_weight > 0:
        r_mean = np.mean(rs)
        obj += eq_weight * np.mean((rs - r_mean) ** 2)
        grad[2::3] += eq_weight * 2.0 * (rs - r_mean) / N
    v = rs - xs; obj += np.sum(_softplus(v, beta)); g = _softplus_grad(v, beta)
    grad[0::3] -= g; grad[2::3] += g
    v = xs + rs - w; obj += np.sum(_softplus(v, beta)); g = _softplus_grad(v, beta)
    grad[0::3] += g; grad[2::3] += g
    v = rs - ys; obj += np.sum(_softplus(v, beta)); g = _softplus_grad(v, beta)
    grad[1::3] -= g; grad[2::3] += g
    v = ys + rs - h; obj += np.sum(_softplus(v, beta)); g = _softplus_grad(v, beta)
    grad[1::3] += g; grad[2::3] += g
    v = -rs; obj += np.sum(_softplus(v, beta)); g = _softplus_grad(v, beta)
    grad[2::3] -= g
    for i in range(N):
        dx = xs[i] - xs[i + 1:]; dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        obj += np.sum(_softplus(v, beta)); g = _softplus_grad(v, beta)
        grad[3 * i] += -np.sum(g * dx / d)
        grad[3 * i + 1] += -np.sum(g * dy / d)
        grad[3 * i + 2] += np.sum(g)
        idx = np.arange(i + 1, N)
        grad[3 * idx] += g * dx / d
        grad[3 * idx + 1] += g * dy / d
        grad[3 * idx + 2] += g
    grad[2::3] += -1.0
    return obj, grad


def _strictly_valid(circles, w, h, tol=1e-8):
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(N):
        dx = xs[i] - xs[i + 1:]; dy = ys[i] - ys[i + 1:]
        d2 = dx * dx + dy * dy
        need = rs[i] + rs[i + 1:] - tol
        if np.any(d2 < need * need):
            return False
    return True


def _delaunay_neighbors(centers):
    try:
        tri = Delaunay(centers)
        nbrs = [set() for _ in range(N)]
        for i, j, k in tri.simplices:
            nbrs[i].add(j); nbrs[i].add(k)
            nbrs[j].add(i); nbrs[j].add(k)
            nbrs[k].add(i); nbrs[k].add(j)
        return [list(s) for s in nbrs]
    except Exception:
        return [list(range(N)) for _ in range(N)]


# ---------------- structural model: tension graph + radii solve ----------------
def _build_tension_edges(circles, w, h):
    """Contact-graph tension edges: Delaunay pairs plus wall pairs.
    Returns list of ('pair', i, j) and ('wall', i, side)."""
    nbrs = _delaunay_neighbors(circles[:, :2])
    seen = set()
    edges = []
    for i in range(N):
        for j in nbrs[i]:
            if j <= i:
                continue
            edges.append((i, j))
            seen.add((i, j))
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    for i in range(N):
        if xs[i] - rs[i] < 1e-3:
            edges.append(('L', i))
        if xs[i] + rs[i] > w - 1e-3:
            edges.append(('R', i))
        if ys[i] - rs[i] < 1e-3:
            edges.append(('B', i))
        if ys[i] + rs[i] > h - 1e-3:
            edges.append(('T', i))
    return edges


def _solve_radii(circles, w, h, edges, iters=200, tol=1e-11):
    """Radii-only solve: maximize sum(r) s.t. d_ij >= r_i+r_j and wall constraints.
    Newton/SDP-style: r_i += sum_j lambda_ij (d_ij - r_i - r_j), mu for walls."""
    xs, ys = circles[:, 0], circles[:, 1]
    r = circles[:, 2].copy()
    # edge lists
    pair = [(i, j) for (i, j) in edges if isinstance(i, int)]
    wall = [(s, i) for (s, i) in edges if isinstance(s, str)]
    # adjacency for pair
    adj = [[] for _ in range(N)]
    for (i, j) in pair:
        adj[i].append(j); adj[j].append(i)
    for _ in range(iters):
        r_new = r.copy()
        # pair updates
        for (i, j) in pair:
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            slack = d - r[i] - r[j]
            if slack < 0:
                r_new[i] += 0.5 * (-slack) * 0.5
                r_new[j] += 0.5 * (-slack) * 0.5
        # wall updates (circle can't exceed wall distance)
        for (s, i) in wall:
            if s == 'L':
                d = xs[i]
            elif s == 'R':
                d = w - xs[i]
            elif s == 'B':
                d = ys[i]
            else:
                d = h - ys[i]
            if r[i] > d:
                r_new[i] = d
        # enforce non-negative
        r_new = np.maximum(r_new, 1e-9)
        # project back: if any pair over-satisfied, shrink
        changed = False
        for (i, j) in pair:
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            if r_new[i] + r_new[j] > d + 1e-9:
                # scale down both to fit
                excess = r_new[i] + r_new[j] - d
                r_new[i] -= 0.5 * excess
                r_new[j] -= 0.5 * excess
                changed = True
        if not changed:
            # check convergence
            if np.max(np.abs(r_new - r)) < tol:
                break
        r = r_new
    # final clamp
    r = np.maximum(r, 1e-9)
    for (i, j) in pair:
        dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        if r[i] + r[j] > d:
            excess = r[i] + r[j] - d
            r[i] -= 0.5 * excess
            r[j] -= 0.5 * excess
    for (s, i) in wall:
        if s == 'L': d = xs[i]
        elif s == 'R': d = w - xs[i]
        elif s == 'B': d = ys[i]
        else: d = h - ys[i]
        r[i] = min(r[i], d)
    r = np.maximum(r, 1e-9)
    out = circles.copy()
    out[:, 2] = r
    return out


def _relax_centers(circles, w, h, iters=300, dt=0.002):
    """Delaunay-nearest-neighbor spring relaxation: push centers apart to create
    contacts, keep inside walls."""
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    for _ in range(iters):
        nbrs = _delaunay_neighbors(np.column_stack([xs, ys]))
        fx = np.zeros(N); fy = np.zeros(N)
        for i in range(N):
            for j in nbrs[i]:
                if j <= i:
                    continue
                dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy + 1e-12)
                target = rs[i] + rs[j] + 0.001
                if d < target:
                    f = (target - d) / max(d, 1e-9)
                    fx[i] += f * dx; fy[i] += f * dy
                    fx[j] -= f * dx; fy[j] -= f * dy
        # wall repulsion
        for i in range(N):
            if xs[i] - rs[i] < 0.01: fx[i] += (0.01 - (xs[i] - rs[i]))
            if xs[i] + rs[i] > w - 0.01: fx[i] -= (xs[i] + rs[i] - (w - 0.01))
            if ys[i] - rs[i] < 0.01: fy[i] += (0.01 - (ys[i] - rs[i]))
            if ys[i] + rs[i] > h - 0.01: fy[i] -= (ys[i] + rs[i] - (h - 0.01))
        xs = np.clip(xs + dt * fx, rs, w - rs)
        ys = np.clip(ys + dt * fy, rs, h - rs)
    out = circles.copy()
    out[:, 0] = xs; out[:, 1] = ys
    return out


# ---------------- init generators (from inspiration) ----------------
def _hex_init(w, h, seed):
    rng = random.Random(seed)
    row_configs = [
        [3, 4, 4, 4, 3, 3], [4, 4, 4, 4, 3, 2], [2, 4, 4, 4, 4, 3],
        [3, 4, 4, 4, 4, 2], [4, 3, 4, 4, 4, 2], [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 3], [4, 4, 3, 4, 4, 2], [2, 4, 3, 4, 4, 4],
        [3, 4, 3, 4, 4, 3], [5, 4, 4, 4, 4], [4, 5, 4, 4, 4],
        [3, 5, 4, 4, 5], [4, 4, 5, 4, 4], [2, 5, 4, 5, 5],
        [5, 5, 5, 5, 1], [1, 5, 5, 5, 5], [3, 4, 5, 4, 5], [5, 4, 5, 4, 3],
    ]
    best = None; best_val = -np.inf
    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts); max_cols = max(row_counts)
        est_r_h = w / (max_cols + 0.5)
        est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = (row_idx + 0.5) * r_est * np.sqrt(3)
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        circles = np.array(circles)
        if _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val; best = circles.copy()
    if best is None:
        a = int(np.ceil(np.sqrt(N * w / h))); b = int(np.ceil(N / a))
        while a * b < N: b += 1
        circles = []
        for i in range(N):
            col = i % a; row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            circles.append([(col + 0.5) * w / a, (row + 0.5) * h / b, r])
        best = np.array(circles)
    return best


def _grid_init(w, h, seed):
    a = int(np.ceil(np.sqrt(N * w / h))); b = int(np.ceil(N / a))
    while a * b < N: b += 1
    circles = []
    for i in range(N):
        col = i % a; row = i // a
        r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
        circles.append([(col + 0.5) * w / a, (row + 0.5) * h / b, r])
    return np.array(circles)


def _random_init(w, h, seed):
    rng = np.random.RandomState(seed)
    r0 = min(w, h) / 2.0 * 0.25
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0
    return circles


def _improve(circles, w, h, time_budget):
    """Full pipeline: structural relax + radii solve, then softplus polish."""
    start = time.time()
    # 1. structural: relax centers to create contacts
    circles = _relax_centers(circles, w, h, iters=200)
    # 2. structural: solve radii from contact graph
    edges = _build_tension_edges(circles, w, h)
    circles = _solve_radii(circles, w, h, edges, iters=150)
    # 3. polish: softplus gradient descent with increasing beta
    x = circles.reshape(-1).copy()
    for beta in [50.0, 100.0, 200.0, 400.0]:
        for step in range(200):
            if time.time() - start > time_budget:
                break
            obj, grad = _objective_and_grad(x, w, h, beta=beta, eq_weight=0.03)
            # line search
            lr = 0.01
            for _ in range(30):
                x_trial = x - lr * grad
                obj_t, _ = _objective_and_grad(x_trial, w, h, beta=beta, eq_weight=0.03)
                if obj_t < obj:
                    break
                lr *= 0.5
            x = x - lr * grad
        # re-solve radii from updated contact graph
        c = x.reshape(-1, 3)
        edges = _build_tension_edges(c, w, h)
        c = _solve_radii(c, w, h, edges, iters=100)
        x = c.reshape(-1)
    return x.reshape(-1, 3)


def _sa_aspect_ratio(best_circles, best_val, time_budget):
    """Simulated annealing over aspect ratio w (h = 2 - w)."""
    start = time.time()
    w = 1.0
    best_c = best_circles.copy()
    best_v = best_val
    T = 0.1
    for _ in range(200):
        if time.time() - start > time_budget:
            break
        w_new = np.clip(w + random.gauss(0, 0.05), 0.3, 1.7)
        h_new = 2.0 - w_new
        # re-init at new aspect
        c = _hex_init(w_new, h_new, seed=int(w_new * 1000))
        c = _improve(c, w_new, h_new, time_budget - (time.time() - start))
        if _strictly_valid(c, w_new, h_new):
            v = float(np.sum(c[:, 2]))
            if v > best_v or random.random() < np.exp((v - best_v) / max(T, 1e-6)):
                best_v = v; best_c = c.copy(); w = w_new
        T *= 0.97
    return best_c, best_v


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    best_result = None
    best_val = 0.0

    # Try multiple aspect ratios and seeds
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.9, 1.1, 1.3]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for ar, seed in zip(aspect_ratios, seeds):
        if time.time() > deadline - 15:
            break
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w
        # time per attempt
        remaining = deadline - time.time()
        n_left = len(aspect_ratios) - aspect_ratios.index(ar)
        time_per = min(8.0, max(2.0, remaining / max(1, n_left) - 2.0))

        for init_fn in [_hex_init, _grid_init, _random_init]:
            if time.time() > deadline - 10:
                break
            try:
                c = init_fn(w, h, seed)
                c = _improve(c, w, h, time_per / 3.0)
                if _strictly_valid(c, w, h):
                    v = float(np.sum(c[:, 2]))
                    if v > best_val:
                        best_val = v
                        best_result = c.copy()
            except Exception:
                continue

        # SA aspect ratio refinement from best so far
        if time.time() < deadline - 20 and best_result is not None:
            try:
                c2, v2 = _sa_aspect_ratio(best_result, best_val,
                                          min(15.0, deadline - time.time() - 10))
                if _strictly_valid(c2, 2.0 - c2.shape[0] and 2.0 - (2.0 * 1.0 / (1.0 + 1.0)), 1.0) or True:
                    # validate with proper w,h
                    pass
            except Exception:
                pass

    if best_result is None:
        # Fallback grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final validation
    min_x = np.min(best_result[:, 0] - best_result[:, 2])
    max_x = np.max(best_result[:, 0] + best_result[:, 2])
    min_y = np.min(best_result[:, 1] - best_result[:, 2])
    max_y = np.max(best_result[:, 1] + best_result[:, 2])
    w_box = max_x - min_x
    h_box = max_y - min_y
    if 2 * (w_box + h_box) > 4.0 + 1e-6:
        # scale down
        scale = 2.0 / (w_box + h_box) * 0.999
        best_result = best_result * scale

    return best_result
