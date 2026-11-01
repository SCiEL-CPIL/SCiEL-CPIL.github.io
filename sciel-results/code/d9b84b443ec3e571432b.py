import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _compute_score(pts):
    """Compute the true packing radius."""
    D = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    L = np.sqrt(np.sum(D**2, axis=2))
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    min_wall = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_wall)

def _soft_min_obj_grad(x, beta):
    """
    Negative soft-min of all constraints (pair distances/2 and wall distances).
    Returns (objective_value, gradient) for minimization.
    """
    C = x.reshape(N, 2)
    n = N

    # Pairwise distances
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.sqrt(np.sum(D**2, axis=2) + 1e-30)
    np.fill_diagonal(L, np.inf)

    iu = np.triu_indices(n, k=1)
    L_flat = L[iu]
    L_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    x_pairs = L_safe / 2.0

    # Wall distances
    r_pts = np.hypot(C[:, 0], C[:, 1])
    x_walls = 1.0 - r_pts

    x_all = np.concatenate([x_pairs, x_walls])
    n_pairs = len(x_pairs)

    # Soft min: -logsumexp(-beta * x) / beta
    m = np.min(x_all)
    shifted = x_all - m
    exp_terms = np.exp(-beta * shifted)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    val = m - np.log(Z) / beta

    # Gradient
    grad = np.zeros_like(C)

    P_pairs = P[:n_pairs]
    D_flat = D[iu]
    U = D_flat / L_safe[:, np.newaxis]
    W = P_pairs * 0.5

    np.add.at(grad, iu[0], W[:, np.newaxis] * U)
    np.add.at(grad, iu[1], -W[:, np.newaxis] * U)

    P_walls = P[n_pairs:]
    r_safe = np.where(r_pts > 1e-12, r_pts, 1.0)
    U_walls = -C / r_safe[:, np.newaxis]
    grad += P_walls[:, np.newaxis] * U_walls

    return -val, -grad

def _hex_init(n, rotation=0.0, seed=42):
    """Hexagonal lattice init with rotation."""
    rng = np.random.RandomState(seed)
    s = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
    k = int(1.0 / s) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 1.0:
                pts.append([x, y])
    pts = np.array(pts)
    if rotation != 0.0:
        c, s_r = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, -s_r], [s_r, c]])
        pts = pts @ R.T
    d = np.linalg.norm(pts, axis=1)
    idx = np.argsort(d, kind='stable')[:n]
    pts = pts[idx].copy()
    pts += rng.normal(0, s * 0.005, size=pts.shape)
    return pts

def _optimize(points, beta, max_iter=400, time_budget=10.0, start=None):
    """Run L-BFGS-B with analytic gradient and time guard."""
    if start is None:
        start = time.time()
    x0 = points.flatten()
    def f(x):
        if time.time() - start > time_budget:
            raise TimeoutError()
        return _soft_min_obj_grad(x, beta)
    try:
        res = minimize(f, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-14})
        if res.success or res.status in (1, 2, 4):
            out = res.x.reshape(N, 2)
            r = np.hypot(out[:, 0], out[:, 1])
            mask = r > 0.9999
            if np.any(mask):
                out[mask] *= (0.9999 / r[mask])[:, None]
            return out
    except Exception:
        pass
    return points

def _gd_polish(points, beta, lr=1e-4, n_iter=200, start=None):
    """Plain gradient descent polish."""
    if start is None:
        start = time.time()
    cur = points.copy()
    for i in range(n_iter):
        if time.time() - start > 8.0:
            break
        _, g = _soft_min_obj_grad(cur.flatten(), beta)
        cur = cur - lr * g.reshape(N, 2)
        r = np.hypot(cur[:, 0], cur[:, 1])
        mask = r > 0.9999
        if np.any(mask):
            cur[mask] *= (0.9999 / r[mask])[:, None]
    return cur

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Multi-start: several hex rotations
    inits = []
    for rot, seed in [(0.0, 42), (np.pi/6, 43), (np.pi/3, 44), (np.pi/4, 45), (np.pi/12, 46), (np.pi/8, 47)]:
        inits.append(_hex_init(N, rotation=rot, seed=seed))

    best_pts = None
    best_score = -np.inf

    # Pass 1: anneal beta for each init
    for init in inits:
        if time.time() > deadline - 15:
            break
        cur = init.copy()
        # Anneal beta: 30 -> 300 in 3 steps
        for beta in [30.0, 100.0, 300.0]:
            if time.time() > deadline - 10:
                break
            cur = _optimize(cur, beta, max_iter=300, time_budget=min(12.0, deadline - time.time() - 5), start=start_time)
        # High-beta polish
        if time.time() < deadline - 8:
            cur = _optimize(cur, 1000.0, max_iter=400, time_budget=min(10.0, deadline - time.time() - 3), start=start_time)
            cur = _gd_polish(cur, 2000.0, lr=5e-5, n_iter=100, start=start_time)
        sc = _compute_score(cur)
        if sc > best_score:
            best_score = sc
            best_pts = cur.copy()

    # Pass 2: refine best with very high beta
    if best_pts is not None and time.time() < deadline - 10:
        cur = best_pts.copy()
        for beta in [3000.0, 8000.0, 20000.0]:
            if time.time() > deadline - 5:
                break
            cur = _optimize(cur, beta, max_iter=500, time_budget=min(12.0, deadline - time.time() - 3), start=start_time)
            cur = _gd_polish(cur, beta * 2, lr=2e-5, n_iter=150, start=start_time)
            sc = _compute_score(cur)
            if sc > best_score:
                best_score = sc
                best_pts = cur.copy()

    # Final validation
    r_final = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r_final > 0.9999
    if np.any(mask):
        best_pts[mask] *= (0.9999 / r_final[mask])[:, None]

    return best_pts
