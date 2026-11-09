# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _normalize(pts):
    centroid = pts.mean(axis=0)
    p = pts - centroid
    d = np.linalg.norm(p, axis=1)
    mean_d = d.mean()
    if mean_d <= 0.01:
        return None, None, None
    p = p / mean_d
    return p, d, mean_d

def _check_strictly_convex(p):
    n = len(p)
    if n < 3:
        return False
    for i in range(n):
        if np.linalg.norm(p[i] - p[(i + 1) % n]) < 1e-9:
            return False
    signs = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        signs.append(1 if cross > 0 else -1)
    if signs.count(signs[0]) != n:
        return False
    angles = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = np.clip(cos_a, -1, 1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sin_a = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.arctan2(sin_a, cos_a)
        angles.append(ang)
    total_turn = sum(angles)
    expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(p, v_idx):
    n = len(p)
    v = p[v_idx]
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    if np.any(d_sorted < 0.05):
        return 0.0
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))
    D = (d_sorted[k] + d_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d_sorted[0:4]
    elif k == 17:
        selected = d_sorted[15:19]
    else:
        selected = d_sorted[k - 1:k + 3]
    if len(selected) < 4:
        return 0.0
    raw = np.mean([max(di / D, D / di) for di in selected])
    if raw < 1e-12:
        return 0.0
    score = min(1.0, 1.0 / raw)
    return score

def _score_polygon(p):
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    min_vs = 1.0
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:
            break
    return min_vs

def _regular_polygon(n=20, radius=1.0):
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_polygon_from_params(params):
    n = 20
    r = params[:n]
    d_theta = params[n:]
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + d_theta
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

def _objective(params):
    pts = _generate_polygon_from_params(params)
    return -_score_polygon(pts)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cos_a = np.cos(base_angles)
    sin_a = np.sin(base_angles)

    best_score = 0.0
    best_radii = np.ones(n)
    best_dtheta = np.zeros(n)

    def radii_to_pts(r, dt=None):
        if dt is None:
            dt = np.zeros(n)
        angles = base_angles + dt
        return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

    def try_candidate(r, dt=None):
        nonlocal best_score, best_radii, best_dtheta
        pts = radii_to_pts(r, dt)
        s = _score_polygon(pts)
        if s > best_score:
            best_score = s
            best_radii = r.copy()
            best_dtheta = (dt if dt is not None else np.zeros(n)).copy()

    try_candidate(np.ones(n), np.zeros(n))

    # Seeds
    seeds = []
    seeds.append(np.ones(40))
    seeds.append(np.concatenate([1.0 + 0.1 * np.array([1, -1] * 10), np.zeros(n)]))
    seeds.append(np.concatenate([1.0 + 0.05 * np.array([1, -1] * 10), np.zeros(n)]))
    seeds.append(np.concatenate([1.0 - 0.1 * np.array([1, -1] * 10), np.zeros(n)]))
    for k in [2, 3, 5, 7]:
        for amp in [0.05, 0.1, 0.2]:
            seeds.append(np.concatenate([1.0 + amp * np.sin(k * base_angles), np.zeros(n)]))
            seeds.append(np.concatenate([1.0 + amp * np.cos(k * base_angles), np.zeros(n)]))
    x = np.linspace(-1, 1, n)
    seeds.append(np.concatenate([1.0 + 0.1 * x**2, np.zeros(n)]))
    seeds.append(np.concatenate([1.0 - 0.1 * x**2, np.zeros(n)]))

    for seed in seeds:
        if time.time() > deadline:
            break
        seed = np.clip(seed[:n], 0.5, 2.0)
        seed_dt = seed[n:]
        try_candidate(seed, seed_dt)

    bounds = [(0.5, 2.0)] * n + [(-0.2, 0.2)] * n

    # Differential evolution with multiple restarts
    maxiter = 300
    popsize = 15
    tol = 1e-4

    # Restart 1: from regular
    if time.time() < deadline:
        try:
            res = differential_evolution(
                _objective, bounds,
                x0=np.ones(40),
                maxiter=maxiter,
                popsize=popsize,
                tol=tol,
                seed=42,
                polish=True
            )
            if res.fun < 0:
                best_params = res.x
                best_r = np.clip(best_params[:n], 0.5, 2.0)
                best_dt = best_params[n:]
                try_candidate(best_r, best_dt)
        except Exception:
            pass

    # Restart 2: from best alternating
    if time.time() < deadline:
        try:
            x0 = np.concatenate([1.0 + 0.1 * np.array([1, -1] * 10), np.zeros(n)])
            res = differential_evolution(
                _objective, bounds,
                x0=x0,
                maxiter=maxiter,
                popsize=popsize,
                tol=tol,
                seed=43,
                polish=True
            )
            if res.fun < 0:
                best_params = res.x
                best_r = np.clip(best_params[:n], 0.5, 2.0)
                best_dt = best_params[n:]
                try_candidate(best_r, best_dt)
        except Exception:
            pass

    # Restart 3: from best sinusoidal
    if time.time() < deadline:
        try:
            x0 = np.concatenate([1.0 + 0.1 * np.sin(2 * base_angles), np.zeros(n)])
            res = differential_evolution(
                _objective, bounds,
                x0=x0,
                maxiter=maxiter,
                popsize=popsize,
                tol=tol,
                seed=44,
                polish=True
            )
            if res.fun < 0:
                best_params = res.x
                best_r = np.clip(best_params[:n], 0.5, 2.0)
                best_dt = best_params[n:]
                try_candidate(best_r, best_dt)
        except Exception:
            pass

    # Final polish: coordinate ascent on radii and angles around global best
    if best_score < 1.0 and time.time() < deadline:
        current_r = best_radii.copy()
        current_dt = best_dtheta.copy()
        step_r = 0.01
        step_dt = 0.01
        for _ in range(200):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                for delta in [step_r, -step_r]:
                    new_r = current_r.copy()
                    new_r[i] += delta
                    if new_r[i] < 0.5 or new_r[i] > 2.5:
                        continue
                    s_new = _score_polygon(radii_to_pts(new_r, current_dt))
                    if s_new > best_score:
                        best_score = s_new
                        best_radii = new_r.copy()
                        current_r = new_r
                        improved = True
                for delta in [step_dt, -step_dt]:
                    new_dt = current_dt.copy()
                    new_dt[i] += delta
                    if new_dt[i] < -0.25 or new_dt[i] > 0.25:
                        continue
                    s_new = _score_polygon(radii_to_pts(current_r, new_dt))
                    if s_new > best_score:
                        best_score = s_new
                        best_dtheta = new_dt.copy()
                        current_dt = new_dt
                        improved = True
            if not improved:
                step_r *= 0.8
                step_dt *= 0.8
                if step_r < 0.0001:
                    break

    return radii_to_pts(best_radii, best_dtheta)

# EVOLVE-BLOCK-END
