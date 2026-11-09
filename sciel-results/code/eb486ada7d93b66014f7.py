# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _vertex_score_raw(polygon):
    n = polygon.shape[0]
    diff = polygon[:, None, :] - polygon[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    scores = np.zeros(n)
    for i in range(n):
        d = np.sort(dist[i].copy())
        d = d[1:]
        if np.any(d < 0.05):
            scores[i] = 0.0
            continue
        gaps = d[1:] - d[:-1]
        k = int(np.argmin(gaps))
        D = 0.5 * (d[k] + d[k + 1])
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            idx = [0, 1, 2, 3]
        elif k == 17:
            idx = [15, 16, 17, 18]
        else:
            idx = [k - 1, k, k + 1, k + 2]
        four = d[idx]
        ratio = np.maximum(four / D, D / four)
        raw = np.mean(ratio)
        scores[i] = 1.0 / raw
    return scores

def _metric(polygon):
    n = polygon.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-9:
                return 0.0
    edges = np.zeros((n, 2))
    for i in range(n):
        edges[i] = polygon[(i + 1) % n] - polygon[i]
    turns = np.zeros(n)
    for i in range(n):
        e1 = edges[i]
        e2 = edges[(i + 1) % n]
        turns[i] = e1[0] * e2[1] - e1[1] * e2[0]
    if np.any(np.abs(turns) < 1e-12):
        return 0.0
    if not (np.all(turns > 0) or np.all(turns < 0)):
        return 0.0
    angles = np.zeros(n)
    for i in range(n):
        e1 = edges[i]
        e2 = edges[(i + 1) % n]
        cos_a = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-30)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        sin_a = turns[i] / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-30)
        angles[i] = np.arctan2(sin_a, cos_a)
    total = np.sum(angles)
    if abs(abs(total) - 2.0 * np.pi) > 1e-6:
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists = np.linalg.norm(p, axis=1)
    mean_d = np.mean(dists)
    if mean_d <= 0.01:
        return 0.0
    p = p / mean_d
    vs = np.minimum(1.0, _vertex_score_raw(p))
    return float(np.min(vs))

def _soft_min(scores, temperature=0.05):
    s = scores / temperature
    m = np.min(s)
    return -temperature * (m + np.log(np.sum(np.exp(s - m))))

def _soft_metric(polygon, temperature=0.05):
    n = polygon.shape[0]
    edges = np.zeros((n, 2))
    for i in range(n):
        edges[i] = polygon[(i + 1) % n] - polygon[i]
    turns = np.zeros(n)
    for i in range(n):
        e1 = edges[i]
        e2 = edges[(i + 1) % n]
        turns[i] = e1[0] * e2[1] - e1[1] * e2[0]
    turn_penalty = 0.0
    min_turn = np.min(np.abs(turns))
    if min_turn < 1e-3:
        turn_penalty += (1e-3 - min_turn) ** 2
    if np.any(turns > 0) and np.any(turns < 0):
        turn_penalty += 1.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists = np.linalg.norm(p, axis=1)
    mean_d = np.mean(dists)
    if mean_d <= 0.01:
        return -10.0
    p = p / mean_d
    raw_scores = _vertex_score_raw(p)
    clamped = np.minimum(1.0, raw_scores)
    sm = _soft_min(clamped, temperature)
    return sm - turn_penalty

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 10.0

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x0 = np.stack([np.cos(angles), np.sin(angles)], axis=1).flatten()
    
    best_score = _metric(x0.reshape(n, 2))
    best_params = x0.copy()

    def _neg_soft_coord(pflat, temperature=0.05):
        p = pflat.reshape(n, 2)
        return -_soft_metric(p, temperature)

    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
    np.random.seed(42)
    
    for temp in temperatures:
        for trial in range(5):
            if time.time() > deadline - safety_margin:
                break
            if trial == 0:
                x0 = best_params.copy()
            else:
                x0 = best_params + np.random.randn(40) * 0.1
            try:
                res = minimize(_neg_soft_coord, x0, args=(temp,), method='L-BFGS-B',
                               options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10})
                candidate = res.x.reshape(n, 2)
                score = _metric(candidate)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
            except Exception:
                pass
        if time.time() > deadline - safety_margin:
            break

    if time.time() < deadline - safety_margin:
        try:
            res = minimize(_neg_soft_coord, best_params, args=(0.001,), method='L-BFGS-B',
                           options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-12})
            candidate = res.x.reshape(n, 2)
            score = _metric(candidate)
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass

    result = best_params.reshape(n, 2)
    if _metric(result) <= 0.0:
        result = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return result

# EVOLVE-BLOCK-END
