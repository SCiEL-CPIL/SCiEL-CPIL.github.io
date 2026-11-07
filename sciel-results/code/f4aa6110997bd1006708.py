"""Heuristic for ACO-CVRP: adaptive parameter selection with depot-aware heuristic."""
import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 300.0
_TIME_MARGIN = 15.0
_N = 51


def _remaining():
    return _DEADLINE - _TIME_MARGIN - time.time()


def _safe_heuristic(dm):
    d = np.asarray(dm, dtype=float).copy()
    np.fill_diagonal(d, 1e9)
    H = 1.0 / (d + 1e-9)
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    np.fill_diagonal(H, 1e-9)
    return H


def _distmat(coords):
    c = np.asarray(coords, dtype=float)
    return np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))


def _ant_route(dist, D, rem, cap, logP, logH, max_steps):
    node = 0
    rem = cap
    cost = 0.0
    visited = np.zeros(_N, bool)
    visited[0] = True
    steps = 0
    while True:
        feas = []
        for j in range(1, _N):
            if not visited[j] and D[j] <= rem + 1e-9:
                feas.append(j)
        if not feas:
            cost += dist[node, 0]
            return cost
        vals = np.array([logP[node, j] + logH[node, j] for j in feas])
        m = vals.max()
        w = np.exp(vals - m)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            nxt = feas[0]
        else:
            r = np.random.random() * s
            acc = 0.0
            for k, j in enumerate(feas):
                acc += w[k]
                if r <= acc:
                    nxt = j
                    break
            else:
                nxt = feas[-1]
        cost += dist[node, nxt]
        rem -= D[nxt]
        visited[nxt] = True
        node = nxt
        steps += 1
        if steps >= max_steps:
            cost += dist[node, 0]
            return cost


def _colony(dist, D, cap, logH, n_ants=20, n_iter=30, max_steps=300):
    P = np.ones((_N, _N))
    bests = []
    for _ in range(n_iter):
        itcost = np.empty(n_ants)
        logP = np.log(P)
        for a in range(n_ants):
            c = _ant_route(dist, D, cap, cap, logP, logH, max_steps)
            itcost[a] = c
        b = float(itcost.min())
        bests.append(b)
        P *= 0.9
        P += 1.0 / b
    return min(bests)


def _build_heuristic(dm, coords, alpha, S, depot_weight=0.0):
    d = np.asarray(dm, dtype=float).copy()
    np.fill_diagonal(d, 1e9)
    base = 1.0 / (d + 1e-9)
    H = S * (base ** alpha)
    if depot_weight > 0.0:
        c = np.asarray(coords, dtype=float)
        depot_dist = np.sqrt((c[0, :] ** 2).sum() if c.ndim == 1 else ((c - c[0]) ** 2).sum(-1))
        depot_boost = 1.0 + depot_weight / (depot_dist + 1e-9)
        H *= depot_boost
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    np.fill_diagonal(H, 1e-9)
    return H


def heuristics(distance_matrix, coordinates, demands, capacity):
    dm = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    D = np.asarray(demands, dtype=float)
    cap = float(capacity)
    n = dm.shape[0]
    
    try:
        if dm.shape != (n, n):
            raise ValueError("shape")
        if not np.isfinite(dm).all() or not np.isfinite(coords).all() or not np.isfinite(D).all():
            raise ValueError("non-finite")
        
        if n != _N:
            return _safe_heuristic(dm)
        
        dist = _distmat(coords)
        
        if _remaining() <= 30.0:
            return _build_heuristic(dm, coords, 1.0, 1.0, 0.0)
        
        # Quick evaluation with a few candidate (alpha, S, dw) tuples
        best_m = 1e30
        best_params = (1.0, 1.0, 0.0)
        
        alpha_vals = [0.7, 1.0, 1.3, 1.6, 2.0]
        S_vals = [0.5, 1.0, 2.0, 4.0]
        dw_vals = [0.0, 0.3, 0.6, 1.0]
        
        n_ants_short = 8
        n_iter_short = 6
        max_steps_short = 120
        
        for alpha in alpha_vals:
            for S in S_vals:
                for dw in dw_vals:
                    if _remaining() <= 30.0:
                        break
                    H = _build_heuristic(dm, coords, alpha, S, dw)
                    logH = np.log(H)
                    m = _colony(dist, D, cap, logH, n_ants_short, n_iter_short, max_steps_short)
                    if m < best_m:
                        best_m = m
                        best_params = (alpha, S, dw)
                if _remaining() <= 30.0:
                    break
            if _remaining() <= 30.0:
                break
        
        alpha_opt, S_opt, dw_opt = best_params
        
        # Refine around the best with full colony if time permits
        if _remaining() > 50.0:
            cur_a, cur_S, cur_dw = alpha_opt, S_opt, dw_opt
            cur_H = _build_heuristic(dm, coords, cur_a, cur_S, cur_dw)
            cur_logH = np.log(cur_H)
            cur_m = _colony(dist, D, cap, cur_logH, 15, 12, 200)
            
            for _round in range(5):
                if _remaining() <= 40.0:
                    break
                improved = False
                for da in [0.2, -0.2, 0.1, -0.1]:
                    na = cur_a + da
                    if na < 0.2 or na > 5.0:
                        continue
                    Ht = _build_heuristic(dm, coords, na, cur_S, cur_dw)
                    logHt = np.log(Ht)
                    mt = _colony(dist, D, cap, logHt, 15, 12, 200)
                    if mt < cur_m - 1e-6:
                        cur_a, cur_H, cur_logH, cur_m = na, Ht, logHt, mt
                        improved = True
                for ds in [1.4, 1/1.4, 1.2, 1/1.2]:
                    nS = cur_S * ds
                    if nS < 0.1 or nS > 20.0:
                        continue
                    Ht = _build_heuristic(dm, coords, cur_a, nS, cur_dw)
                    logHt = np.log(Ht)
                    mt = _colony(dist, D, cap, logHt, 15, 12, 200)
                    if mt < cur_m - 1e-6:
                        cur_S, cur_H, cur_logH, cur_m = nS, Ht, logHt, mt
                        improved = True
                for ddw in [0.3, -0.3, 0.15, -0.15]:
                    ndw = cur_dw + ddw
                    if ndw < 0.0 or ndw > 3.0:
                        continue
                    Ht = _build_heuristic(dm, coords, cur_a, cur_S, ndw)
                    logHt = np.log(Ht)
                    mt = _colony(dist, D, cap, logHt, 15, 12, 200)
                    if mt < cur_m - 1e-6:
                        cur_dw, cur_H, cur_logH, cur_m = ndw, Ht, logHt, mt
                        improved = True
                if not improved:
                    break
            
            alpha_opt, S_opt, dw_opt = cur_a, cur_S, cur_dw
        
        H_final = _build_heuristic(dm, coords, alpha_opt, S_opt, dw_opt)
        return H_final
        
    except Exception:
        try:
            return _safe_heuristic(dm)
        except Exception:
            return np.ones_like(dm)
