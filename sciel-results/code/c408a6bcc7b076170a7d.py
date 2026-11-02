import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

def construct():
    start = time.time()
    TL = 145.0  # hard wall-clock guard (evaluator kills at 150 s)

    # ---- vectorized triangle area over all C(13,3)=286 triples ----
    ii, jj, kk = [], [], []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                ii.append(a); jj.append(b); kk.append(c)
    ii = np.array(ii, dtype=np.int32); jj = np.array(jj, dtype=np.int32); kk = np.array(kk, dtype=np.int32)

    def _hull_area(P):
        try:
            return float(ConvexHull(P).volume)
        except Exception:
            return 0.0

    def _areas(P):
        x = P[:, 0]; y = P[:, 1]
        v1x = x[jj] - x[ii]; v1y = y[jj] - y[ii]
        v2x = x[kk] - x[ii]; v2y = y[kk] - y[ii]
        return 0.5 * np.abs(v1x * v2y - v1y * v2x)

    def _score(P):
        ha = _hull_area(P)
        if ha < 1e-9:
            return 0.0
        return float(np.min(_areas(P))) / ha

    def _negscore(P):
        return -_score(P)

    def _softneg(P, beta, ref):
        ha = _hull_area(P)
        if ha < 1e-9:
            return 1e6
        ar = _areas(P)
        neg = -beta * ar
        m = np.max(neg)
        if m < -1e6:
            return 1e6
        soft_min = -(m + np.log(np.sum(np.exp(neg - m)))) / beta
        return -(soft_min / ha) + 0.5 * (1.0 - ha / ref) ** 2

    # ---- parametric family: center + n_in inner ring + n_out outer ring ----
    def _build(n_in, r_in, phi_in, n_out, r_out, phi_out):
        parts = [np.array([[0.0, 0.0]])]
        if n_in > 0:
            a = np.linspace(phi_in, phi_in + 2 * np.pi, n_in, endpoint=False)
            parts.append(np.column_stack([r_in * np.cos(a), r_in * np.sin(a)]))
        if n_out > 0:
            b = np.linspace(phi_out, phi_out + 2 * np.pi, n_out, endpoint=False)
            parts.append(np.column_stack([r_out * np.cos(b), r_out * np.sin(b)]))
        return np.vstack(parts)

    def _param_obj(p, n_in, n_out, ref=1.0):
        P = _build(n_in, p[0], p[1], n_out, p[2], p[3])
        return _softneg(P, 60.0, ref)

    def _hard_param(p, n_in, n_out):
        P = _build(n_in, p[0], p[1], n_out, p[2], p[3])
        return -_score(P)

    def _local_polish(P, deadline):
        """Targeted search on the active (smallest) triangle, then global NM."""
        best = P.copy(); bs = _score(best)
        # 1) nudge the 3 points of the current minimum triangle
        try:
            ar = _areas(best)
            idx = int(np.argmin(ar))
            tri = [int(ii[idx]), int(jj[idx]), int(kk[idx])]
            base = best[tri].copy()
            def _tri_obj(v):
                Q = best.copy(); Q[tri] = v.reshape(3, 2)
                return -_score(Q)
            r = opt.minimize(_tri_obj, base.flatten(), method='Nelder-Mead',
                             options={'maxiter': 300, 'adaptive': True,
                                      'xatol': 1e-11, 'fatol': 1e-13})
            Q = r.x.reshape(3, 2); best[tri] = Q
            s = _score(best)
            if s > bs:
                bs = s
        except Exception:
            pass
        # 2) full Nelder-Mead polish on all 26 coords
        if time.time() < deadline - 1.0:
            try:
                r = opt.minimize(_negscore, best.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 400, 'adaptive': True,
                                          'xatol': 1e-12, 'fatol': 1e-14})
                Q = r.x.reshape(N, 2)
                s = _score(Q)
                if s > bs:
                    bs = s; best = Q
            except Exception:
                pass
        # 3) high-beta soft L-BFGS to smooth the landscape
        if time.time() < deadline - 1.0:
            try:
                ref = max(_hull_area(best), 1e-6)
                r = opt.minimize(lambda x: _softneg(x.reshape(N, 2), 400.0, ref),
                                 best.flatten(), method='L-BFGS-B',
                                 options={'maxiter': 80, 'ftol': 1e-15,
                                          'gtol': 1e-13, 'maxls': 60})
                Q = r.x.reshape(N, 2)
                s = _score(Q)
                if s > bs:
                    bs = s; best = Q
            except Exception:
                pass
        return best, bs

    # ---- 1) global differential_evolution over the few parametric DOF ----
    best_P = None; best_s = -1e18
    splits = [(6, 6), (5, 7), (7, 5), (4, 8), (8, 4), (3, 9), (9, 3), (2, 10), (10, 2)]
    rng = np.random.RandomState(0)
    for n_in, n_out in splits:
        if time.time() > TL - 40.0:
            break
        bounds = [(0.0, 1.0), (0.0, 6.2832), (0.55, 1.6), (0.0, 6.2832)]
        x0 = rng.uniform(0.1, 0.9, size=4)
        try:
            r = opt.differential_evolution(
                lambda p: _param_obj(p, n_in, n_out), bounds, seed=123,
                maxiter=30, popsize=14, tol=1e-6, mutation=(0.4, 1.0),
                recombination=0.8, polish=False, init='latinhypercube',
                updating='deferred')
            P = _build(n_in, r.x[0], r.x[1], n_out, r.x[2], r.x[3])
            s = _score(P)
            if s > best_s:
                best_s = s; best_P = P.copy()
        except Exception:
            continue

    # ---- 2) parametric NM polish on the best split ----
    if best_P is not None and time.time() < TL - 30.0:
        # recover a parametric start near best_P: use the first split that built it
        for n_in, n_out in splits:
            if time.time() > TL - 25.0:
                break
            bounds = [(0.0, 1.0), (0.0, 6.2832), (0.55, 1.6), (0.0, 6.2832)]
            try:
                r = opt.minimize(lambda p: _hard_param(p, n_in, n_out),
                                 np.array([0.4, 0.0, 1.0, 0.0]),
                                 method='Nelder-Mead',
                                 options={'maxiter': 300, 'adaptive': True,
                                          'xatol': 1e-12, 'fatol': 1e-14})
                P = _build(n_in, r.x[0], r.x[1], n_out, r.x[2], r.x[3])
                s = _score(P)
                if s > best_s:
                    best_s = s; best_P = P.copy()
            except Exception:
                continue

    if best_P is None:
        # fallback: pure regular 13-gon
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_P = np.column_stack([np.cos(a), np.sin(a)]); best_s = _score(best_P)

    # ---- 3) targeted local search on the active triangle + global polish ----
    if time.time() < TL - 12.0:
        best_P, best_s = _local_polish(best_P, TL - 12.0)

    # ---- 4) a couple of tiny-perturbation NM restarts from the best ----
    if time.time() < TL - 8.0:
        for ps in [7, 99]:
            if time.time() > TL - 6.0:
                break
            rngp = np.random.RandomState(ps)
            Pp = best_P + rngp.randn(N, 2) * 0.004
            try:
                r = opt.minimize(_negscore, Pp.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 300, 'adaptive': True,
                                          'xatol': 1e-12, 'fatol': 1e-14})
                Q = r.x.reshape(N, 2)
                s = _score(Q)
                if s > best_s:
                    best_s = s; best_P = Q
            except Exception:
                pass

    # ---- final safety: always return a valid (N,2) array ----
    if best_P is None or best_P.shape != (N, 2) or not np.isfinite(best_P).all():
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_P = np.column_stack([np.cos(a), np.sin(a)])

    return best_P
