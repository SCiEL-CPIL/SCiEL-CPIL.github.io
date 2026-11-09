# EVOLVE-BLOCK-START
import numpy as np
import time


# ---------- scoring (mirrors the task contract) ----------
def _vertex_scores(P):
    n = P.shape[0]
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    scores = np.zeros(n)
    for i in range(n):
        d = np.delete(D[i], i)
        d = np.sort(d)
        if d[0] < 0.05:
            scores[i] = 0.0
            continue
        gaps = d[1:] - d[:-1]
        k = int(np.argmin(gaps))
        Dm = 0.5 * (d[k] + d[k + 1])
        if Dm < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            idx = [0, 1, 2, 3]
        elif k == n - 3:  # k == 17 for n=20
            idx = [n - 5, n - 4, n - 3, n - 2]  # 15..18
        else:
            idx = [k - 1, k, k + 1, k + 2]
        vals = np.array([d[j] for j in idx])
        raw = np.mean(np.maximum(vals / Dm, Dm / vals))
        scores[i] = min(1.0, 1.0 / raw)
    return scores


def _score(P):
    return float(np.min(_vertex_scores(P)))


def _is_valid(P):
    n = P.shape[0]
    if not np.all(np.isfinite(P)):
        return False
    if np.any(np.diff(P, axis=0, append=P[:1]) == 0.0):
        return False
    for i in range(n):
        a = P[(i + 1) % n] - P[i]
        b = P[(i + 2) % n] - P[(i + 1) % n]
        cross = a[0] * b[1] - a[1] * b[0]
        if cross == 0.0:
            return False
    # all same sign
    signs = []
    for i in range(n):
        a = P[(i + 1) % n] - P[i]
        b = P[(i + 2) % n] - P[(i + 1) % n]
        cross = a[0] * b[1] - a[1] * b[0]
        signs.append(np.sign(cross))
    if len(set(signs)) != 1:
        return False
    return True


def _normalize(P):
    c = P.mean(axis=0)
    Q = P - c
    m = np.mean(np.linalg.norm(Q, axis=1))
    if m <= 0.01:
        return None
    return Q / m


def _regular(n, radius=1.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)


# ---------- smooth surrogate objective ----------
def _softmin(vals, beta):
    vals = np.asarray(vals)
    mx = np.max(vals)
    if not np.isfinite(mx):
        return -1e9
    s = np.sum(np.exp(-beta * (vals - mx)))
    return mx - np.log(s) / beta


def _obj(theta, phi, beta=20.0):
    """Return a smooth surrogate to maximize (higher = better)."""
    r = np.clip(phi, 0.3, 3.0)
    P = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    P = _normalize(P)
    if P is None:
        return -1e9
    if not _is_valid(P):
        return -1e9
    vs = _vertex_scores(P)
    return _softmin(vs, beta)


# ---------- search ----------
def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    t0 = time.time()
    deadline = t0 + 600.0

    def remaining():
        return deadline - time.time()

    # baseline
    base = _regular(n)
    best_P = base.copy()
    best_score = _score(base)
    best_theta = np.linspace(0, 2 * np.pi, n, endpoint=False).copy()
    best_phi = np.ones(n)

    def record(P, s, theta, phi):
        nonlocal best_P, best_score, best_theta, best_phi
        if s > best_score + 1e-12:
            best_P = P.copy()
            best_score = s
            best_theta = theta.copy()
            best_phi = phi.copy()

    # ---- Stage A: Differential Evolution (global) on (theta, phi) ----
    def de_stage():
        n = 20
        pop = 60
        theta0 = np.linspace(0, 2 * np.pi, n, endpoint=False)
        phi0 = np.ones(n)
        lo_t = theta0 - 0.4
        hi_t = theta0 + 0.4
        lo_p = 0.5
        hi_p = 1.6
        F = 0.8
        CR = 0.9
        pop_theta = np.array([theta0 + (np.random.rand(n) * 2 - 1) * 0.35 for _ in range(pop)])
        pop_theta[:, 0] = 0.0  # fix rotation
        pop_phi = np.array([phi0 + (np.random.rand(n) * 2 - 1) * 0.3 for _ in range(pop)])
        pop_phi[:, 0] = 1.0
        # ensure sorted theta within window
        for i in range(pop):
            pop_theta[i] = np.sort(pop_theta[i])
        pop_obj = np.array([_obj(pop_theta[i], pop_phi[i]) for i in range(pop)])
        best_i = int(np.argmax(pop_obj))
        record(_normalize(np.stack([pop_phi[best_i] * np.cos(pop_theta[best_i]),
                                     pop_phi[best_i] * np.sin(pop_theta[best_i])], axis=1)),
               _score(_normalize(np.stack([pop_phi[best_i] * np.cos(pop_theta[best_i]),
                                            pop_phi[best_i] * np.sin(pop_theta[best_i])], axis=1))),
               pop_theta[best_i], pop_phi[best_i])
        max_gen = 80
        for gen in range(max_gen):
            if remaining() < 40:
                break
            for i in range(pop):
                if remaining() < 25:
                    break
                a_, b_, c_ = np.random.choice([j for j in range(pop) if j != i], 3, replace=False)
                v_t = pop_theta[a_] + F * (pop_theta[b_] - pop_theta[c_])
                v_p = pop_phi[a_] + F * (pop_phi[b_] - pop_phi[c_])
                mask_t = np.random.rand(n) < CR
                mask_p = np.random.rand(n) < CR
                v_t = np.where(mask_t, v_t, pop_theta[i])
                v_p = np.where(mask_p, v_p, pop_phi[i])
                v_t[0] = 0.0
                v_p[0] = 1.0
                v_t = np.clip(v_t, lo_t, hi_t)
                v_p = np.clip(v_p, lo_p, hi_p)
                v_t = np.sort(v_t)
                obj_v = _obj(v_t, v_p)
                if obj_v >= pop_obj[i]:
                    pop_theta[i] = v_t
                    pop_phi[i] = v_p
                    pop_obj[i] = obj_v
                    bi = int(np.argmax(pop_obj))
                    Pn = _normalize(np.stack([pop_phi[bi] * np.cos(pop_theta[bi]),
                                               pop_phi[bi] * np.sin(pop_theta[bi])], axis=1))
                    if Pn is not None and _is_valid(Pn):
                        s = _score(Pn)
                        record(Pn, s, pop_theta[bi], pop_phi[bi])
        # return best of pop
        bi = int(np.argmax(pop_obj))
        return pop_theta[bi], pop_phi[bi]

    if remaining() > 60:
        bt, bp = de_stage()

    # ---- Stage B: local polish with Nelder-Mead on the smooth objective ----
    from scipy.optimize import minimize

    def polish(theta, phi, iters=400):
        n = 20
        x0 = np.concatenate([theta, phi])
        # bounds handled by clipping inside obj via a wrapper
        def wrapped(x):
            t = np.clip(x[:n], best_theta - 0.6, best_theta + 0.6)
            t[0] = 0.0
            t = np.sort(t)
            p = np.clip(x[n:], 0.3, 3.0)
            p[0] = 1.0
            return -_obj(t, p)
        try:
            res = minimize(wrapped, x0, method='Nelder-Mead',
                           options={'maxiter': iters, 'xatol': 1e-9, 'fatol': 1e-12})
            t = np.clip(res.x[:n], best_theta - 0.6, best_theta + 0.6)
            t[0] = 0.0
            t = np.sort(t)
            p = np.clip(res.x[n:], 0.3, 3.0)
            p[0] = 1.0
            Pn = _normalize(np.stack([p * np.cos(t), p * np.sin(t)], axis=1))
            if Pn is not None and _is_valid(Pn):
                s = _score(Pn)
                record(Pn, s, t, p)
        except Exception:
            pass

    # polish from DE best and a few random restarts
    if remaining() > 40:
        polish(best_theta, best_phi, iters=600)

    restarts = 3
    for r in range(restarts):
        if remaining() < 60:
            break
        t0r = np.sort(np.linspace(0, 2 * np.pi, n, endpoint=False) + (np.random.rand(n) * 2 - 1) * 0.25)
        t0r[0] = 0.0
        p0r = np.clip(np.ones(n) + (np.random.rand(n) * 2 - 1) * 0.2, 0.5, 1.6)
        p0r[0] = 1.0
        polish(t0r, p0r, iters=400)

    # ---- Final high-precision polish on the best ----
    if remaining() > 30:
        polish(best_theta, best_phi, iters=1200)

    # ensure validity of returned polygon
    if not _is_valid(best_P):
        best_P = base
    return best_P


# EVOLVE-BLOCK-END
