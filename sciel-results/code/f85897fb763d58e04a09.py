# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 630.0
_EPS = 1e-6
_SQRT3 = np.sqrt(3.0)
_C30 = np.cos(np.deg2rad(30.0))  # apothem of unit hex = sqrt(3)/2


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))


def _hex_edge_normals(a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6) + np.deg2rad(90.0)
    return np.column_stack((np.cos(th), np.sin(th)))


def _gap2(va, na, vb, nb):
    # va,vb: (6,2) vertices; na,nb: (6,2) normals
    pa = va @ na.T  # (6,6): projection of va vertices onto na normals
    pb = vb @ na.T
    g1 = np.min(pb, axis=0) - np.max(pa, axis=0)  # (6,)
    pa2 = va @ nb.T
    pb2 = vb @ nb.T
    g2 = np.min(pb2, axis=0) - np.max(pa2, axis=0)  # (6,)
    g = np.maximum(g1, g2)
    return float(np.max(g) - _EPS)


def _fit_side(vertices, a_deg):
    c = vertices.mean(axis=0)
    n = _hex_edge_normals(a_deg)
    d = vertices - c
    caps = d @ n.T  # (6,6)
    return float(np.max(np.max(caps, axis=0)))


def _softplus(x):
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -700, 700))))


def _build_seed():
    axials = [
        (0, 0),
        (1, 0), (1, -1), (0, -1),
        (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2),
        (-2, 0), (-2, 1),
    ]
    pos = []
    for q, r in axials:
        x = 2.0 * q + r
        y = _SQRT3 * r
        pos.append((x, y))
    pos = np.array(pos)
    pos -= pos.mean(axis=0)
    inner = np.zeros((11, 3))
    inner[:, 0] = pos[:, 0]
    inner[:, 1] = pos[:, 1]
    inner[:, 2] = 0.0
    verts = []
    for i in range(11):
        verts.append(_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]))
    allv = np.vstack(verts)
    side = _fit_side(allv, 0.0)
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, side


def _candidate_side(params):
    inner = params[:33].reshape(11, 3)
    outer = params[33:36]
    cx, cy, oa = outer
    vs = np.empty((11, 6, 2))
    for i in range(11):
        vs[i] = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
    allv = vs.reshape(-1, 2)
    d = allv - np.array([cx, cy])
    no = _hex_edge_normals(oa)
    caps = d @ no.T
    return float(np.max(np.max(caps, axis=0)))


def _unpack(params):
    inner = params[:33].reshape(11, 3)
    outer = params[33:36]
    side = _candidate_side(params)
    return inner, outer, side


def _objective(params, n):
    inner = params[:33].reshape(11, 3)
    outer = params[33:36]
    cx, cy, oa = outer

    vs = np.empty((11, 6, 2))
    ns = np.empty((11, 6, 2))
    for i in range(11):
        vs[i] = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        ns[i] = _hex_edge_normals(inner[i, 2])

    allv = vs.reshape(-1, 2)
    d = allv - np.array([cx, cy])
    no = _hex_edge_normals(oa)
    caps = d @ no.T  # (66, 6)
    side = float(np.max(np.max(caps, axis=0)))
    if not np.isfinite(side) or side <= 0:
        return 1e6

    obj = 1.0 / side

    # Containment barrier
    margin = 0.01
    cont = 0.0
    for i in range(11):
        for j in range(6):
            dist = (side * _C30) - (d[i * 6 + j] @ no)
            m = float(np.min(dist))
            cont += _softplus(-(m - margin))
    obj += 1000.0 * cont

    # Non-penetration barrier between inner pairs
    sep = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            g = _gap2(vs[i], ns[i], vs[j], ns[j])
            if g < 0:
                sep += _softplus(-g * 50.0)
    obj += 1000.0 * sep

    return obj


def hexagon_packing_11():
    inner, outer, side = _build_seed()
    best_params = np.concatenate([inner.ravel(), outer])
    best_side = side

    def remaining():
        return _DEADLINE - (time.time() - _START)

    if remaining() < 20.0:
        return inner, outer, side

    x0 = best_params.copy()
    try:
        res = minimize(
            _objective,
            x0,
            args=(11,),
            method="L-BFGS-B",
            options={"maxiter": 300, "maxfun": 20000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if res.x is not None:
            cand_side = _candidate_side(res.x)
            if np.isfinite(cand_side) and cand_side < best_side:
                best_params = res.x
                best_side = cand_side
    except Exception:
        pass

    if remaining() < 15.0:
        return _unpack(best_params)

    try:
        res2 = minimize(
            _objective,
            best_params,
            args=(11,),
            method="L-BFGS-B",
            options={"maxiter": 200, "maxfun": 20000, "ftol": 1e-13, "gtol": 1e-9},
        )
        if res2.x is not None:
            cand_side = _candidate_side(res2.x)
            if np.isfinite(cand_side) and cand_side < best_side:
                best_params = res2.x
                best_side = cand_side
    except Exception:
        pass

    return _unpack(best_params)


# EVOLVE-BLOCK-END
