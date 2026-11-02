"""Hexagon packing: minimize enclosing regular hexagon side length."""
import math
import time
import numpy as np

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def _hex_verts(cx, cy, s, ang):
    # vertices of regular hexagon, side s, center (cx,cy), rotation ang (rad)
    a = ang
    r = s
    xs = cx + r * np.cos(a + np.arange(6) * np.pi / 3.0)
    ys = cy + r * np.sin(a + np.arange(6) * np.pi / 3.0)
    return np.column_stack([xs, ys])

def _poly_area_signed(pts):
    x = pts[:, 0]; y = pts[:, 1]
    x1 = np.roll(x, -1); y1 = np.roll(y, -1)
    return 0.5 * np.sum(x * y1 - x1 * y)

def _poly_overlap_area(ptsA, ptsB):
    # approximate overlap area via Sutherland-Hodgman clip of A by B
    def clip(subject, clipper):
        # clip polygon subject by convex clipper
        def inside(p, e1, e2):
            return np.cross(e2 - e1, p - e1) >= -1e-12
        out = list(subject)
        for i in range(len(clipper)):
            e1 = clipper[i]; e2 = clipper[(i + 1) % len(clipper)]
            inp = out
            out = []
            if not inp:
                continue
            for j in range(len(inp)):
                cur = inp[j]; nxt = inp[(j + 1) % len(inp)]
                ci = inside(cur, e1, e2); ni = inside(nxt, e1, e2)
                if ci and ni:
                    out.append(cur)
                elif ci and not ni:
                    # exit: intersect
                    d = e2 - e1; f = cur - e1
                    denom = np.cross(d, nxt - cur)
                    if abs(denom) > 1e-15:
                        t = np.cross(f, d) / denom
                        out.append(cur + t * (nxt - cur))
                elif not ci and ni:
                    d = e2 - e1; f = cur - e1
                    denom = np.cross(d, nxt - cur)
                    if abs(denom) > 1e-15:
                        t = np.cross(f, d) / denom
                        out.append(cur + t * (nxt - cur))
                    out.append(nxt)
        if not out:
            return None
        return np.array(out)
    res = clip(ptsA, ptsB)
    if res is None or len(res) < 3:
        return 0.0
    return abs(_poly_area_signed(res))

def _containment_deficit(pts_inner, cx, cy, S, ang):
    # max distance of inner vertices from center along 6 hex directions minus S*inradius
    # A regular hexagon of side S centered at (cx,cy) rotated ang: a point p is inside
    # iff for each of 6 outward normals n_k, (p - c) . n_k <= S * inradius_unit * ... 
    # inradius of side-S hexagon = S * sqrt(3)/2
    inr = S * math.sqrt(3) / 2.0
    n = np.array([[math.cos(ang + k * np.pi / 3.0),
                   math.sin(ang + k * np.pi / 3.0)] for k in range(6)])
    rel = pts_inner - np.array([cx, cy])
    dists = rel @ n.T  # (V,6)
    return np.max(dists) - inr  # positive means a vertex sticks out

def _objective(params, t_start):
    # params: 2N (positions) + N (angles) + 4 (outer cx,cy,S,ang)
    px = params[:N]; py = params[N:2*N]
    inner_ang = params[2*N:3*N]
    ocx = params[3*N]; ocy = params[3*N+1]; S = params[3*N+2]; oang = params[3*N+3]
    if S < 0.5:
        S = 0.5
    verts = []
    for i in range(N):
        verts.append(_hex_verts(px[i], py[i], 1.0, inner_ang[i]))
    # pairwise overlap penalty
    ov_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            ov = _poly_overlap_area(verts[i], verts[j])
            ov_pen += ov * ov
    # containment: outer hexagon
    allv = np.vstack(verts)
    deficit = _containment_deficit(allv, ocx, ocy, S, oang)
    cont_pen = max(0.0, deficit) ** 2
    return S + 100.0 * ov_pen + 1000.0 * cont_pen

def _grad_step(params, t_start, lr):
    # simple gradient via finite differences (sparse)
    f0 = _objective(params, t_start)
    g = np.zeros_like(params)
    h = 1e-4
    for i in range(len(params)):
        p2 = params.copy(); p2[i] += h
        g[i] = (_objective(p2, t_start) - f0) / h
    return f0, g

def _optimize(params0, t_start, iters, lr0, lr_decay):
    params = params0.copy()
    best = _objective(params, t_start)
    best_p = params.copy()
    lr = lr0
    for it in range(iters):
        if time.time() > DEADLINE:
            break
        f0, g = _grad_step(params, t_start, lr)
        if f0 < best:
            best = f0; best_p = params.copy()
        step = lr * g
        # simple line search
        for _ in range(10):
            cand = params - step
            fc = _objective(cand, t_start)
            if fc < f0:
                params = cand
                break
            step *= 0.5
        lr *= lr_decay
    return best_p, best

def _initial_guess(seed):
    rng = np.random.RandomState(seed)
    # honeycomb-ish start
    px = np.zeros(N); py = np.zeros(N)
    # center + ring
    pos = [(0,0)]
    for k in range(6):
        a = k * np.pi / 3.0
        pos.append((math.sqrt(3) * math.cos(a), math.sqrt(3) * math.sin(a)))
    # extra 4 near
    for k in range(4):
        a = (k + 0.5) * np.pi / 3.0
        pos.append((2 * math.sqrt(3) * math.cos(a), 2 * math.sqrt(3) * math.sin(a)))
    for i in range(N):
        px[i] = pos[i][0] + rng.normal(0, 0.1)
        py[i] = pos[i][1] + rng.normal(0, 0.1)
    inner_ang = rng.uniform(0, 2 * np.pi, N)
    params = np.concatenate([px, py, inner_ang, [0.0, 0.0, 3.0, 0.0]])
    return params

def construct():
    t_start = time.time()
    best_score = None
    best_params = None
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    iters_per_seed = 60
    for s in seeds:
        if time.time() > DEADLINE - 5.0:
            break
        p0 = _initial_guess(s)
        pp, sc = _optimize(p0, t_start, iters_per_seed, lr0=0.05, lr_decay=0.9)
        # compute true S from best params
        S = max(0.5, pp[3*N+2])
        if best_score is None or sc < best_score:
            best_score = sc
            best_params = pp
    if best_params is None:
        # fallback: row packing, compute enclosing side
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    px = best_params[:N]; py = best_params[N:2*N]
    inner_ang = best_params[2*N:3*N]
    ocx = best_params[3*N]; ocy = best_params[3*N+1]; S = best_params[3*N+2]; oang = best_params[3*N+3]
    # verify: compute actual required S by max projection
    verts = []
    for i in range(N):
        verts.append(_hex_verts(px[i], py[i], 1.0, inner_ang[i]))
    allv = np.vstack(verts)
    inr_unit = math.sqrt(3) / 2.0
    n = np.array([[math.cos(oang + k * np.pi / 3.0),
                   math.sin(oang + k * np.pi / 3.0)] for k in range(6)])
    rel = allv - np.array([ocx, ocy])
    dists = rel @ n.T
    S_req = np.max(dists) / inr_unit
    S_final = max(S, S_req, 0.5)
    inner = [[float(px[i]), float(py[i]), float(math.degrees(inner_ang[i]))] for i in range(N)]
    return inner, (float(ocx), float(ocy)), float(S_final), float(math.degrees(oang))
