import numpy as np
from scipy.optimize import minimize, differential_evolution, linprog
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0

tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)
N_TRI = len(tri_idx)

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_TOTAL = np.sqrt(3) / 4.0

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(x, y):
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - w - v
    return [u, v]

def calc_areas_bary(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v

    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]

    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    return np.abs(det)

def clamp_bary(bary, eps=1e-10):
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.maximum(sums[invalid], 1e-12)
        u[invalid] *= scale
        v[invalid] *= scale
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    return np.column_stack([u, v])

def get_min_area(bary):
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return float(np.min(areas))

def hard_min_obj(bary_flat):
    bary = clamp_bary(bary_flat.reshape(N, 2))
    return -np.min(calc_areas_bary(bary))

def soft_min_obj(bary_flat, beta=200.0):
    bary = clamp_bary(bary_flat.reshape(N, 2))
    areas = calc_areas_bary(bary)
    if np.any(areas < 1e-15):
        return 1e10
    scaled = -beta * areas
    mx = np.max(scaled)
    if mx > 700:
        return -np.min(areas)
    log_sum = mx + np.log(np.sum(np.exp(scaled - mx)))
    return -(-log_sum / beta)

def solve_active_qp(bary, active_set, t_init, max_iter=8, tol=1e-12):
    """
    Max-min via active-set QP.
    Variables: p (2N bary coords) + t (min area).
    Maximize t subject to:
      area_i(p) >= t  for i in active_set (linearized around current p)
      0 <= u_i, v_i, u_i+v_i <= 1
    We solve with SLSQP on the augmented objective.
    """
    p = bary.flatten().copy()
    t = t_init
    n = 2 * N + 1

    def qp_obj(x):
        p_ = x[:2*N]
        t_ = x[2*N]
        bary_ = clamp_bary(p_.reshape(N, 2))
        areas = calc_areas_bary(bary_)
        # penalty for active constraints violated
        viol = np.maximum(0.0, t_ - areas[active_set])
        return -t_ + 1e4 * np.sum(viol**2)

    def qp_grad(x):
        p_ = x[:2*N]
        t_ = x[2*N]
        bary_ = clamp_bary(p_.reshape(N, 2))
        u = bary_[:, 0]; v = bary_[:, 1]; w = 1 - u - v
        # compute area gradient for active triangles
        g = np.zeros(n)
        areas = calc_areas_bary(bary_)
        for k, idx in enumerate(active_set):
            a, b, c = tri_idx[idx]
            # area = |det|; assume sign fixed (use signed det)
            det = (u[a]*(v[b]*w[c]-v[c]*w[b]) - v[a]*(u[b]*w[c]-u[c]*w[b]) + w[a]*(u[b]*v[c]-u[c]*v[b]))
            sign = 1.0 if det >= 0 else -1.0
            # gradient of det w.r.t. u[a],v[a],u[b],v[b],u[c],v[c]
            # d(det)/du[a] = v[b]*w[c] - v[c]*w[b]
            # d(det)/dv[a] = u[c]*w[b] - u[b]*w[c]
            # d(det)/du[b] = v[c]*w[a] - v[a]*w[c]
            # d(det)/dv[b] = u[a]*w[c] - u[c]*w[a]
            # d(det)/du[c] = v[a]*w[b] - v[b]*w[a]
            # d(det)/dv[c] = u[b]*w[a] - u[a]*w[b]
            grads = sign * np.array([
                v[b]*w[c] - v[c]*w[b],
                u[c]*w[b] - u[b]*w[c],
                v[c]*w[a] - v[a]*w[c],
                u[a]*w[c] - u[c]*w[a],
                v[a]*w[b] - v[b]*w[a],
                u[b]*w[a] - u[a]*w[b],
            ])
            g[2*a]   += grads[0]
            g[2*a+1] += grads[1]
            g[2*b]   += grads[2]
            g[2*b+1] += grads[3]
            g[2*c]   += grads[4]
            g[2*c+1] += grads[5]
        g[2*N] = 1.0
        # penalty gradient
        for k, idx in enumerate(active_set):
            viol = t - areas[idx]
            if viol > 0:
                g[2*N] += 2e4 * viol
        return g

    bounds = [(0, 1)] * (2*N) + [(0, 1.0)]

    x0 = np.concatenate([p, [t]])
    best_x = x0.copy()
    best_val = qp_obj(x0)

    for it in range(max_iter):
        try:
            res = minimize(qp_obj, best_x, jac=qp_grad, method='SLSQP',
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-14})
            if res.x is not None:
                cand_val = qp_obj(res.x)
                if cand_val < best_val:
                    best_val = cand_val
                    best_x = res.x.copy()
        except Exception:
            break
        # check convergence of t
        if it > 0 and abs(best_x[2*N] - x0[2*N]) < tol:
            break
        x0 = best_x.copy()

    p_new = clamp_bary(best_x[:2*N].reshape(N, 2))
    return p_new, float(np.min(calc_areas_bary(p_new)))

def active_set_solve(bary, tol=1e-6, max_rounds=6):
    """Iterative active-set: find triangles near min, solve QP, repeat."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    t = float(np.min(areas))
    for r in range(max_rounds):
        active = np.where(areas <= t * (1 + tol) + 1e-12)[0]
        if len(active) == 0:
            active = np.array([np.argmin(areas)])
        # cap active set size
        if len(active) > 30:
            active = active[np.argsort(areas[active])[:30]]
        new_bary, new_t = solve_active_qp(bary, active, t, max_iter=5)
        if new_t <= t * (1 - 1e-10):
            break
        bary = new_bary
        areas = calc_areas_bary(bary)
        t = float(np.min(areas))
    return bary, t

def refine_local(bary_init, time_limit=5.0):
    start = time.time()
    bary = clamp_bary(bary_init.copy())
    val = get_min_area(bary)

    # Stage 1: soft-min L-BFGS-B to get into a good basin
    try:
        res = minimize(soft_min_obj, bary.flatten(), args=(200.0,),
                       method='L-BFGS-B', bounds=[(0,1)]*(2*N),
                       options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10})
        nb = clamp_bary(res.x.reshape(N, 2))
        nv = get_min_area(nb)
        if nv > val:
            bary, val = nb, nv
    except Exception:
        pass
    if time.time() - start > time_limit * 0.35:
        return bary

    # Stage 2: active-set QP refinement (the structural core)
    try:
        bary, val2 = active_set_solve(bary, tol=1e-5, max_rounds=8)
        if val2 > val:
            bary, val = bary, val2
    except Exception:
        pass
    if time.time() - start > time_limit * 0.7:
        return bary

    # Stage 3: coordinate ascent on single points
    try:
        for _ in range(2):
            for i in range(N):
                if time.time() - start > time_limit * 0.9:
                    break
                def sp_obj(xi):
                    bt = bary.copy()
                    bt[i] = xi
                    bt = clamp_bary(bt)
                    return -np.min(calc_areas_bary(bt))
                res_i = minimize(sp_obj, bary[i], method='Nelder-Mead',
                                 options={'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                nb = bary.copy()
                nb[i] = clamp_bary(res_i.x.reshape(1, 2))[0]
                nv = get_min_area(nb)
                if nv > val:
                    bary, val = nb, nv
    except Exception:
        pass

    # Stage 4: final active-set polish
    try:
        bary, val3 = active_set_solve(bary, tol=1e-7, max_rounds=5)
        if val3 > val:
            bary, val = bary, val3
    except Exception:
        pass

    return bary

def gen_hexagonal():
    pts = []
    for i in range(4):
        pts.append([i/3.0, 0.0])
    for i in range(3):
        pts.append([(i+0.5)/3.0, H/3.0])
    for i in range(2):
        pts.append([(i+1.0)/3.0, 2*H/3.0])
    pts.append([0.5, H])
    return [np.array(pts)]

def gen_k3_lattice():
    K = 3
    pts = []
    for i in range(K+1):
        for j in range(K+1-i):
            pts.append([i/K, j/K])
    return [np.array(pts)]

def gen_k4_lattice_minus():
    K = 4
    pts = []
    for i in range(K+1):
        for j in range(K+1-i):
            pts.append([i/K, j/K])
    pts = np.array(pts)
    interior = [i for i, p in enumerate(pts) if p[0] > 0.01 and p[1] > 0.01 and (1-p[0]-p[1]) > 0.01]
    remove = set(interior[:4])
    keep = [i for i in range(len(pts)) if i not in remove]
    return [pts[keep]]

def gen_symmetric():
    cands = []
    corners = np.array([[0,0],[1,0],[0.5,H]])
    # 3 corners + 6 edge + 2 interior
    edge = np.array([[0.25,0],[0.75,0],[0.125,H*0.5],[0.375,H*0.75],[0.625,H*0.75],[0.875,H*0.5]])
    interior = np.array([[1/3,1/3],[0.4,0.4]])
    cands.append(np.vstack([corners, edge, interior]))
    # 3 corners + 4 edge + 4 interior
    edge2 = np.array([[0.2,0],[0.8,0],[0.1,H*0.4],[0.35,H*0.7]])
    interior2 = np.array([[1/3,1/3],[0.2,0.3],[0.5,0.2],[0.4,0.4]])
    cands.append(np.vstack([corners, edge2, interior2]))
    return cands

def gen_random(rng, n=5):
    cands = []
    for _ in range(n):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        cands.append(np.column_stack([u, v]))
    return cands

def gen_perturbed(base, rng, n=3, scale=0.03):
    cands = []
    for _ in range(n):
        p = base + rng.normal(0, scale, size=base.shape)
        cands.append(clamp_bary(p))
    return cands

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT
    best_bary = None
    best_val = -np.inf

    def update(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    candidates = []
    candidates.extend(gen_hexagonal())
    candidates.extend(gen_k3_lattice())
    candidates.extend(gen_k4_lattice_minus())
    candidates.extend(gen_symmetric())
    rng = np.random.default_rng(42)
    candidates.extend(gen_random(rng, n=6))

    n_cands = len(candidates)
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        remaining = deadline - 30 - time.time()
        if remaining < 4:
            break
        if i < 3:
            tl = min(8.0, remaining)
        elif i < 8:
            tl = min(5.0, remaining)
        else:
            tl = min(3.5, remaining)
        try:
            refined = refine_local(cand, time_limit=tl)
            update(refined)
        except Exception:
            continue

    # Perturbed best
    if best_bary is not None and time.time() < deadline - 35:
        rng2 = np.random.default_rng(999)
        for pc in gen_perturbed(best_bary, rng2, n=3, scale=0.02):
            if time.time() > deadline - 30:
                break
            remaining = deadline - 30 - time.time()
            if remaining < 4:
                break
            try:
                update(refine_local(pc, time_limit=min(6.0, remaining)))
            except Exception:
                continue

    # Global DE search
    if time.time() < deadline - 45:
        try:
            def de_obj(x):
                return -np.min(calc_areas_bary(clamp_bary(x.reshape(N, 2))))
            res_de = differential_evolution(de_obj, [(0,1)]*(2*N), seed=42,
                                            maxiter=30, popsize=14, tol=1e-6, polish=False)
            de_bary = clamp_bary(res_de.x.reshape(N, 2))
            remaining = deadline - 28 - time.time()
            if remaining > 6:
                update(refine_local(de_bary, time_limit=min(10.0, remaining)))
        except Exception:
            pass

    # Second DE
    if time.time() < deadline - 35:
        try:
            def de_obj(x):
                return -np.min(calc_areas_bary(clamp_bary(x.reshape(N, 2))))
            res_de2 = differential_evolution(de_obj, [(0,1)]*(2*N), seed=7,
                                             maxiter=20, popsize=12, tol=1e-6, polish=False)
            de_bary2 = clamp_bary(res_de2.x.reshape(N, 2))
            remaining = deadline - 25 - time.time()
            if remaining > 5:
                update(refine_local(de_bary2, time_limit=min(8.0, remaining)))
        except Exception:
            pass

    # Final polish
    if best_bary is not None and time.time() < deadline - 6:
        try:
            bary, val = active_set_solve(best_bary, tol=1e-8, max_rounds=6)
            update(bary)
        except Exception:
            pass
        if time.time() < deadline - 3:
            try:
                res = minimize(hard_min_obj, best_bary.flatten(), method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True})
                update(clamp_bary(res.x.reshape(N, 2)))
            except Exception:
                pass

    if best_bary is None:
        best_bary = gen_hexagonal()[0]

    return bary_to_cart(best_bary)
