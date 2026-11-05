import numpy as np
import time
import itertools
from scipy.optimize import minimize

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    if a < 1e-12:
        t = f / e
        t = max(0, min(1, t))
        return np.linalg.norm(p1 - (p3 + t*d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        s = c / a
        s = max(0, min(1, s))
        return np.linalg.norm((p1 + s*d1) - p3)
    b = np.dot(d1, d2)
    denom = a*e - b*b
    if denom > 1e-12:
        s = (b*f - c*e)/denom
        s = max(0, min(1, s))
        t = (b*s + f)/e
        if t < 0:
            t = 0
            s = max(0, min(1, c/a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b+f)/a))
        return np.linalg.norm((p1 + s*d1) - (p3 + t*d2))
    else:
        t = f/e
        t = max(0, min(1, t))
        return np.linalg.norm(p1 - (p3 + t*d2))

def _poly_separation(p1, p2):
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    min_dist = float('inf')
    n1 = len(p1); n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]; a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]; b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _build_topology(ring2_angles, eps=1e-9):
    """Build centers for one discrete topology. Returns (11,2) array."""
    centers = [np.array([0.0, 0.0])]
    r1 = np.sqrt(3.0) + eps
    for k in range(6):
        a = np.deg2rad(30 + k*60)
        centers.append(r1*np.array([np.cos(a), np.sin(a)]))
    r2 = 3.0 + 2*eps
    for ad in ring2_angles:
        a = np.deg2rad(ad)
        centers.append(r2*np.array([np.cos(a), np.sin(a)]))
    return np.array(centers)

def _base_vertices(angle_deg=0.0):
    a = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    return np.column_stack([np.cos(a), np.sin(a)])

def _side_for_theta(centers, theta, base):
    """Exact enclosing side for a rigid config at outer rotation theta (rad)."""
    normals = np.column_stack([
        np.cos(np.deg2rad(30 + np.array([0, 60, 120, 180, 240, 300])) + theta),
        np.sin(np.deg2rad(30 + np.array([0, 60, 120, 180, 240, 300])) + theta),
    ])
    # verts: (11,6,2)
    verts = centers[:, None, :] + base[None, :, :]
    # project onto each normal
    proj = np.einsum('ijk,kj->ij', verts, normals.T)  # (11,6)
    max_proj = np.max(proj)
    return 2.0 * max_proj / np.sqrt(3.0)

def _critical_angles(centers, base):
    """Candidate outer-theta values where a vertex aligns with a normal (extrema)."""
    cand = set()
    # normals at theta: angles 30+60k + theta. vertex angle (global) = v_ang + theta_v? base fixed.
    # A vertex global angle = base_angle. Normal angle = 30+60k+theta.
    # alignment when base_angle - (30+60k+theta) = 0 mod 180 -> theta = base_angle - 30 - 60k (mod 180)
    base_ang = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    for ba in base_ang:
        for k in range(6):
            th = ba - np.deg2rad(30 + 60*k)
            cand.add(th)
    # also offset by each center? alignment depends only on angle, not position.
    # reduce to [0, pi/6)
    out = []
    for th in cand:
        th = th % (np.pi/6)
        out.append(th)
    return out

def _exact_min_theta(centers, base, deadline):
    """Find min enclosing side over outer rotation theta in [0, pi/6)."""
    # coarse grid
    grid = np.linspace(0.0, np.pi/6, 400)
    vals = np.array([_side_for_theta(centers, t, base) for t in grid])
    i0 = int(np.argmin(vals))
    t0 = grid[i0]
    # golden-section refine in neighborhood
    lo = max(0.0, grid[max(0, i0-1)])
    hi = min(np.pi/6, grid[min(len(grid)-1, i0+1)])
    if hi - lo < 1e-12:
        return t0, vals[i0]
    gr = (np.sqrt(5.0)-1.0)/2.0
    a = lo; b = hi
    c = b - gr*(b-a); d = a + gr*(b-a)
    fc = _side_for_theta(centers, c, base); fd = _side_for_theta(centers, d, base)
    it = 0
    while b - a > 1e-12 and it < 60 and time.time() < deadline - 2:
        if fc < fd:
            b = d; d = c; fd = fc
            c = b - gr*(b-a); fc = _side_for_theta(centers, c, base)
        else:
            a = c; c = d; fc = fd
            d = a + gr*(b-a); fd = _side_for_theta(centers, d, base)
        it += 1
    t_star = 0.5*(a+b)
    s_star = _side_for_theta(centers, t_star, base)
    return t_star, s_star

def _smooth_side_and_penalty(params, n, outer_angle_deg=0.0, beta=60.0):
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    xs = params[:, 0]; ys = params[:, 1]
    angles_rad = np.deg2rad(params[:, 2])
    base_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    base_cos = np.cos(base_angles); base_sin = np.sin(base_angles)
    cos_a = np.cos(angles_rad)[:, np.newaxis]
    sin_a = np.sin(angles_rad)[:, np.newaxis]
    vx = base_cos*cos_a - base_sin*sin_a + xs[:, np.newaxis]
    vy = base_sin*cos_a + base_cos*sin_a + ys[:, np.newaxis]
    proj_all = vx[:, :, np.newaxis]*normals[np.newaxis, :, 0] + vy[:, :, np.newaxis]*normals[np.newaxis, :, 1]
    pmj = np.max(proj_all, axis=1)
    p_ik = (np.log(np.sum(np.exp(beta*(proj_all - pmj[:, np.newaxis, :])), axis=1)) + beta*pmj)/beta
    pmi = np.max(p_ik, axis=0)
    H_k = (np.log(np.sum(np.exp(beta*(p_ik - pmi[np.newaxis, :])), axis=0)) + beta*pmi)/beta
    hmax = np.max(H_k)
    S = (np.log(np.sum(np.exp(beta*(H_k - hmax)))) + beta*hmax)/beta
    side = 2.0*S/np.sqrt(3.0)
    penalty = 0.0
    centers = params[:, :2]
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    for i in range(n):
        for j in range(i+1, n):
            if dists[i, j] < 2.0:
                v1 = np.column_stack([vx[i], vy[i]])
                v2 = np.column_stack([vx[j], vy[j]])
                sep = _poly_separation(v1, v2)
                if sep < 0.01:
                    penalty += 100.0*(0.01 - sep)**2
    return side, penalty

def _objective(params, n):
    side, pen = _smooth_side_and_penalty(params, n)
    return side + pen

def _fast_side_and_penalty(params, n, outer_angle_deg=0.0):
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    xs = params[:, 0]; ys = params[:, 1]
    angles_rad = np.deg2rad(params[:, 2])
    base_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    base_cos = np.cos(base_angles); base_sin = np.sin(base_angles)
    cos_a = np.cos(angles_rad)[:, np.newaxis]
    sin_a = np.sin(angles_rad)[:, np.newaxis]
    vx = base_cos*cos_a - base_sin*sin_a + xs[:, np.newaxis]
    vy = base_sin*cos_a + base_cos*sin_a + ys[:, np.newaxis]
    proj = vx @ normals.T + vy @ normals.T
    max_proj = np.max(proj, axis=0)
    side = 2.0*np.max(max_proj)/np.sqrt(3.0)
    penalty = 0.0
    centers = params[:, :2]
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    for i in range(n):
        for j in range(i+1, n):
            if dists[i, j] < 2.0:
                v1 = np.column_stack([vx[i], vy[i]])
                v2 = np.column_stack([vx[j], vy[j]])
                sep = _poly_separation(v1, v2)
                if sep < 0.01:
                    penalty += 100.0*(0.01 - sep)**2
    return side, penalty

def _is_valid(params, side, n, min_sep=1e-8, outer_angle_deg=0.0):
    side_check, _ = _fast_side_and_penalty(params, n, outer_angle_deg)
    if side_check > side + 1e-9:
        return False
    centers = params[:, :2]
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    for i in range(n):
        for j in range(i+1, n):
            if dists[i, j] < 2.0:
                v1 = _hex_vertices(params[i, 0], params[i, 1], params[i, 2])
                v2 = _hex_vertices(params[j, 0], params[j, 1], params[j, 2])
                sep = _poly_separation(v1, v2)
                if sep < min_sep:
                    return False
    return True

def _seed_solution():
    eps = 1e-9
    centers = [(0.0, 0.0)]
    for k in range(6):
        a = np.deg2rad(k*60)
        centers.append(((np.sqrt(3)+eps)*np.cos(a), (np.sqrt(3)+eps)*np.sin(a)))
    for k in range(4):
        a = np.deg2rad(30 + k*60)
        centers.append(((3.0+2*eps)*np.cos(a), (3.0+2*eps)*np.sin(a)))
    inner = [[x, y, 30.0] for x, y in centers]
    max_x = (2+eps)*np.sqrt(3)
    max_y = 4 + 2*eps
    side = max(2*max_x/np.sqrt(3), max_y) + 1e-9
    return inner, (0.0, 0.0), side, 30.0

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    n = 11

    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = seed_inner
    best_side = seed_side
    best_outer_angle = 30.0
    best_valid = True

    base = _base_vertices(0.0)  # inner hexagons at angle 0 (rigid)
    all_angles = [0, 60, 120, 180, 240, 300]
    combinations = list(itertools.combinations(all_angles, 4))

    # Phase A: discrete topology enumeration with exact outer-rotation solve
    for combo in combinations:
        if time.time() > DEADLINE - 15:
            break
        centers = _build_topology(combo, eps=1e-9)
        t_star, s_star = _exact_min_theta(centers, base, DEADLINE)
        if s_star < best_side:
            best_side = s_star
            best_outer_angle = np.rad2deg(t_star)
            best_inner = [[float(centers[i, 0]), float(centers[i, 1]), 0.0] for i in range(n)]
            best_valid = True

    # Phase B: smooth continuous optimization starting from best rigid config
    init_params = np.zeros((n, 3))
    for i in range(n):
        init_params[i] = [best_inner[i][0], best_inner[i][1], best_inner[i][2]]

    try:
        result = minimize(
            _objective, init_params.flatten(), args=(n,),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True},
        )
        if time.time() < DEADLINE - 10:
            opt = result.x.reshape(n, 3)
            s, p = _fast_side_and_penalty(opt, n)
            if p < 1e-6 and _is_valid(opt, s, n):
                if s < best_side:
                    best_side = s
                    best_inner = [[float(opt[i, 0]), float(opt[i, 1]), float(opt[i, 2])] for i in range(n)]
                    best_outer_angle = 0.0
                    best_valid = True
    except Exception:
        pass

    # Phase C: final high-precision polish
    if best_valid and time.time() < DEADLINE - 4:
        try:
            pp = np.array(best_inner, dtype=float)
            res = minimize(
                _objective, pp.flatten(), args=(n,),
                method='L-BFGS-B',
                bounds=[(-6, 6), (-6, 6), (0, 180)]*n,
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14},
            )
            pp = res.x.reshape(n, 3)
            s, p = _fast_side_and_penalty(pp, n)
            if p < 1e-8 and _is_valid(pp, s, n):
                if s < best_side:
                    best_side = s
                    best_inner = [[float(pp[i, 0]), float(pp[i, 1]), float(pp[i, 2])] for i in range(n)]
                    best_outer_angle = 0.0
        except Exception:
            pass

    # safety
    if time.time() > DEADLINE - 2:
        return best_inner, (0.0, 0.0), best_side, best_outer_angle

    return best_inner, (0.0, 0.0), best_side, best_outer_angle
