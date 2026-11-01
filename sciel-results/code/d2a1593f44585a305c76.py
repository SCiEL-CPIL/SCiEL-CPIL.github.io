"""Combined: force-directed adjacency layout + rigid composite optimization + polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side(inner, outer_center, outer_angle_deg):
    """Compute minimum outer side length to contain all inner hexagons."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        cur = float(projs.max())
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """Smooth log-sum-exp surrogate for the max-projection objective."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        all_projs.append(projs)
    P = np.concatenate([p.ravel() for p in all_projs])
    if P.size == 0:
        return 0.0
    m = float(P.max())
    if m < -1e6:
        return 0.0
    smooth_max = m + math.log(float(np.sum(np.exp(beta * (P - m))))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (arrays of (x,y))."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = (p2[0] - p1[0], p2[1] - p1[1])
            n = (-e[1], e[0])
            ln = math.hypot(n[0], n[1])
            if ln > 1e-12:
                axes.append((n[0] / ln, n[1] / ln))
    for ax in axes:
        min1 = min(float(p[0] * ax[0] + p[1] * ax[1]) for p in v1)
        max1 = max(float(p[0] * ax[0] + p[1] * ax[1]) for p in v1)
        min2 = min(float(p[0] * ax[0] + p[1] * ax[1]) for p in v2)
        max2 = max(float(p[0] * ax[0] + p[1] * ax[1]) for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = [hex_vertices(x, y, 1.0, math.radians(a)) for (x, y, a) in inner]
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def overlap_penalty(inner, min_dist=SQRT3 - 0.002):
    """Quadratic penalty when center distances fall below min_dist."""
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

def build_lattice_config(axial_coords, s=1.0 + 1e-8):
    """Build inner hexagons from axial coordinates with spacing s."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + q / 2.0)) for q, r in axial_coords]
    return [[x, y, 0.0] for x, y in centers]

def force_directed_layout(axial_coords, deadline, max_iter=3000):
    """
    Force-directed layout over the adjacency graph of the 11 hexagons.
    - Repulsive potential pushes apart any pair closer than SQRT3.
    - Spring (contact) constraints pull adjacent pairs toward SQRT3.
    - A weak radial term keeps the cluster compact (minimizes diameter).
    Returns list of [x, y, 0.0].
    """
    n = len(axial_coords)
    # Build adjacency: pairs that are lattice-neighbors (distance ~ SQRT3 at s=1)
    base = build_lattice_config(axial_coords, s=1.0)
    adj = []
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(base[i][0] - base[j][0], base[i][1] - base[j][1])
            if d < 1.6 * SQRT3:  # lattice neighbors are ~SQRT3 apart
                adj.append((i, j))
    
    pos = np.array([[base[i][0], base[i][1]] for i in range(n)], dtype=float)
    cx0 = float(pos[:, 0].mean())
    cy0 = float(pos[:, 1].mean())
    
    min_dist = SQRT3
    k_rep = 50.0
    k_spring = 20.0
    k_radial = 0.05
    lr = 0.02
    cooldown = 0.999
    
    for it in range(max_iter):
        if time.time() > deadline - 30:
            break
        grad = np.zeros_like(pos)
        # Repulsion (all pairs)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = math.hypot(dx, dy)
                if d < 1e-9:
                    dx, dy = 1e-5, 0.0
                    d = 1e-5
                if d < min_dist:
                    f = k_rep * (min_dist - d)
                    fx = f * dx / d
                    fy = f * dy / d
                    grad[i, 0] += fx
                    grad[i, 1] += fy
                    grad[j, 0] -= fx
                    grad[j, 1] -= fy
        # Springs (adjacent pairs toward min_dist)
        for (i, j) in adj:
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            d = math.hypot(dx, dy)
            if d < 1e-9:
                continue
            f = k_spring * (d - min_dist)
            fx = f * dx / d
            fy = f * dy / d
            grad[i, 0] += fx
            grad[i, 1] += fy
            grad[j, 0] -= fx
            grad[j, 1] -= fy
        # Radial compaction toward centroid
        for i in range(n):
            grad[i, 0] += k_radial * (pos[i, 0] - cx0)
            grad[i, 1] += k_radial * (pos[i, 1] - cy0)
        
        pos -= lr * grad
        lr *= cooldown
    
    return [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in range(n)]

def optimize_rigid_composite(base_positions, deadline):
    """
    Treat the 11 hexagons as a rigid composite. Optimize:
    composite_angle, dx, dy, outer_cx, outer_cy, outer_angle (6 DOF).
    Returns (best_inner, best_outer_center, best_outer_angle, best_side).
    """
    cx0 = sum(p[0] for p in base_positions) / len(base_positions)
    cy0 = sum(p[1] for p in base_positions) / len(base_positions)
    
    def make_inner(p):
        comp_angle = p[0]
        dx, dy = p[1], p[2]
        ca = math.cos(math.radians(comp_angle))
        sa = math.sin(math.radians(comp_angle))
        inner = []
        for i in range(N):
            bx, by, ba = base_positions[i]
            rx = bx - cx0
            ry = by - cy0
            nx = cx0 + ca * rx - sa * ry + dx
            ny = cy0 + sa * rx + ca * ry + dy
            inner.append([nx, ny, ba + comp_angle])
        return inner
    
    def obj_smooth(p):
        inner = make_inner(p)
        return compute_required_side_smooth(inner, (p[3], p[4]), p[5], beta=50.0)
    
    def obj_exact(p):
        inner = make_inner(p)
        return compute_required_side(inner, (p[3], p[4]), p[5])
    
    best_val = float('inf')
    best_params = None
    
    # Coarse angle sweep for composite and outer
    for comp_init in [0, 15, 30, 45]:
        if time.time() > deadline - 25:
            break
        for outer_init in [0, 15, 30, 45, 60]:
            init_params = [float(comp_init), 0.0, 0.0, cx0, cy0, float(outer_init)]
            try:
                res = minimize(obj_smooth, init_params, method='Nelder-Mead',
                               options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 1500})
                res2 = minimize(obj_exact, res.x, method='Nelder-Mead',
                                options={'maxiter': 1500, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 3000})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x
            except Exception:
                pass
    
    if best_params is None:
        best_params = np.array([0.0, 0.0, 0.0, cx0, cy0, 0.0])
        best_val = obj_exact(best_params)
    
    best_inner = make_inner(best_params)
    best_outer_center = (float(best_params[3]), float(best_params[4]))
    best_outer_angle = float(best_params[5])
    best_side = float(best_val)
    return best_inner, best_outer_center, best_outer_angle, best_side

def joint_refine(inner, outer_center, outer_angle, deadline):
    """
    Joint refinement of all inner positions + outer (36 DOF) using SA with
    the smooth objective + overlap penalty, then exact polish.
    """
    def obj_joint(params):
        inner_list = []
        for i in range(N):
            inner_list.append([params[i*3], params[i*3+1], params[i*3+2]])
        side = compute_required_side_smooth(inner_list, (params[N*3], params[N*3+1]), params[N*3+2], beta=30.0)
        penalty = overlap_penalty(inner_list)
        return side + 1000.0 * penalty
    
    current = np.array([v for h in inner for v in h] + [outer_center[0], outer_center[1], outer_angle], dtype=float)
    current_val = obj_joint(current)
    best = current.copy()
    best_val = current_val
    
    T = 0.4
    T_min = 0.001
    cooling = 0.9996
    max_iter = 12000
    for i in range(max_iter):
        if time.time() > deadline - 15:
            break
        neighbor = current.copy()
        step_scale = 0.3 * (T / T_min) ** 0.3
        if np.random.random() < 0.75:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 3)
            if dim < 2:
                delta = np.random.normal(0, 0.12 * step_scale)
            else:
                delta = np.random.normal(0, 2.0 * step_scale)
            neighbor[idx*3 + dim] += delta
        else:
            dim = np.random.randint(N*3, N*3+3)
            if dim == N*3+2:
                delta = np.random.normal(0, 2.0 * step_scale)
            else:
                delta = np.random.normal(0, 0.08 * step_scale)
            neighbor[dim] += delta
        nv = obj_joint(neighbor)
        if nv < current_val or np.random.random() < math.exp(-(nv - current_val) / T):
            current = neighbor
            current_val = nv
            if nv < best_val:
                best = current.copy()
                best_val = nv
        T *= cooling
        if T < T_min:
            break
    
    # Exact polish of best
    def obj_exact_joint(params):
        inner_list = []
        for i in range(N):
            inner_list.append([params[i*3], params[i*3+1], params[i*3+2]])
        side = compute_required_side(inner_list, (params[N*3], params[N*3+1]), params[N*3+2])
        penalty = overlap_penalty(inner_list)
        return side + 1000.0 * penalty
    
    try:
        res = minimize(obj_exact_joint, best, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 4000})
        if res.fun < best_val:
            best = res.x
            best_val = res.fun
    except Exception:
        pass
    
    best_inner = [[float(best[i*3]), float(best[i*3+1]), float(best[i*3+2])] for i in range(N)]
    best_outer_center = (float(best[N*3]), float(best[N*3+1]))
    best_outer_angle = float(best[N*3+2])
    return best_inner, best_outer_center, best_outer_angle

def final_polish_outer(inner, outer_center, outer_angle, deadline):
    """High-precision polish of the outer container only."""
    def obj(p):
        return compute_required_side(inner, (p[0], p[1]), p[2])
    best_val = compute_required_side(inner, outer_center, outer_angle)
    best_p = np.array([outer_center[0], outer_center[1], outer_angle])
    starts = [
        [outer_center[0], outer_center[1], outer_angle],
        [outer_center[0], outer_center[1], outer_angle + 1],
        [outer_center[0], outer_center[1], outer_angle - 1],
        [outer_center[0], outer_center[1], outer_angle + 0.5],
    ]
    for ps in starts:
        if time.time() > deadline - 2:
            break
        try:
            res = minimize(obj, ps, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 6000})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x
        except Exception:
            pass
    return (float(best_p[0]), float(best_p[1])), float(best_p[2]), float(best_val)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial layout: 11 hexagons in 3-4-4 arrangement (center + ring1 + partial ring2)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # Phase A: force-directed layout to find tight equilibrium positions
    fd_inner = force_directed_layout(axial_coords, deadline, max_iter=3000)
    
    # Phase B: rigid composite optimization (global orientation + outer)
    best_inner, best_outer_center, best_outer_angle, best_side = optimize_rigid_composite(fd_inner, deadline)
    
    # Phase C: joint refinement (all positions + outer) via SA + exact polish
    if time.time() < deadline - 20:
        jr_inner, jr_center, jr_angle = joint_refine(best_inner, best_outer_center, best_outer_angle, deadline)
        if check_disjoint(jr_inner, eps=1e-5):
            side_jr = compute_required_side(jr_inner, jr_center, jr_angle)
            if side_jr < best_side:
                best_inner, best_outer_center, best_outer_angle, best_side = jr_inner, jr_center, jr_angle, side_jr
    
    # Phase D: final high-precision outer polish
    if time.time() < deadline - 3:
        pc, pa, pv = final_polish_outer(best_inner, best_outer_center, best_outer_angle, deadline)
        best_outer_center, best_outer_angle, best_side = pc, pa, pv
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
