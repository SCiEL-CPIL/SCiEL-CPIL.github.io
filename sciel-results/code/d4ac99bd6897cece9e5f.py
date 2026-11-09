import numpy as np
import time
from scipy.optimize import minimize
import warnings

N = 11
DEADLINE = 145.0

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon (flat-top orientation at angle=0)."""
    angle = np.radians(angle_deg)
    thetas = angle + np.pi * np.arange(6) / 3.0
    return np.column_stack([
        cx + side * np.cos(thetas),
        cy + side * np.sin(thetas)
    ])

def _hex_edge_normals(angle_deg):
    """6 outward edge normals of a regular hexagon."""
    angle = np.radians(angle_deg)
    mid = angle + np.pi / 6.0 + np.pi * np.arange(6) / 3.0
    return np.column_stack([np.cos(mid), np.sin(mid)])

def _outer_side_direct(centers, inner_angles, outer_angle_deg):
    """
    For a fixed outer hexagon orientation, compute the minimum side length
    and center that enclose all inner hexagons. Uses the LP solution in closed form.
    Returns (side, center).
    """
    normals = _hex_edge_normals(outer_angle_deg)
    all_pts = np.vstack([
        _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for i in range(N)
    ])
    proj = all_pts @ normals.T  # (n_pts, 6)
    A = proj.max(axis=0)  # max projection per direction

    # LP: min r s.t. n_i·c - r <= -A_i for all i
    # Optimal: r = max_i (n_i·c + A_i) minimized over c
    # This is a convex piecewise-linear function of c.
    # Use scipy linprog for robustness.
    c_obj = np.array([0.0, 0.0, 1.0])
    A_mat = np.zeros((6, 3))
    b_vec = np.zeros(6)
    for i in range(6):
        A_mat[i, 0] = normals[i, 0]
        A_mat[i, 1] = normals[i, 1]
        A_mat[i, 2] = -1.0
        b_vec[i] = -A[i]
    
    from scipy.optimize import linprog
    res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
    if res.status != 0:
        return float('inf'), np.array([0.0, 0.0])
    
    cx, cy, r = res.x
    side = 2 * r / np.sqrt(3)
    return side, np.array([cx, cy])

def _optimize_outer(centers, inner_angles, max_angle_steps=20):
    """Find optimal outer hexagon orientation and side."""
    best_side = float('inf')
    best_center = np.array([0.0, 0.0])
    best_angle = 0.0
    
    # Coarse sweep
    for oa in np.linspace(0, 60, max_angle_steps + 1):
        side, center = _outer_side_direct(centers, inner_angles, oa)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = oa
    
    # Fine refinement
    if np.isfinite(best_side):
        for da in np.linspace(-1.5, 1.5, 7):
            ta = best_angle + da
            side, center = _outer_side_direct(centers, inner_angles, ta)
            if side < best_side:
                best_side = side
                best_center = center
                best_angle = ta
    
    return best_side, best_center, best_angle

def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """SAT overlap check between two unit hexagons."""
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            ex = verts[(i+1)%n, 0] - verts[i, 0]
            ey = verts[(i+1)%n, 1] - verts[i, 1]
            nx, ny = -ey, ex
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append((nx/norm, ny/norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        p1 = v1[:,0]*ax + v1[:,1]*ay
        p2 = v2[:,0]*ax + v2[:,1]*ay
        if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
            return False
    return True

def _overlap_penalty(centers, angles, soft_beta=50.0):
    """Smooth sum of overlaps between all pairs."""
    total = 0.0
    for i in range(N):
        for j in range(i+1, N):
            v1 = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            v2 = _hex_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
            
            def get_axes(verts):
                axes = []
                for k in range(6):
                    ex = verts[(k+1)%6, 0] - verts[k, 0]
                    ey = verts[(k+1)%6, 1] - verts[k, 1]
                    nx, ny = -ey, ex
                    norm = np.sqrt(nx*nx + ny*ny)
                    if norm > 1e-12:
                        axes.append((nx/norm, ny/norm))
                return axes
            
            axes = get_axes(v1) + get_axes(v2)
            max_ov = 0.0
            for ax, ay in axes:
                p1 = v1[:,0]*ax + v1[:,1]*ay
                p2 = v2[:,0]*ax + v2[:,1]*ay
                ov = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
                if ov > max_ov:
                    max_ov = ov
            # Smooth penalty
            total += np.log(1.0 + np.exp(soft_beta * max_ov)) / soft_beta
    return total

def _smooth_cost(centers, angles, outer_angle_deg, lam=100.0):
    """Cost = outer_side + lam * overlap_penalty."""
    side, _ = _outer_side_direct(centers, angles, outer_angle_deg)
    if not np.isfinite(side):
        return 1e10
    pen = _overlap_penalty(centers, angles)
    return side + lam * pen

def _contact_graph_optimize(centers, angles, max_time=30.0, outer_angle=0.0):
    """Nelder-Mead optimization on positions and angles."""
    start = time.time()
    
    # Flatten: [x0,y0,ang0, x1,y1,ang1, ...]
    def flatten(cs, as_):
        v = []
        for i in range(N):
            v.extend([cs[i][0], cs[i][1], as_[i]])
        return np.array(v)
    
    def unflatten(v):
        cs = []
        as_ = []
        for i in range(N):
            cs.append((v[3*i], v[3*i+1]))
            as_.append(v[3*i+2])
        return cs, as_
    
    def obj(v):
        if time.time() - start > max_time:
            return 1e10
        cs, as_ = unflatten(v)
        return _smooth_cost(cs, as_, outer_angle, lam=50.0)
    
    x0 = flatten(centers, angles)
    res = minimize(obj, x0, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
    
    if time.time() - start > max_time:
        return centers, angles, float('inf'), None, None
    
    best_cs, best_as = unflatten(res.x)
    side, center, oa = _optimize_outer(best_cs, best_as, max_angle_steps=15)
    return best_cs, best_as, side, center, oa

def _force_directed_relax(centers, angles, max_iter=2000, tol=1e-6):
    """Push apart overlapping hexagons."""
    centers = [list(c) for c in centers]
    angles = list(angles)
    
    for it in range(max_iter):
        forces = [[0.0, 0.0] for _ in range(N)]
        max_pen = 0.0
        
        for i in range(N):
            for j in range(i+1, N):
                v1 = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                v2 = _hex_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
                
                def get_axes(verts):
                    axes = []
                    for k in range(6):
                        ex = verts[(k+1)%6, 0] - verts[k, 0]
                        ey = verts[(k+1)%6, 1] - verts[k, 1]
                        nx, ny = -ey, ex
                        norm = np.sqrt(nx*nx + ny*ny)
                        if norm > 1e-12:
                            axes.append((nx/norm, ny/norm))
                    return axes
                
                axes = get_axes(v1) + get_axes(v2)
                min_ov = float('inf')
                best_ax = None
                for ax, ay in axes:
                    p1 = v1[:,0]*ax + v1[:,1]*ay
                    p2 = v2[:,0]*ax + v2[:,1]*ay
                    ov = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
                    if ov > max_pen:
                        max_pen = ov
                    if ov > 0 and ov < min_ov:
                        min_ov = ov
                        best_ax = (ax, ay)
                
                if min_ov > 0 and best_ax:
                    ax, ay = best_ax
                    forces[i][0] -= ax * min_ov
                    forces[i][1] -= ay * min_ov
                    forces[j][0] += ax * min_ov
                    forces[j][1] += ay * min_ov
        
        if max_pen < tol:
            break
        
        step = 0.5
        for i in range(N):
            centers[i][0] += forces[i][0] * step
            centers[i][1] += forces[i][1] * step
    
    return centers, angles

def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check containment and non-overlap."""
    sqrt3 = np.sqrt(3)
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * sqrt3 / 2
    
    for i in range(N):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    for i in range(N):
        for j in range(i+1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    return True

def _get_valid_fallback():
    """Known valid configuration."""
    sqrt3 = np.sqrt(3)
    d1 = 2 * sqrt3
    centers = [(0.0, 0.0)]
    for i in range(6):
        a = np.pi/3 * i
        centers.append((d1*np.cos(a), d1*np.sin(a)))
    for i in range(4):
        a = np.pi/3 * i
        centers.append((2*d1*np.cos(a), 2*d1*np.sin(a)))
    angles = [0.0] * N
    side, center, angle = _optimize_outer(centers, angles)
    return centers, angles, center, side, angle

def _generate_initial_configs():
    """Generate diverse starting configurations."""
    configs = []
    sqrt3 = np.sqrt(3)
    d1 = 2 * sqrt3
    a = d1
    
    # Hex lattice: closest 11 points
    lattice = []
    for qi in range(-2, 3):
        for qj in range(-2, 3):
            x = a * (qi + qj * 0.5)
            y = a * qj * sqrt3 / 2
            lattice.append((x, y))
    lattice.sort(key=lambda p: p[0]**2 + p[1]**2)
    configs.append((lattice[:N], [0.0]*N))
    
    # Ring + center
    centers = [(0.0, 0.0)]
    for i in range(6):
        ang = np.pi/3 * i
        centers.append((d1*np.cos(ang), d1*np.sin(ang)))
    for i in range(4):
        ang = np.pi/3 * i + np.pi/6
        centers.append((2*d1*np.cos(ang), 2*d1*np.sin(ang)))
    configs.append((centers, [0.0]*N))
    
    # Rotated lattice
    centers_rot = []
    for x, y in lattice[:N]:
        ang = np.pi/12  # 15 degrees
        rx = x*np.cos(ang) - y*np.sin(ang)
        ry = x*np.sin(ang) + y*np.cos(ang)
        centers_rot.append((rx, ry))
    configs.append((centers_rot, [0.0]*N))
    
    # Random perturbation of lattice
    np.random.seed(99)
    centers_rand = [(x + np.random.normal(0, 0.2), y + np.random.normal(0, 0.2)) 
                    for x, y in lattice[:N]]
    configs.append((centers_rand, [0.0]*N))
    
    return configs

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    # Phase 1: Evaluate initial configs
    for centers, angles in _generate_initial_configs():
        if time.time() - start_time > 8:
            break
        # Force-directed relax first
        rc, ra = _force_directed_relax(centers, angles, max_iter=500)
        
        # Check validity
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if _hex_overlap(rc[i], ra[i], rc[j], ra[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            side, center, oa = _optimize_outer(rc, ra, max_angle_steps=20)
            if np.isfinite(side) and _check_validity(rc, ra, center, side, oa):
                if side < best_side:
                    best_side = side
                    best_result = (rc, ra, center, side, oa)
    
    if best_result is None:
        fb = _get_valid_fallback()
        best_result = fb
        best_side = fb[3]
    
    # Phase 2: Contact graph optimization (Nelder-Mead on smooth cost)
    if time.time() - start_time < DEADLINE - 40:
        bc, ba, _, _, _ = best_result
        cg_time = min(30.0, DEADLINE - time.time() - 20)
        cg_c, cg_a, cg_side, cg_center, cg_oa = _contact_graph_optimize(
            bc, ba, max_time=cg_time
        )
        if np.isfinite(cg_side) and _check_validity(cg_c, cg_a, cg_center, cg_side, cg_oa):
            if cg_side < best_side:
                best_side = cg_side
                best_result = (cg_c, cg_a, cg_center, cg_side, cg_oa)
    
    # Phase 3: Simulated annealing with angle moves
    if time.time() - start_time < DEADLINE - 15:
        bc, ba, _, _, _ = best_result
        centers = [list(c) for c in bc]
        angles = list(ba)
        
        cur_side, cur_center, cur_oa = _optimize_outer(centers, angles, max_angle_steps=15)
        if not np.isfinite(cur_side):
            cur_side, cur_center, cur_oa = best_result[2], best_result[3], best_result[4]
            centers = [list(c) for c in bc]
            angles = list(ba)
        
        best_sa_side = cur_side
        best_sa_c = [list(c) for c in centers]
        best_sa_a = list(angles)
        best_sa_center = cur_center
        best_sa_oa = cur_oa
        
        T = 0.2
        T_min = 0.003
        cool = 0.999
        np.random.seed(7)
        
        while time.time() - start_time < DEADLINE - 15 and T > T_min:
            nc = [list(c) for c in centers]
            na = list(angles)
            
            idx = np.random.randint(N)
            scale = 0.05 + 0.1 * (T / T_min)
            nc[idx][0] += np.random.normal(0, scale)
            nc[idx][1] += np.random.normal(0, scale)
            
            if np.random.rand() < 0.35:
                ri = np.random.randint(N)
                na[ri] += np.random.normal(0, 2.5)
            
            # Quick overlap check
            ok = True
            for i in range(N):
                for j in range(i+1, N):
                    if _hex_overlap(nc[i], na[i], nc[j], na[j]):
                        ok = False
                        break
                if not ok:
                    break
            
            if not ok:
                T *= cool
                continue
            
            ns, ncenter, noa = _optimize_outer(nc, na, max_angle_steps=10)
            if not np.isfinite(ns):
                T *= cool
                continue
            
            delta = ns - cur_side
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                centers = nc
                angles = na
                cur_side = ns
                cur_center = ncenter
                cur_oa = noa
                if cur_side < best_sa_side:
                    best_sa_side = cur_side
                    best_sa_c = [list(c) for c in centers]
                    best_sa_a = list(angles)
                    best_sa_center = cur_center
                    best_sa_oa = cur_oa
            
            T *= cool
        
        if best_sa_side < best_side:
            if _check_validity(best_sa_c, best_sa_a, best_sa_center, best_sa_side, best_sa_oa):
                best_side = best_sa_side
                best_result = (best_sa_c, best_sa_a, best_sa_center, best_sa_side, best_sa_oa)
    
    # Phase 4: Final polish - tight Nelder-Mead on the best result
    if time.time() - start_time < DEADLINE - 5:
        bc, ba, _, _, _ = best_result
        polish_time = min(8.0, DEADLINE - time.time() - 3)
        pc, pa, p_side, p_center, p_oa = _contact_graph_optimize(
            bc, ba, max_time=polish_time
        )
        if np.isfinite(p_side) and _check_validity(pc, pa, p_center, p_side, p_oa):
            if p_side < best_side:
                best_side = p_side
                best_result = (pc, pa, p_center, p_side, p_oa)
    
    # Final validation
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    if not _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle):
        fb = _get_valid_fallback()
        centers, inner_angles, outer_center, outer_side, outer_angle = fb
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
