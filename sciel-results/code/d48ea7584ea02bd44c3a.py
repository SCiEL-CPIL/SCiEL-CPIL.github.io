"""Rigid-body composite packing: treat 11 hexagons as one object with contact-graph constraints,
optimize global orientation and container via 1D support-function minimization."""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def hex_vertices_local(side=1.0, angle_rad=0.0):
    """Return 6 vertices of a unit hexagon centered at origin with given rotation."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([side * np.cos(thetas), side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    """
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
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """Smooth (log-sum-exp) surrogate for the max of |projection| over all 6 face normals."""
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
    abs_P = np.abs(P)
    m = abs_P.max()
    if m < 1e-15:
        return 0.0
    # Adaptive beta: higher beta when spread is small (near optimum), lower when far
    spread = m - abs_P.mean()
    if spread < 0.01:
        eff_beta = min(beta * 2.0, 400.0)
    elif spread < 0.1:
        eff_beta = beta
    else:
        eff_beta = beta * 0.5
    eff_beta = min(eff_beta, 400.0)
    if m * eff_beta > 500:
        eff_beta = 500.0 / m
    smooth_max = m + math.log(np.sum(np.exp(eff_beta * (abs_P - m)))) / eff_beta
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (lists of (x,y))."""
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
        min1 = min(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        max1 = max(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        min2 = min(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        max2 = max(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def overlap_penalty(inner, min_dist=SQRT3 - 0.002):
    """Quadratic penalty when center distances fall below min_dist.
    Uses a tighter threshold and stronger penalty to prevent overlaps."""
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, ai = inner[i]
        for j in range(i + 1, n):
            xj, yj, aj = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_dist:
                pen += (min_dist - d) ** 2
    # Add a softer long-range repulsion to encourage spreading
    for i in range(n):
        xi, yi, ai = inner[i]
        for j in range(i + 1, n):
            xj, yj, aj = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < SQRT3 * 1.5 and d > 0:
                pen += 0.01 * (SQRT3 * 1.5 - d) ** 2
    return pen

def build_lattice_config(axial_coords, s=1.0 + 1e-8):
    """Build inner hexagons from axial coordinates with spacing s."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + q / 2.0)) for q, r in axial_coords]
    return [[x, y, 0.0] for x, y in centers]

def rigid_composite_objective(offset, rotations, outer_angle_deg, outer_center, beta=60.0):
    """
    For a rigid composite: all inner hexagons share the same rotation relative to a base config.
    offset: (dx, dy) translation of the composite
    rotations: list of per-hexagon angle offsets (relative to base 0)
    outer_angle_deg: rotation of outer hexagon
    outer_center: center of outer hexagon
    Returns the required side length (smooth or exact based on beta).
    """
    inner = []
    for i in range(N):
        bx, by, ba = base_positions[i]
        x = bx + offset[0]
        y = by + offset[1]
        a = ba + rotations[i]
        inner.append([x, y, a])
    if beta > 0:
        return compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=beta)
    else:
        return compute_required_side_fast(inner, outer_center, outer_angle_deg)

# Base positions will be set per configuration
base_positions = None

def optimize_composite_rigid(axial_coords, deadline):
    """
    Treat the 11 hexagons as a rigid body (all same relative positions and angles),
    optimize only: global rotation, translation, outer center, outer angle.
    This reduces to a 6-DOF problem (3 for composite pose + 3 for outer).
    """
    global base_positions
    s0 = 1.0 + 1e-8
    centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + q / 2.0)) for q, r in axial_coords]
    base_positions = [[x, y, 0.0] for x, y in centers]
    
    cx0 = sum(c[0] for c in centers) / len(centers)
    cy0 = sum(c[1] for c in centers) / len(centers)
    
    def obj_rigid(p):
        """p = [comp_angle, dx, dy, ox, oy, outer_angle]"""
        comp_angle = p[0]
        dx, dy = p[1], p[2]
        ox, oy = p[3], p[4]
        outer_angle = p[5]
        
        ca = math.cos(math.radians(comp_angle))
        sa = math.sin(math.radians(comp_angle))
        inner = []
        for i in range(N):
            bx, by, ba = base_positions[i]
            rx = bx - cx0
            ry = by - cy0
            nx = cx0 + ca * rx - sa * ry + dx
            ny = cy0 + sa * rx + ca * ry + dy
            na = ba + comp_angle
            inner.append([nx, ny, na])
        
        return compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=60.0)
    
    def obj_rigid_exact(p):
        comp_angle = p[0]
        dx, dy = p[1], p[2]
        ox, oy = p[3], p[4]
        outer_angle = p[5]
        ca = math.cos(math.radians(comp_angle))
        sa = math.sin(math.radians(comp_angle))
        inner = []
        for i in range(N):
            bx, by, ba = base_positions[i]
            rx = bx - cx0
            ry = by - cy0
            nx = cx0 + ca * rx - sa * ry + dx
            ny = cy0 + sa * rx + ca * ry + dy
            na = ba + comp_angle
            inner.append([nx, ny, na])
        return compute_required_side_fast(inner, (ox, oy), outer_angle)
    
    best_val = float('inf')
    best_p = None
    
    # Multi-start over composite angle and outer angle - finer grid
    for comp_a in range(0, 60, 3):
        for outer_a in range(0, 60, 5):
            if time.time() > deadline - 25:
                break
            p0 = np.array([float(comp_a), 0.0, 0.0, cx0, cy0, float(outer_a)])
            try:
                r1 = minimize(obj_rigid, p0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 500})
                r2 = minimize(obj_rigid_exact, r1.x, method='Nelder-Mead',
                              options={'maxiter': 600, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 800})
                if r2.fun < best_val:
                    best_val = r2.fun
                    best_p = r2.x.copy()
            except Exception:
                pass
    
    if best_p is None:
        inner = [[x, y, 0.0] for x, y in centers]
        return inner, (cx0, cy0), 0.0, compute_required_side_fast(inner, (cx0, cy0), 0.0)
    
    comp_angle = float(best_p[0])
    dx, dy = float(best_p[1]), float(best_p[2])
    ox, oy = float(best_p[3]), float(best_p[4])
    outer_angle = float(best_p[5])
    
    ca = math.cos(math.radians(comp_angle))
    sa = math.sin(math.radians(comp_angle))
    inner = []
    for i in range(N):
        bx, by, ba = base_positions[i]
        rx = bx - cx0
        ry = by - cy0
        nx = cx0 + ca * rx - sa * ry + dx
        ny = cy0 + sa * rx + ca * ry + dy
        na = ba + comp_angle
        inner.append([nx, ny, na])
    
    return inner, (ox, oy), outer_angle, best_val

def semi_rigid_optimize(axial_coords, deadline):
    """
    Semi-rigid: inner hexagons can have individual angle offsets but positions are fixed
    relative to each other (rigid translation). Optimize per-hexagon angles + outer params.
    This is a 11 + 3 = 14 DOF problem.
    """
    global base_positions
    s0 = 1.0 + 1e-8
    centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + q / 2.0)) for q, r in axial_coords]
    base_positions = [[x, y, 0.0] for x, y in centers]
    
    cx0 = sum(c[0] for c in centers) / len(centers)
    cy0 = sum(c[1] for c in centers) / len(centers)
    
    def obj_semi(p):
        angles = p[:N]
        ox, oy, outer_angle = p[N], p[N+1], p[N+2]
        inner = []
        for i in range(N):
            bx, by, _ = base_positions[i]
            inner.append([bx, by, angles[i]])
        return compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=50.0)
    
    def obj_semi_exact(p):
        angles = p[:N]
        ox, oy, outer_angle = p[N], p[N+1], p[N+2]
        inner = []
        for i in range(N):
            bx, by, _ = base_positions[i]
            inner.append([bx, by, angles[i]])
        return compute_required_side_fast(inner, (ox, oy), outer_angle)
    
    best_val = float('inf')
    best_p = None
    
    for outer_a in range(0, 60, 10):
        if time.time() > deadline - 20:
            break
        p0 = np.zeros(N + 3)
        p0[N] = cx0
        p0[N+1] = cy0
        p0[N+2] = float(outer_a)
        np.random.seed(hash(outer_a) % 10000)
        p0[:N] = np.random.uniform(-20, 20, N)
        
        try:
            r1 = minimize(obj_semi, p0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 800})
            r2 = minimize(obj_semi_exact, r1.x, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 1200})
            if r2.fun < best_val:
                best_val = r2.fun
                best_p = r2.x.copy()
        except Exception:
            pass
    
    if best_p is None:
        inner = [[x, y, 0.0] for x, y in centers]
        return inner, (cx0, cy0), 0.0, compute_required_side_fast(inner, (cx0, cy0), 0.0)
    
    angles = best_p[:N]
    ox, oy, outer_angle = float(best_p[N]), float(best_p[N+1]), float(best_p[N+2])
    inner = []
    for i in range(N):
        bx, by, _ = base_positions[i]
        inner.append([bx, by, float(angles[i])])
    
    return inner, (ox, oy), outer_angle, best_val

def full_joint_optimize(init_inner, init_oc, init_oa, deadline):
    """Full joint optimization with simulated annealing + DE exploration."""
    def obj_joint_smooth(params, beta=60.0):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 2000.0 * pen

    def obj_joint_exact(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 10000.0 * pen

    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(init_inner):
        p0[i * 3] = x
        p0[i * 3 + 1] = y
        p0[i * 3 + 2] = a
    p0[N * 3] = init_oc[0]
    p0[N * 3 + 1] = init_oc[1]
    p0[N * 3 + 2] = init_oa

    current = p0.copy()
    current_val = obj_joint_smooth(current)
    best_params = current.copy()
    best_val_j = current_val

    # Phase 1: Simulated Annealing with adaptive step sizes
    T = 1.0
    T_min = 0.001
    cool = 0.9996
    max_iter = 8000
    np.random.seed(123)
    for it in range(max_iter):
        if time.time() > deadline - 40:
            break
        nb = current.copy()
        r = np.random.random()
        # Scale step with temperature for better exploration early, exploitation late
        scale = max(0.1, T)
        if r < 0.60:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx * 3 + dim] += np.random.normal(0, 0.3 * scale)
        elif r < 0.78:
            idx = np.random.randint(0, N)
            nb[idx * 3 + 2] += np.random.normal(0, 12.0 * scale)
        elif r < 0.88:
            dim = np.random.randint(0, 2)
            nb[N * 3 + dim] += np.random.normal(0, 0.2 * scale)
        else:
            nb[N * 3 + 2] += np.random.normal(0, 5.0 * scale)

        nb_val = obj_joint_smooth(nb)
        if nb_val < current_val or np.random.random() < math.exp(-(nb_val - current_val) / max(T, 1e-10)):
            current = nb
            current_val = nb_val
            if current_val < best_val_j:
                best_params = current.copy()
                best_val_j = current_val
        T *= cool
        if T < T_min:
            break

    # Phase 2: Differential Evolution with cached evaluations
    if time.time() < deadline - 30:
        de_best = best_params.copy()
        de_best_val = best_val_j
        
        np.random.seed(456)
        pop_size = 15
        population = [best_params.copy()]
        for _ in range(pop_size - 1):
            p = best_params.copy()
            for i in range(N):
                p[i * 3] += np.random.normal(0, 0.6)
                p[i * 3 + 1] += np.random.normal(0, 0.6)
                p[i * 3 + 2] += np.random.normal(0, 18.0)
            p[N * 3] += np.random.normal(0, 0.4)
            p[N * 3 + 1] += np.random.normal(0, 0.4)
            p[N * 3 + 2] += np.random.normal(0, 6.0)
            population.append(p)
        
        # Cache population evaluations
        pop_vals = [obj_joint_smooth(p) for p in population]
        
        F = 0.7
        CR = 0.8
        max_gen = 40
        for gen in range(max_gen):
            if time.time() > deadline - 30:
                break
            new_pop = population[:]
            new_vals = pop_vals[:]
            for i in range(pop_size):
                a, b, c = i, i, i
                while a == i:
                    a = np.random.randint(0, pop_size)
                while b == i or b == a:
                    b = np.random.randint(0, pop_size)
                while c == i or c == a or c == b:
                    c = np.random.randint(0, pop_size)
                
                trial = population[a].copy()
                for j in range(len(trial)):
                    if np.random.random() < CR or j == np.random.randint(0, len(trial)):
                        trial[j] = population[a][j] + F * (population[b][j] - population[c][j])
                
                trial_val = obj_joint_smooth(trial)
                if trial_val <= new_vals[i]:
                    new_pop[i] = trial
                    new_vals[i] = trial_val
            
            population = new_pop
            pop_vals = new_vals
            for v, p in zip(pop_vals, population):
                if v < de_best_val:
                    de_best_val = v
                    de_best = p.copy()
        
        if de_best_val < best_val_j:
            best_params = de_best
            best_val_j = de_best_val

    best_inner_j = []
    for i in range(N):
        best_inner_j.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    best_oc_j = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
    best_oa_j = float(best_params[N * 3 + 2])

    return best_inner_j, best_oc_j, best_oa_j, best_params

def final_polish(params, deadline):
    """High-precision multi-stage polish: smooth NM -> exact NM -> fine coordinate descent."""
    def obj_smooth(p, beta=80.0):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 5000.0 * pen

    def obj_exact(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 20000.0 * pen

    current = params.copy()
    
    # Stage 1: Smooth Nelder-Mead
    try:
        r1 = minimize(obj_smooth, current, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 3000})
        current = r1.x
    except Exception:
        pass
    
    # Stage 2: Exact Nelder-Mead
    if time.time() < deadline - 8:
        try:
            r2 = minimize(obj_exact, current, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 3000})
            current = r2.x
        except Exception:
            pass
    
    # Stage 3: Fine coordinate descent with adaptive step on exact objective
    if time.time() < deadline - 3:
        cur_val = obj_exact(current)
        step = 0.01
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            improved = False
            for j in range(len(current)):
                for sign in [1, -1]:
                    trial = current.copy()
                    trial[j] += sign * step
                    trial_val = obj_exact(trial)
                    if trial_val < cur_val - 1e-13:
                        current = trial
                        cur_val = trial_val
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if step < 1e-10:
                    break
    
    # Stage 4: Very fine local search on outer parameters only
    if time.time() < deadline - 2:
        cur_val = obj_exact(current)
        fine_step = 0.005
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            improved = False
            # Only optimize outer center and angle for fine polish
            for j in [N * 3, N * 3 + 1, N * 3 + 2]:
                for sign in [1, -1]:
                    trial = current.copy()
                    trial[j] += sign * fine_step
                    trial_val = obj_exact(trial)
                    if trial_val < cur_val - 1e-14:
                        current = trial
                        cur_val = trial_val
                        improved = True
                        break
            if not improved:
                fine_step *= 0.5
                if fine_step < 1e-11:
                    break
    
    return current

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin

    # --- Define multiple candidate configurations ---
    axial1 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, -1), (1, 1), (-2, 1), (-1, -1)]
    
    axial2 = [(-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1), (2, 1)]
    
    axial3 = [(-1, -1), (0, -1), (1, -1),
              (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
              (-1, 1), (0, 1), (1, 1)]
    
    axial4 = [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1)]
    
    axial5 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, 0), (0, 2), (-2, 1), (-2, -1)]
    
    axial6 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
              (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)]
    
    axial7 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
              (2, -1), (2, 0), (1, 1), (0, 2), (-1, 2)]
    
    # Additional diverse topologies
    axial8 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
              (1, -1), (2, -1), (2, 0), (0, 2), (-2, 1)]
    
    axial9 = [(-1, -1), (0, -1), (1, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1),
              (2, 0), (-2, 0)]
    
    axial10 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
               (1, -1), (2, 0), (0, 2), (-2, 0), (2, -1)]
    
    configs = [axial1, axial2, axial3, axial4, axial5, axial6, axial7, axial8, axial9, axial10]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None
    
    # Phase 1a: Rigid composite optimization for each config
    for ci, axial in enumerate(configs):
        if time.time() > deadline - 75:
            break
        inner, oc, oa, side_val = optimize_composite_rigid(axial, deadline)
        
        if check_disjoint(inner, eps=1e-5) and side_val < best_side:
            best_side = side_val
            best_inner = inner
            best_outer_center = oc
            best_outer_angle = oa
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(inner):
                p0[i * 3] = x
                p0[i * 3 + 1] = y
                p0[i * 3 + 2] = a
            p0[N * 3] = oc[0]
            p0[N * 3 + 1] = oc[1]
            p0[N * 3 + 2] = oa
            best_params = p0.copy()
    
    # Phase 1b: Semi-rigid optimization (individual angles) for top configs
    for ci, axial in enumerate(configs[:5]):
        if time.time() > deadline - 50:
            break
        inner, oc, oa, side_val = semi_rigid_optimize(axial, deadline)
        
        if check_disjoint(inner, eps=1e-5) and side_val < best_side:
            best_side = side_val
            best_inner = inner
            best_outer_center = oc
            best_outer_angle = oa
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(inner):
                p0[i * 3] = x
                p0[i * 3 + 1] = y
                p0[i * 3 + 2] = a
            p0[N * 3] = oc[0]
            p0[N * 3 + 1] = oc[1]
            p0[N * 3 + 2] = oa
            best_params = p0.copy()
    
    if best_inner is None:
        axial = configs[0]
        s0 = 1.0 + 1e-8
        best_inner = build_lattice_config(axial, s0)
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
        p0 = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(best_inner):
            p0[i * 3] = x
            p0[i * 3 + 1] = y
            p0[i * 3 + 2] = a
        p0[N * 3] = cx0
        p0[N * 3 + 1] = cy0
        p0[N * 3 + 2] = 0.0
        best_params = p0.copy()
    
    # Phase 2: Full joint optimization from best config
    if time.time() < deadline - 45:
        joint_inner, joint_oc, joint_oa, joint_params = full_joint_optimize(
            best_inner, best_outer_center, best_outer_angle, deadline)
        
        if check_disjoint(joint_inner, eps=1e-5):
            joint_side = compute_required_side_fast(joint_inner, joint_oc, joint_oa)
            if joint_side < best_side:
                best_inner = joint_inner
                best_outer_center = joint_oc
                best_outer_angle = joint_oa
                best_side = joint_side
                best_params = joint_params
    
    # Phase 2b: Try joint optimization from perturbed best (multiple perturbations)
    if best_params is not None and time.time() < deadline - 35:
        for trial_seed in [789, 1024]:
            if time.time() > deadline - 35:
                break
            np.random.seed(trial_seed)
            perturbed = best_params.copy()
            for i in range(N):
                perturbed[i * 3] += np.random.normal(0, 0.3)
                perturbed[i * 3 + 1] += np.random.normal(0, 0.3)
                perturbed[i * 3 + 2] += np.random.normal(0, 8.0)
            perturbed[N * 3] += np.random.normal(0, 0.2)
            perturbed[N * 3 + 1] += np.random.normal(0, 0.2)
            perturbed[N * 3 + 2] += np.random.normal(0, 3.0)
            
            pert_inner = []
            for i in range(N):
                pert_inner.append([float(perturbed[i * 3]), float(perturbed[i * 3 + 1]), float(perturbed[i * 3 + 2])])
            pert_oc = (float(perturbed[N * 3]), float(perturbed[N * 3 + 1]))
            pert_oa = float(perturbed[N * 3 + 2])
            
            if check_disjoint(pert_inner, eps=1e-4):
                pert_side = compute_required_side_fast(pert_inner, pert_oc, pert_oa)
                if pert_side < best_side * 1.05:
                    j2_inner, j2_oc, j2_oa, j2_params = full_joint_optimize(
                        pert_inner, pert_oc, pert_oa, deadline)
                    if check_disjoint(j2_inner, eps=1e-5):
                        j2_side = compute_required_side_fast(j2_inner, j2_oc, j2_oa)
                        if j2_side < best_side:
                            best_inner = j2_inner
                            best_outer_center = j2_oc
                            best_outer_angle = j2_oa
                            best_side = j2_side
                            best_params = j2_params
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 15 and best_params is not None:
        polished = final_polish(best_params, deadline)
        polish_inner = []
        for i in range(N):
            polish_inner.append([float(polished[i * 3]), float(polished[i * 3 + 1]), float(polished[i * 3 + 2])])
        polish_oc = (float(polished[N * 3]), float(polished[N * 3 + 1]))
        polish_oa = float(polished[N * 3 + 2])
        
        if check_disjoint(polish_inner, eps=1e-5):
            polish_side = compute_required_side_fast(polish_inner, polish_oc, polish_oa)
            if polish_side < best_side:
                best_inner = polish_inner
                best_outer_center = polish_oc
                best_outer_angle = polish_oa
                best_side = polish_side
    
    # Phase 4: Bisection-guided refinement - try to squeeze tighter by adjusting outer angle
    if time.time() < deadline - 8 and best_params is not None:
        cur_inner = best_inner
        cur_oc = best_outer_center
        cur_oa = best_outer_angle
        cur_side = best_side
        
        # Try fine outer angle sweeps around current best
        for da in [1, -1, 2, -2, 0.5, -0.5]:
            if time.time() > deadline - 5:
                break
            new_oa = cur_oa + da
            new_side = compute_required_side_fast(cur_inner, cur_oc, new_oa)
            if new_side < cur_side - 1e-9:
                cur_oa = new_oa
                cur_side = new_side
        
        # Try fine outer center shifts
        for dx in [0.01, -0.01, 0.02, -0.02]:
            for dy in [0.01, -0.01, 0.02, -0.02]:
                if time.time() > deadline - 5:
                    break
                new_oc = (cur_oc[0] + dx, cur_oc[1] + dy)
                new_side = compute_required_side_fast(cur_inner, new_oc, cur_oa)
                if new_side < cur_side - 1e-9:
                    cur_oc = new_oc
                    cur_side = new_side
        
        if cur_side < best_side:
            best_outer_center = cur_oc
            best_outer_angle = cur_oa
            best_side = cur_side
    
    # Final verification
    if not check_disjoint(best_inner, eps=1e-5):
        s0 = 1.0 + 1e-8
        best_inner = build_lattice_config(configs[0], s0)
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
