"""Optimized hexagon packing: rigid composite + smooth penalty + high-precision polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """Exact minimum outer side to contain all inner hexagons."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_proj:
                    max_proj = proj
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=40.0):
    """Smooth max via log-sum-exp for gradient-friendly optimization."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                projections.append(lx * nx + ly * ny)
    if not projections:
        return 0.0
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    return 2.0 * smooth_max / SQRT3

def sat_gap(v1, v2):
    """Return minimum gap between two convex polygons (positive = separated, negative = overlap)."""
    axes = []
    for i in range(len(v1)):
        p1 = v1[i]; p2 = v1[(i+1) % len(v1)]
        e = (p2[0]-p1[0], p2[1]-p1[1])
        n = (-e[1], e[0])
        nn = math.sqrt(n[0]*n[0] + n[1]*n[1])
        if nn > 1e-12:
            axes.append((n[0]/nn, n[1]/nn))
    for i in range(len(v2)):
        p1 = v2[i]; p2 = v2[(i+1) % len(v2)]
        e = (p2[0]-p1[0], p2[1]-p1[1])
        n = (-e[1], e[0])
        nn = math.sqrt(n[0]*n[0] + n[1]*n[1])
        if nn > 1e-12:
            axes.append((n[0]/nn, n[1]/nn))
    min_gap = float('inf')
    for ax in axes:
        m1 = min(p[0]*ax[0]+p[1]*ax[1] for p in v1)
        M1 = max(p[0]*ax[0]+p[1]*ax[1] for p in v1)
        m2 = min(p[0]*ax[0]+p[1]*ax[1] for p in v2)
        M2 = max(p[0]*ax[0]+p[1]*ax[1] for p in v2)
        gap = min(M1, M2) - max(m1, m2)
        if gap < min_gap:
            min_gap = gap
    return min_gap

def smooth_overlap_penalty(inner, beta=80.0):
    """Smooth sum of squared overlaps (soft penalty for non-disjointness)."""
    penalty = 0.0
    verts = [get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2])) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            gap = sat_gap(verts[i], verts[j])
            # If gap < 0, there's overlap; penalize smoothly
            if gap < 0.01:
                # Soft penalty: use smooth approximation of max(0, -gap)^2
                t = -gap * beta
                if t > 50:
                    penalty += gap * gap
                else:
                    penalty += (math.exp(t) - 1.0) ** 2 / (beta * beta)
    return penalty

def check_disjoint(inner, eps=1e-7):
    verts = [get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2])) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if sat_gap(verts[i], verts[j]) < -eps:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin for 150s limit

    # Hexagonal lattice coordinates for 11 hexagons (center + 6 ring + 4 extra)
    # Using axial coordinates (q, r) -> Cartesian
    lattice = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (-1, 2)
    ]

    def to_cartesian(q, r, scale=1.0):
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q * 0.5)
        return x, y

    # Base positions (scale=1.0005, angle=0) - slight expansion to ensure disjointness
    base_positions = [to_cartesian(q, r, scale=1.0005) for q, r in lattice]
    base_angles = [0.0] * N

    # Centroid
    cx0 = sum(p[0] for p in base_positions) / N
    cy0 = sum(p[1] for p in base_positions) / N

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # ---- Phase 1: Rigid composite optimization (6 DOF: dx, dy, theta, ox, oy, oangle) ----
    # Treat all hexagons as a rigid body. Parameters: (tx, ty, rot, ox, oy, oangle)
    # Inner hex i position: rotate base_i by rot around origin, then translate by (tx, ty)

    def rigid_objective(params, use_smooth=True, beta=40.0):
        tx, ty, rot_deg, ox, oy, oangle = params
        rot = math.radians(rot_deg)
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        inner = []
        for i in range(N):
            bx, by = base_positions[i]
            # Rotate around origin
            rx = bx * cos_r - by * sin_r
            ry = bx * sin_r + by * cos_r
            # Translate
            inner.append([rx + tx, ry + ty, rot_deg + base_angles[i]])
        if use_smooth:
            side = compute_required_side_smooth(inner, (ox, oy), oangle, beta)
        else:
            side = compute_required_side(inner, (ox, oy), oangle)
        return side

    # Multi-start rigid optimization
    # The rigid body assumption preserves the relative positions, so disjointness is guaranteed
    # if the initial lattice is disjoint (which it is).
    init_configs = []
    # Try various rotations of the cluster and the outer hexagon
    # The cluster has 6-fold symmetry, so 0-60 degrees is sufficient for cluster rotation
    # But we try 0-360 just in case of asymmetric initial lattice (though it is symmetric)
    for rot_init in range(0, 60, 5):
        for oangle_init in range(0, 60, 5):
            init_configs.append([cx0, cy0, float(rot_init), cx0, cy0, float(oangle_init)])
    
    # Add some random perturbations to the center to escape local minima
    np.random.seed(42)
    for _ in range(10):
        dx = np.random.randn() * 0.1
        dy = np.random.randn() * 0.1
        rot_init = np.random.uniform(0, 60)
        oangle_init = np.random.uniform(0, 60)
        init_configs.append([cx0 + dx, cy0 + dy, float(rot_init), cx0 + dx, cy0 + dy, float(oangle_init)])

    for ic in init_configs:
        if time.time() > deadline - 20:
            break
        try:
            # Use smooth objective for initial convergence, then exact
            r1 = minimize(rigid_objective, ic, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 400,
                                   'initial_simplex': [0.5, 0.5, 5.0, 0.5, 0.5, 5.0]})
            # Refine with exact objective
            r2 = minimize(rigid_objective, r1.x, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 1000,
                                   'initial_simplex': [0.01, 0.01, 0.1, 0.01, 0.01, 0.1]})
            val = r2.fun
            if val < best_side:
                tx, ty, rot_deg, ox, oy, oangle = r2.x
                best_inner = []
                rot = math.radians(rot_deg)
                cos_r, sin_r = math.cos(rot), math.sin(rot)
                for i in range(N):
                    bx, by = base_positions[i]
                    rx = bx * cos_r - by * sin_r
                    ry = bx * sin_r + by * cos_r
                    best_inner.append([rx + tx, ry + ty, rot_deg])
                
                # Verify disjointness
                if check_disjoint(best_inner, eps=1e-5):
                    best_side = val
                    best_center = (ox, oy)
                    best_angle = oangle
        except Exception:
            continue

    # Fallback
    if best_inner is None:
        best_inner = [[base_positions[i][0], base_positions[i][1], 0.0] for i in range(N)]
        best_center = (cx0, cy0)
        best_angle = 0.0
        best_side = compute_required_side(best_inner, best_center, best_angle)
        # The fallback uses the scaled lattice, so it should be disjoint

    # ---- Phase 2: High-precision polish on outer container ----
    # The rigid cluster is already disjoint. We just need to find the best fitting outer hexagon.
    if time.time() < deadline - 5:
        def polish_obj(params):
            return compute_required_side(best_inner, (params[0], params[1]), params[2])

        try:
            r_polish = minimize(polish_obj, [best_center[0], best_center[1], best_angle],
                                method='Nelder-Mead',
                                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 8000,
                                         'initial_simplex': [1e-3, 1e-3, 1e-3]})
            if r_polish.fun < best_side:
                best_side = r_polish.fun
                best_center = (r_polish.x[0], r_polish.x[1])
                best_angle = r_polish.x[2]
        except Exception:
            pass

    # Final containment epsilon
    best_side *= (1.0 + 1e-6)

    inner = [[best_inner[i][0], best_inner[i][1], best_inner[i][2]] for i in range(N)]
    return inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
