"""Optimized 11-hexagon packing: multi-start, rigid-composite init, smooth surrogate SA, high-precision polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = math.sqrt(3) / 2.0

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def container_normals(angle_deg):
    """Return 6 unit normals of the outer hexagon walls."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    base = np.array([[math.cos(math.radians(30 + i * 60)),
                      math.sin(math.radians(30 + i * 60))] for i in range(6)])
    R = np.array([[ca, -sa], [sa, ca]])
    return base @ R.T

def support_function_exact(inner, outer_center, outer_angle_deg):
    """
    Compute the maximum projection of all inner hexagon vertices onto the 6
    outer-hexagon wall normals, relative to outer_center.
    Returns max_projection (scalar).
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    max_proj = -np.inf
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        rel = verts - np.array([ox, oy])
        projs = rel @ normals.T  # (6,)
        m = projs.max()
        if m > max_proj:
            max_proj = m
    return float(max_proj)

def compute_required_side(inner, outer_center, outer_angle_deg):
    """Minimum outer side length to contain all inner hexagons."""
    mp = support_function_exact(inner, outer_center, outer_angle_deg)
    return 2.0 * mp / SQRT3

def support_function_smooth(inner, outer_center, outer_angle_deg, beta=40.0):
    """
    Smooth log-sum-exp approximation of the max projection.
    Returns (smooth_max, exact_max).
    """
    if not inner:
        return 0.0, 0.0
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    all_projs = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        rel = verts - np.array([ox, oy])
        projs = rel @ normals.T
        all_projs.append(projs)
    P = np.concatenate([p.ravel() for p in all_projs])
    if P.size == 0:
        return 0.0, 0.0
    m = float(P.max())
    smooth_val = m + math.log(float(np.sum(np.exp(beta * (P - m))))) / beta
    return float(smooth_val), m

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=40.0):
    """Smooth surrogate for required side length."""
    sm, _ = support_function_smooth(inner, outer_center, outer_angle_deg, beta)
    return 2.0 * sm / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT. Returns True if they overlap."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    for axis in axes:
        proj1 = v1 @ axis
        proj2 = v2 @ axis
        if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
            return False
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    polys = [get_hex_vertices(x, y, 1.0, math.radians(a)) for (x, y, a) in inner]
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(polys[i], polys[j], eps):
                return False
    return True

def overlap_penalty_smooth(inner, min_dist=SQRT3 - 0.003):
    """
    Smooth penalty based on center-to-center distance.
    For two unit hexagons, the minimum center distance for non-overlap is SQRT3
    when aligned, but can be less when rotated. We use a conservative min_dist.
    """
    n = len(inner)
    if n < 2:
        return 0.0
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists = np.sqrt(dx * dx + dy * dy + 1e-12)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    diff = np.maximum(0.0, min_dist - d)
    return float(np.sum(diff * diff))

def build_axial_layout(coords):
    """Convert axial (q, r) coordinates to Cartesian centers with unit spacing."""
    centers = []
    for q, r in coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))
    return centers

def generate_candidates():
    """Generate diverse initial hexagonal lattice configurations for N=11."""
    candidates = []

    # Flower: center + 6 ring + 4 extra
    c1 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1), (2,0),(-2,0),(0,2),(0,-2)]
    candidates.append((c1, "flower"))

    # 3-5-3 rows
    c2 = [(0,0),(1,0),(2,0), (0,1),(1,1),(2,1),(3,1),(4,1), (0,2),(1,2),(2,2)]
    candidates.append((c2, "353"))

    # 4-3-4 rows
    c3 = [(0,0),(1,0),(2,0),(3,0), (0,1),(1,1),(2,1), (0,2),(1,2),(2,2),(3,2)]
    candidates.append((c3, "434"))

    # 5-3-3
    c4 = [(0,0),(1,0),(2,0),(3,0),(4,0), (0,1),(1,1),(2,1), (0,2),(1,2),(2,2)]
    candidates.append((c4, "533"))

    # 3-4-4
    c5 = [(0,0),(1,0),(2,0), (0,1),(1,1),(2,1),(3,1), (0,2),(1,2),(2,2),(3,2)]
    candidates.append((c5, "344"))

    # Compact diamond
    c6 = [(0,0), (1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1), (2,0),(-2,0),(0,2)]
    candidates.append((c6, "diamond"))

    # 2-3-3-3
    c7 = [(0,0),(1,0), (0,1),(1,1),(2,1), (0,2),(1,2),(2,2), (0,3),(1,3),(2,3)]
    candidates.append((c7, "2333"))

    # Asymmetric
    c8 = [(0,0),(1,0),(2,0), (0,1),(1,1),(2,1),(3,1), (1,2),(2,2),(3,2),(4,2)]
    candidates.append((c8, "asym"))

    return candidates

def optimize_outer_for_fixed_inner(inner, outer_center_init, outer_angle_init, deadline):
    """
    Optimize outer container (center, angle) for fixed inner layout.
    Returns (best_center, best_angle, best_side).
    """
    def obj_smooth(p):
        cx, cy, ang = p
        return compute_required_side_smooth(inner, (cx, cy), ang, beta=40.0)

    def obj_exact(p):
        cx, cy, ang = p
        return compute_required_side(inner, (cx, cy), ang)

    best_val = float('inf')
    best_params = None

    # Multi-start over angles
    for ang0 in range(0, 360, 30):
        if time.time() > deadline:
            break
        p0 = [outer_center_init[0], outer_center_init[1], float(ang0)]
        try:
            res = minimize(obj_smooth, p0, method='Nelder-Mead',
                          options={'maxiter': 600, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 1200})
            res2 = minimize(obj_exact, res.x, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 2000})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass

    if best_params is not None:
        return (best_params[0], best_params[1]), best_params[2], best_val
    else:
        return outer_center_init, outer_angle_init, compute_required_side(inner, outer_center_init, outer_angle_init)

def joint_sa_optimize(inner_init, outer_center, outer_angle, adjacency, deadline, max_iter=12000):
    """
    Joint simulated annealing optimization of inner positions/angles and outer container.
    """
    n = len(inner_init)
    nparams = n * 3 + 3

    def decode(params):
        inner_list = []
        for i in range(n):
            inner_list.append([params[i*3], params[i*3+1], params[i*3+2]])
        return inner_list, (params[n*3], params[n*3+1]), params[n*3+2]

    def objective(params, beta=30.0):
        inner_list, oc, oa = decode(params)
        sm, _ = support_function_smooth(inner_list, oc, oa, beta)
        side = 2.0 * sm / SQRT3
        pen = overlap_penalty_smooth(inner_list, min_dist=SQRT3 - 0.002)
        return side + 200.0 * pen

    # Build adjacency from initial layout
    # (precomputed and passed in)

    current = np.array([v for h in inner_init for v in h] + [outer_center[0], outer_center[1], outer_angle])
    current_val = objective(current, beta=30.0)
    best = current.copy()
    best_val = current_val

    T = 1.0
    T_min = 0.001
    cooling = 0.9999
    patience = 2000
    no_improve = 0

    for it in range(max_iter):
        if time.time() > deadline:
            break

        beta_sa = 30.0 + 25.0 * min(1.0, it / max_iter)
        beta_sa = min(beta_sa, 55.0)

        neighbor = current.copy()
        r = np.random.random()
        if r < 0.65:
            idx = np.random.randint(0, n * 3)
            if idx % 3 == 2:  # angle
                delta = np.random.normal(0, 0.15 * (1 + 0.5 * T))
            else:  # position
                delta = np.random.normal(0, 0.10 * (1 + 0.5 * T))
            neighbor[idx] += delta
        else:
            idx = n * 3 + np.random.randint(0, 3)
            if idx == n * 3 + 2:  # outer angle
                delta = np.random.normal(0, 2.0 * (1 + 0.3 * T))
            else:
                delta = np.random.normal(0, 0.06 * (1 + 0.5 * T))
            neighbor[idx] += delta

        neighbor_val = objective(neighbor, beta=beta_sa)

        if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / max(T, T_min)):
            current = neighbor
            current_val = neighbor_val
            if current_val < best_val:
                best = current.copy()
                best_val = current_val
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > patience:
                no_improve = 0
                T = max(T * 0.5, T_min * 2)

        T *= cooling
        if T < T_min:
            break

    inner_best, oc_best, oa_best = decode(best)
    return inner_best, (oc_best[0], oc_best[1]), oa_best, best_val

def polish_inner_positions(inner, outer_center, outer_angle, deadline, max_iter=8000):
    """
    Polish inner positions with outer fixed, using smooth objective + overlap penalty.
    """
    n = len(inner)

    def objective(params, beta=40.0):
        inner_list = [[params[i*3], params[i*3+1], params[i*3+2]] for i in range(n)]
        sm, _ = support_function_smooth(inner_list, outer_center, outer_angle, beta)
        side = 2.0 * sm / SQRT3
        pen = overlap_penalty_smooth(inner_list, min_dist=SQRT3 - 0.002)
        return side + 200.0 * pen

    current = np.array([v for h in inner for v in h])
    current_val = objective(current, beta=40.0)
    best = current.copy()
    best_val = current_val

    T = 0.5
    T_min = 0.001
    cooling = 0.9998
    patience = 1500
    no_improve = 0

    for it in range(max_iter):
        if time.time() > deadline:
            break
        beta_sa = 40.0 + 15.0 * min(1.0, it / max_iter)
        beta_sa = min(beta_sa, 55.0)

        neighbor = current.copy()
        idx = np.random.randint(0, n * 3)
        if idx % 3 == 2:
            delta = np.random.normal(0, 0.08 * (1 + 0.5 * T))
        else:
            delta = np.random.normal(0, 0.06 * (1 + 0.5 * T))
        neighbor[idx] += delta

        neighbor_val = objective(neighbor, beta=beta_sa)
        if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / max(T, T_min)):
            current = neighbor
            current_val = neighbor_val
            if current_val < best_val:
                best = current.copy()
                best_val = current_val
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > patience:
                no_improve = 0
                T = max(T * 0.5, T_min * 2)
        T *= cooling
        if T < T_min:
            break

    inner_best = [[best[i*3], best[i*3+1], best[i*3+2]] for i in range(n)]
    return inner_best

def polish_outer_container(inner, outer_center, outer_angle, deadline):
    """
    High-precision polish of outer container with fixed inner.
    """
    def obj_exact(p):
        cx, cy, ang = p
        return compute_required_side(inner, (cx, cy), ang)

    best_val = compute_required_side(inner, outer_center, outer_angle)
    best_params = [outer_center[0], outer_center[1], outer_angle]

    # Diverse starts
    starts = []
    for da in [-10, -5, -2, 0, 2, 5, 10]:
        for dc in [(-0.1, 0), (0.1, 0), (0, -0.1), (0, 0.1), (0, 0)]:
            starts.append([outer_center[0] + dc[0], outer_center[1] + dc[1], outer_angle + da])

    for ps in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(obj_exact, ps, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 10000})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    return (best_params[0], best_params[1]), best_params[2], best_val

def try_fix_overlaps(inner, outer_center, outer_angle, max_attempts=80):
    """Nudge overlapping hexagons apart."""
    fixed = [list(h) for h in inner]
    for _ in range(max_attempts):
        overlap_found = False
        for i in range(len(fixed)):
            vi = get_hex_vertices(fixed[i][0], fixed[i][1], 1.0, math.radians(fixed[i][2]))
            for j in range(i + 1, len(fixed)):
                vj = get_hex_vertices(fixed[j][0], fixed[j][1], 1.0, math.radians(fixed[j][2]))
                if sat_overlap_check(vi, vj, eps=1e-6):
                    overlap_found = True
                    dx = fixed[j][0] - fixed[i][0]
                    dy = fixed[j][1] - fixed[i][1]
                    d = math.sqrt(dx*dx + dy*dy) + 1e-9
                    nx, ny = dx/d, dy/d
                    fixed[i][0] -= 0.005 * nx
                    fixed[i][1] -= 0.005 * ny
                    fixed[j][0] += 0.005 * nx
                    fixed[j][1] += 0.005 * ny
        if not overlap_found:
            break
    return fixed

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    candidates = generate_candidates()
    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0

    for idx, (axial_coords, name) in enumerate(candidates):
        if time.time() > deadline - 15:
            break

        # Build initial layout
        centers = build_axial_layout(axial_coords)
        # Ensure exactly N
        while len(centers) < N:
            centers.append((0, 0))
        centers = centers[:N]

        # Slight expansion to avoid initial overlaps
        eps_scale = 1.0 + 0.002
        centers_scaled = [(x * eps_scale, y * eps_scale) for x, y in centers]
        initial_inner = [[x, y, 0.0] for x, y in centers_scaled]

        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)

        # Phase 1: Optimize outer for fixed inner
        oc, oa, side1 = optimize_outer_for_fixed_inner(
            initial_inner, (cx_avg, cy_avg), 0.0, deadline - 25)

        # Phase 2: Joint SA
        # Build adjacency
        adjacency = set()
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                if math.sqrt(dx*dx + dy*dy) < SQRT3 * 1.15:
                    adjacency.add((i, j))

        inner2, oc2, oa2, val2 = joint_sa_optimize(
            initial_inner, oc, oa, adjacency, deadline - 12, max_iter=10000)

        # Phase 3: Polish inner positions
        inner3 = polish_inner_positions(inner2, oc2, oa2, deadline - 6, max_iter=6000)

        # Phase 4: Polish outer container
        oc3, oa3, side3 = polish_outer_container(inner3, oc2, oa2, deadline - 2)

        # Check disjointness
        if check_all_disjoint(inner3, eps=1e-4):
            if side3 < best_global_side:
                best_global_side = side3
                best_global_inner = inner3
                best_global_center = oc3
                best_global_angle = oa3
        else:
            # Try to fix
            fixed = try_fix_overlaps(inner3, oc3, oa3)
            if check_all_disjoint(fixed, eps=1e-5):
                side_f = compute_required_side(fixed, oc3, oa3)
                if side_f < best_global_side:
                    best_global_side = side_f
                    best_global_inner = fixed
                    best_global_center = oc3
                    best_global_angle = oa3
            else:
                # Fall back to initial
                side_init = compute_required_side(initial_inner, (cx_avg, cy_avg), 0.0)
                if side_init < best_global_side:
                    best_global_side = side_init
                    best_global_inner = initial_inner
                    best_global_center = (cx_avg, cy_avg)
                    best_global_angle = 0.0

    # Fallback
    if best_global_inner is None:
        centers = build_axial_layout(candidates[0][0][:N])
        best_global_inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_global_center = (cx_avg, cy_avg)
        best_global_angle = 0.0
        best_global_side = compute_required_side(best_global_inner, best_global_center, best_global_angle)

    return best_global_inner, best_global_center, best_global_side, best_global_angle

# EVOLVE-BLOCK-END
