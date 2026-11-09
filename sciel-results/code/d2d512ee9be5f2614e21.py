import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    sqrt3 = math.sqrt(3.0)

    # Precompute normals for outer hexagon (flat-top, centered at origin)
    outer_normals = np.array([[math.cos(math.radians(30 + 60*k)),
                               math.sin(math.radians(30 + 60*k))] for k in range(6)])

    def get_hex_vertices_np(cx, cy, angle_deg):
        """Return 6x2 array of vertices for a unit hexagon."""
        angles = np.radians(angle_deg + 60 * np.arange(6))
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

    def get_all_vertices(params):
        """Given params of length 3*N, return list of 6x2 vertex arrays."""
        verts = []
        for i in range(N):
            x, y, th = params[3*i], params[3*i+1], params[3*i+2]
            verts.append(get_hex_vertices_np(x, y, th))
        return verts

    def compute_L(verts_list):
        """Compute minimum outer hexagon side length to contain all vertices."""
        all_pts = np.vstack(verts_list)
        projections = all_pts @ outer_normals.T  # (36, 6)
        max_proj = projections.max()
        return max_proj / (sqrt3 / 2.0)

    def compute_overlap_penalty(verts_list, smooth=True):
        """Compute smooth overlap penalty between all pairs of hexagons."""
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                v1 = verts_list[i]
                v2 = verts_list[j]
                # Check if any vertex of v1 is inside v2
                # Use winding/ray-cast via cross products for convex polygons
                # For two convex polygons, they overlap iff:
                # - any vertex of one is inside the other, OR
                # - edges intersect
                # Simple approach: check point-in-polygon for all 12 vertices
                for v in v1:
                    if point_in_convex_np(v, v2):
                        d = point_to_convex_boundary_dist_np(v, v2)
                        penalty += d * d
                for v in v2:
                    if point_in_convex_np(v, v1):
                        d = point_to_convex_boundary_dist_np(v, v1)
                        penalty += d * d
        return penalty

    def point_in_convex_np(pt, poly):
        """Check if point is inside convex polygon (6 vertices)."""
        # For convex polygon, point is inside iff it's on the same side of all edges
        for k in range(6):
            p1 = poly[k]
            p2 = poly[(k + 1) % 6]
            edge = p2 - p1
            to_pt = pt - p1
            cross = edge[0] * to_pt[1] - edge[1] * to_pt[0]
            if cross < -1e-10:
                return False
        return True

    def point_to_convex_boundary_dist_np(pt, poly):
        """Distance from point to boundary of convex polygon."""
        min_d = float('inf')
        for k in range(6):
            p1 = poly[k]
            p2 = poly[(k + 1) % 6]
            # Project pt onto segment p1-p2
            d = p2 - p1
            len_sq = d @ d
            if len_sq < 1e-20:
                dist = np.linalg.norm(pt - p1)
            else:
                t = np.dot(pt - p1, d) / len_sq
                t = max(0.0, min(1.0, t))
                proj = p1 + t * d
                dist = np.linalg.norm(pt - proj)
            if dist < min_d:
                min_d = dist
        return min_d

    def objective(params):
        if time.time() > deadline:
            return 1e10
        verts = get_all_vertices(params)
        L = compute_L(verts)
        pen = compute_overlap_penalty(verts)
        return L + 10.0 * pen

    # Epigraph formulation: minimize t subject to L(params) <= t and overlap_penalty(params) <= eps
    # We'll use a combined smooth objective instead for simplicity with scipy

    # Initializations

    # 1. Hexagonal lattice: 12 closest centers from a hex grid
    grid_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                grid_centers.append((x, y))

    dists = [(math.hypot(x, y), i) for i, (x, y) in enumerate(grid_centers)]
    dists.sort()
    init_grid_idx = [d[1] for d in dists[:12]]

    params_grid = []
    for i in init_grid_idx:
        x, y = grid_centers[i]
        params_grid.extend([x, y, 0.0])

    # 2. Ring-based initialization: 6 inner + 6 outer
    params_ring = []
    # Inner ring: 6 hexagons at radius ~1.8
    for k in range(6):
        ang = math.radians(60 * k)
        x = 1.8 * math.cos(ang)
        y = 1.8 * math.sin(ang)
        params_ring.extend([x, y, 0.0])
    # Outer ring: 6 hexagons at radius ~3.2
    for k in range(6):
        ang = math.radians(60 * k + 30)
        x = 3.2 * math.cos(ang)
        y = 3.2 * math.sin(ang)
        params_ring.extend([x, y, 0.0])

    # 3. Random initialization
    np.random.seed(42)
    params_random = []
    for _ in range(N):
        x = np.random.uniform(-2.5, 2.5)
        y = np.random.uniform(-2.5, 2.5)
        th = np.random.uniform(0, 360)
        params_random.extend([x, y, th])

    best_params = None
    best_L = float('inf')
    best_penalty = float('inf')

    def evaluate(params):
        """Evaluate and return (L, penalty, is_valid)."""
        verts = get_all_vertices(params)
        L = compute_L(verts)
        pen = compute_overlap_penalty(verts)
        is_valid = (pen < 1e-6)
        return L, pen, is_valid

    # Try multiple initializations with Nelder-Mead
    inits = [params_grid, params_ring, params_random]
    init_names = ["grid", "ring", "random"]

    for idx, (init_params, name) in enumerate(zip(inits, init_names)):
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-6, 'fatol': 1e-6,
                                    'adaptive': True})
            if time.time() > deadline - 10:
                break
            L, pen, valid = evaluate(res.x)
            if valid and L < best_L:
                best_L = L
                best_penalty = pen
                best_params = res.x.copy()
        except Exception:
            pass

    # Polish pass: if we have a valid solution, do a tighter optimization
    if best_params is not None and time.time() < deadline - 15:
        try:
            # Use tighter tolerances
            res2 = minimize(objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8,
                                     'adaptive': True})
            L2, pen2, valid2 = evaluate(res2.x)
            if valid2 and L2 < best_L:
                best_L = L2
                best_penalty = pen2
                best_params = res2.x.copy()
        except Exception:
            pass

    # Fallback: if no valid solution, use grid
    if best_params is None:
        best_params = np.array(params_grid, dtype=float)
        best_L, best_penalty, _ = evaluate(best_params)

    # Build output
    positions = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        th = best_params[3*i+2]
        positions.append([x, y, th])

    return positions, (0.0, 0.0), best_L, 0.0
