# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from itertools import combinations

N = 13
# Pre-compute all C(13,3) = 286 triangle index triples
TRI_IDX = np.array(list(combinations(range(N), 3)), dtype=np.int32)

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull (vectorized).
    """
    n = len(points)
    if n < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 1e-12:
        return 0.0

    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]

    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5

    A_min = np.min(areas)
    if A_min <= 1e-12:
        return 0.0
    return A_min / A_hull

def compute_areas(points):
    """Return all 286 triangle areas."""
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.abs(cross) * 0.5

def delaunay_repair(points, time_limit, max_iter=8000):
    """
    Delaunay-based local repair: iteratively move vertices incident to
    minimum-area triangles toward the centroid of their adjacent triangles
    to increase A_min while preserving hull structure.
    """
    points = points.copy()
    current_score = compute_score(points)
    best_score = current_score
    best_points = points.copy()

    no_improve = 0
    step_scale = 0.3

    for iteration in range(max_iter):
        if time.time() > time_limit:
            break

        areas = compute_areas(points)
        min_area = np.min(areas)
        if min_area <= 1e-10:
            break

        # Get indices of triangles with area below threshold (bottom 15%)
        threshold = min_area * 1.15
        low_tri_indices = np.where(areas < threshold)[0]

        if len(low_tri_indices) == 0:
            break

        # Collect vertices that are in low-area triangles
        low_vertices = set()
        for tri_idx in low_tri_indices:
            tri = TRI_IDX[tri_idx]
            low_vertices.add(tri[0])
            low_vertices.add(tri[1])
            low_vertices.add(tri[2])

        # Compute Delaunay triangulation to find adjacent triangles
        try:
            tri = Delaunay(points)
        except Exception:
            # Fallback: just move toward overall centroid
            centroid = points.mean(axis=0)
            for v in low_vertices:
                points[v] += step_scale * (centroid - points[v])
            new_score = compute_score(points)
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
                no_improve = 0
            else:
                no_improve += 1
                step_scale *= 0.95
            continue

        # For each low-area vertex, find its adjacent triangles and move toward their centroid
        for v in low_vertices:
            # Find all Delaunay triangles containing vertex v
            adj_tris = []
            for t_idx in range(len(tri.simplices)):
                simplex = tri.simplices[t_idx]
                if v in simplex:
                    adj_tris.append(simplex)

            if len(adj_tris) == 0:
                continue

            # Compute centroid of adjacent triangles
            adj_centroid = np.mean([points[t].mean(axis=0) for t in adj_tris], axis=0)

            # Move vertex toward adjacent centroid
            direction = adj_centroid - points[v]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                points[v] += step_scale * direction

        new_score = compute_score(points)
        if new_score > best_score:
            best_score = new_score
            best_points = points.copy()
            no_improve = 0
        else:
            no_improve += 1
            step_scale *= 0.97
            # Revert to best if no improvement for a while
            if no_improve > 40:
                points = best_points.copy()
                step_scale = 0.3
                no_improve = 0

        if step_scale < 1e-6:
            break

    return best_points, best_score

def nelder_mead_optimize(points, time_limit, temperature=0.01, max_iter=5000):
    """
    Nelder-Mead optimization on the soft-min surrogate, then refine with hard min.
    """
    points = points.copy()
    n = len(points)
    flat = points.flatten()

    # Create initial simplex
    simplex = [flat.copy()]
    step = 0.1
    for i in range(2 * n):
        p = flat.copy()
        p[i] += step
        simplex.append(p)

    def objective(flat_arr):
        pts = flat_arr.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        try:
            hull = ConvexHull(pts)
            A_hull = hull.volume
        except Exception:
            return 1e10
        if A_hull <= 0:
            return 1e10

        p1 = pts[TRI_IDX[:, 0]]
        p2 = pts[TRI_IDX[:, 1]]
        p3 = pts[TRI_IDX[:, 2]]

        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = np.abs(cross) * 0.5

        scaled = -areas / temperature
        max_scaled = np.max(scaled)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min_area = -temperature * log_sum

        return -soft_min_area / A_hull

    best_flat = flat.copy()
    best_val = objective(flat)

    # Nelder-Mead iterations
    for iteration in range(max_iter):
        if time.time() > time_limit:
            break

        # Sort simplex
        vals = [objective(s) for s in simplex]
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]

        if vals[0] < best_val:
            best_val = vals[0]
            best_flat = simplex[0].copy()

        # Centroid of all but worst
        worst = simplex[-1]
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        alpha = 1.0
        reflected = centroid + alpha * (centroid - worst)
        ref_val = objective(reflected)

        if ref_val < vals[0]:
            # Expansion
            gamma = 2.0
            expanded = centroid + gamma * (centroid - worst)
            exp_val = objective(expanded)
            if exp_val < ref_val:
                simplex[-1] = expanded
            else:
                simplex[-1] = reflected
        elif ref_val < vals[-2]:
            simplex[-1] = reflected
        else:
            # Contraction
            rho = 0.5
            contracted = centroid + rho * (worst - centroid)
            con_val = objective(contracted)
            if con_val < vals[-1]:
                simplex[-1] = contracted
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])

    # Convert back to points
    best_points = best_flat.reshape(n, 2)

    # Polish with hard min via local search
    hard_score = compute_score(best_points)

    return best_points, hard_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.

    Uses Delaunay-based constrained repair as the primary strategy, with
    random perturbation and Nelder-Mead as fallbacks.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit

    n = N

    # Best configuration found so far
    best_points = None
    best_score = 0.0

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Try multiple initializations
    initializations = []

    # 1. Hexagonal lattice pattern
    hex_pts = np.zeros((n, 2))
    hex_pts[0] = [0.0, 0.0]
    for i in range(6):
        angle = i * np.pi / 3
        hex_pts[i + 1] = [np.cos(angle), np.sin(angle)]
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        hex_pts[i + 7] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
    initializations.append(hex_pts)

    # 2. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(reg_pts)

    # 3. Grid-based
    grid_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [0.5, 0.5],
        [1.5, 0.5],
        [0.5, 1.5],
        [1.5, 1.5],
    ], dtype=float)
    initializations.append(grid_pts)

    # 4. Random points in unit square
    rng = np.random.default_rng(seed=12345)
    for _ in range(5):
        initializations.append(rng.random((n, 2)))

    # 5. Perturbed hexagonal
    for _ in range(3):
        initializations.append(hex_pts + rng.normal(size=(n, 2)) * 0.1)

    # Evaluate initializations
    for init_pts in initializations:
        if time.time() > deadline - 60:
            break
        s = compute_score(init_pts)
        update_best(init_pts, s)

    # Primary: Delaunay-based repair from best initialization
    if best_points is not None:
        dl_pts, dl_score = delaunay_repair(best_points, deadline - 30)
        update_best(dl_pts, dl_score)

    # Secondary: Nelder-Mead from best
    if time.time() < deadline - 60 and best_points is not None:
        nm_pts, nm_score = nelder_mead_optimize(best_points, deadline - 30)
        update_best(nm_pts, nm_score)

    # Tertiary: Random restarts with Delaunay repair
    for restart in range(8):
        if time.time() > deadline - 30:
            break

        # Start from a random point or perturbed best
        if restart < 4:
            perturbed = best_points + rng.normal(size=(n, 2)) * 0.2
        else:
            perturbed = rng.random((n, 2))

        s = compute_score(perturbed)
        if s > best_score:
            best_points = perturbed.copy()
            best_score = s

        # Quick Delaunay repair
        dl_pts, dl_score = delaunay_repair(perturbed, deadline - 30, max_iter=3000)
        update_best(dl_pts, dl_score)

    # Final validation
    if best_points is None or not np.all(np.isfinite(best_points)):
        fallback = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
        ], dtype=float)
        return fallback

    final_score = compute_score(best_points)
    if final_score <= 0:
        fallback = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
        ], dtype=float)
        return fallback

    return best_points

# EVOLVE-BLOCK-END
