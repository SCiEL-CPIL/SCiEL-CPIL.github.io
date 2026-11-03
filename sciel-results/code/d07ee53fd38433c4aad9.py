"""Rewritten program for Heilbronn convex regions task."""
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    n = len(pts)
    min_area = np.inf
    # Use vectorized computation for speed
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shapely-like computation
        from scipy.spatial import ConvexHull as CH
        try:
            h = CH(pts)
            return h.volume
        except:
            return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_a


def _fermat_spiral_init(n, center=(0, 0), radius=1.0):
    """Generate points using Fermat/Vogel spiral."""
    i = np.arange(n)
    r = radius * np.sqrt(i / n)
    theta = i * 2.399963229728653  # Golden angle in radians
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack((x, y))


def _concentric_circle_init(n, center=(0, 0), outer_radius=1.0):
    """
    Initialize with concentric circles: 
    - 1 point at center
    - 6 points on inner circle
    - 6 points on outer circle
    This exploits known Heilbronn structure.
    """
    pts = [np.array(center, dtype=float)]
    # Inner ring: 6 points
    r_inner = outer_radius * 0.5
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append(np.array([center[0] + r_inner * np.cos(angle),
                             center[1] + r_inner * np.sin(angle)]))
    # Outer ring: 6 points (offset by 30 degrees)
    r_outer = outer_radius
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        pts.append(np.array([center[0] + r_outer * np.cos(angle),
                             center[1] + r_outer * np.sin(angle)]))
    return np.array(pts)


def _perturb_worst_triangles(pts, num_perturbations=50, step_size=0.05):
    """
    Perturb points that are part of minimum-area triangles.
    Returns a new point set.
    """
    n = len(pts)
    pts = pts.copy()
    
    # Find all triangles and their areas
    best_tri = None
    best_area = np.inf
    all_tris = []
    
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                all_tris.append((area, i, j, k))
                if area < best_area:
                    best_area = area
                    best_tri = (i, j, k)
    
    # Sort by area, take worst ones
    all_tris.sort(key=lambda x: x[0])
    
    rng = np.random.default_rng(42)
    for _ in range(num_perturbations):
        # Pick a random triangle from the worst 20
        idx = rng.integers(0, min(20, len(all_tris)))
        _, i, j, k = all_tris[idx]
        # Perturb one of the three points
        point_idx = [i, j, k][rng.integers(0, 3)]
        pts[point_idx] += rng.normal(0, step_size, size=2)
    
    return pts


def _local_search(pts, max_iters=200, step_size=0.02, cooling=0.95):
    """
    Local search: iteratively perturb points in worst triangles,
    accept if score improves.
    """
    current = pts.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    n = len(current)
    all_tri_indices = list(itertools.combinations(range(n), 3))
    
    def compute_all_areas(pts):
        areas = np.zeros(len(all_tri_indices))
        for idx, (i, j, k) in enumerate(all_tri_indices):
            ax, ay = pts[i]
            bx, by = pts[j]
            cx, cy = pts[k]
            areas[idx] = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        return areas
    
    step = step_size
    for it in range(max_iters):
        areas = compute_all_areas(current)
        min_area = np.min(areas)
        # Find all triangles with area close to minimum
        worst_mask = areas < min_area * 1.05
        worst_indices = np.where(worst_mask)[0]
        
        if len(worst_indices) == 0:
            break
        
        # Pick a random worst triangle
        tri_idx = worst_indices[np.random.default_rng(it).integers(0, len(worst_indices))]
        i, j, k = all_tri_indices[tri_idx]
        
        # Try perturbing each of the 3 points in 8 directions
        improved = False
        for point_idx in [i, j, k]:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    trial = current.copy()
                    trial[point_idx] += step * np.array([dx, dy])
                    trial_score = _score(trial)
                    if trial_score > current_score:
                        current = trial
                        current_score = trial_score
                        if current_score > best_score:
                            best = current.copy()
                            best_score = current_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            step *= cooling
            if step < 0.001:
                break
    
    return best, best_score


def construct():
    """
    Construct 13 points in unit-area convex region to maximize
    minimum triangle area / convex hull area.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_pts = None
    best_score = 0.0
    
    # Multiple initialization strategies
    inits = []
    
    # 1. Fermat spiral
    pts1 = _fermat_spiral_init(N, center=(0, 0), radius=1.0)
    inits.append(pts1)
    
    # 2. Concentric circles
    pts2 = _concentric_circle_init(N, center=(0, 0), outer_radius=1.0)
    inits.append(pts2)
    
    # 3. Regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    pts3 = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(pts3)
    
    # 4. Random with seed
    rng = np.random.default_rng(N)
    pts4 = rng.random((N, 2)) * 2 - 1
    inits.append(pts4)
    
    # Evaluate all inits
    for pts in inits:
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Local search from each init
    for pts in inits:
        if time.time() - start_time > deadline:
            break
        refined, refined_score = _local_search(pts, max_iters=300, step_size=0.03, cooling=0.9)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined.copy()
    
    # Additional refinement with smaller steps
    if best_pts is not None and time.time() - start_time < deadline - 20:
        refined2, refined2_score = _local_search(best_pts, max_iters=500, step_size=0.01, cooling=0.9)
        if refined2_score > best_score:
            best_score = refined2_score
            best_pts = refined2.copy()
    
    # Final check
    if best_pts is None:
        best_pts = inits[0]
    
    return best_pts

# EVOLVE-BLOCK-END
