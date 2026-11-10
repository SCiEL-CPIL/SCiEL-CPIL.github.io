# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def heilbronn_convex13():
    """
    Maximize the ratio of the smallest triangle area to the convex hull area
    for 13 points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Precompute triangle indices
    tri_indices = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    n_tri = len(tri_indices)

    # Vectorized area calculation
    def calc_areas(points):
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return np.abs(cross) * 0.5

    def calc_hull_area(points):
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def evaluate(points):
        areas = calc_areas(points)
        min_area = np.min(areas)
        if min_area <= 0:
            return 0.0
        hull_area = calc_hull_area(points)
        if hull_area <= 0:
            return 0.0
        return min_area / hull_area

    def evaluate_fast(points):
        """
        Fast evaluation without convex hull, assuming points are in a reasonable convex position.
        We approximate hull area as the area of the bounding box or use a fixed reference.
        Actually, for local search, the hull area changes slowly.
        But to be safe, we'll compute hull area every few steps or use a surrogate.
        However, the score is min_area / hull_area.
        If we just maximize min_area, we might push points to corners, increasing hull area.
        But the ratio is scale invariant.
        
        Let's use a surrogate: The area of the convex hull is roughly proportional to the spread.
        A better surrogate: The area of the polygon formed by the points in order? No.
        
        Let's stick to the exact evaluation but optimize the loop.
        """
        return evaluate(points)

    # Initialization strategies
    def init_circle():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = 0.5 + 0.4 * np.cos(angles)
        y = 0.5 + 0.4 * np.sin(angles)
        return np.column_stack((x, y))

    def init_hex_grid():
        points = []
        s = 0.25
        for row in range(-2, 3):
            for col in range(-2, 3):
                x = 0.5 + col * s + (row % 2) * s / 2
                y = 0.5 + row * s * np.sqrt(3) / 2
                points.append([x, y])
        points = np.array(points)
        mask = (points[:, 0] - 0.5)**2 + (points[:, 1] - 0.5)**2 <= 0.45**2
        points = points[mask]
        if len(points) > n:
            idx = np.random.choice(len(points), n, replace=False)
            points = points[idx]
        while len(points) < n:
            p = np.random.random(2) * 0.8 + 0.1
            points = np.vstack([points, p])
        return points[:n]

    def init_random():
        return np.random.random((n, 2)) * 0.8 + 0.1

    def init_best_known():
        # Use a known good approximation
        # Heilbronn 13 points known configuration is complex.
        # Let's use a grid-like structure that is well-spaced.
        # A 3x5 grid is 15 points, we take 13.
        pts = []
        # Centered grid
        for i in range(3):
            for j in range(5):
                pts.append([0.15 + i * 0.35, 0.15 + j * 0.15])
        pts = np.array(pts)
        return pts[:n]

    # Advanced Local Search with adaptive step size
    def local_search(points, max_iters=2000, initial_step=0.05):
        current_points = points.copy()
        current_score = evaluate(current_points)
        step = initial_step
        min_step = 1e-6
        stall = 0
        best_score = current_score
        best_points = current_points.copy()

        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Randomize point order
            order = np.random.permutation(n)
            
            for i in order:
                if time.time() > deadline:
                    break
                
                # Try moving point i in 8 directions
                # Plus random perturbation
                directions = [
                    [1, 0], [-1, 0], [0, 1], [0, -1],
                    [1, 1], [-1, -1], [1, -1], [-1, 1]
                ]
                # Normalize
                for d in directions:
                    d = np.array(d)
                    d = d / np.linalg.norm(d)
                    
                    new_point = current_points.copy()
                    new_point[i] += step * d
                    # Keep within bounds [0, 1]
                    np.clip(new_point, 0, 1, out=new_point)
                    
                    new_score = evaluate(new_point)
                    if new_score > current_score:
                        current_points = new_point
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_points.copy()
                        break
                if improved:
                    break
            
            if not improved:
                # Try a random jump
                idx = np.random.randint(n)
                new_point = current_points.copy()
                new_point[idx] += np.random.normal(0, step, 2)
                np.clip(new_point, 0, 1, out=new_point)
                new_score = evaluate(new_point)
                if new_score > current_score:
                    current_points = new_point
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
                
            if not improved:
                stall += 1
                if stall > 5:
                    step *= 0.7
                    stall = 0
                    if step < min_step:
                        break
            else:
                # Occasionally increase step if improving rapidly?
                # Or just keep it.
                pass

        return best_points, best_score

    best_points = None
    best_score = 0.0

    # Generate initial candidates
    candidates = [
        init_circle(),
        init_hex_grid(),
        init_random(),
        init_best_known()
    ]
    
    # Add more random starts
    for _ in range(10):
        candidates.append(init_random())
    
    # Evaluate all candidates
    scored_cands = []
    for cand in candidates:
        if time.time() > deadline - 20:
            break
        if not np.all(np.isfinite(cand)):
            continue
        try:
            s = evaluate(cand)
            if s > 0:
                scored_cands.append((s, cand))
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
        except Exception:
            pass

    # Sort candidates by score
    scored_cands.sort(key=lambda x: -x[0])
    
    # Refine top candidates
    # We have ~14 candidates. Let's refine the top 5.
    for s, cand in scored_cands[:5]:
        if time.time() > deadline - 30:
            break
        refined_pts, refined_score = local_search(cand, max_iters=1500)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()

    # If we have time, do a final polish on the best
    if best_points is not None and time.time() < deadline - 10:
        final_pts, final_score = local_search(best_points, max_iters=2000, initial_step=0.01)
        if final_score > best_score:
            best_points = final_pts
            best_score = final_score

    if best_points is None:
        best_points = init_circle()

    return best_points
# EVOLVE-BLOCK-END
