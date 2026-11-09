# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import minimize
import time
from itertools import combinations


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize
    the area of the smallest triangle formed by these points.

    Strategy:
    - Start with structured topologies (regular polygon with interior points, concentric rings).
    - Use Lloyd's algorithm (Voronoi centroid iteration) to relax interior points for uniform density.
    - Use a soft-min surrogate for the objective during optimization for smoother landscape.
    - Refine with exact minimum triangle area using Powell's method.
    - Time-budgeted search with deadline guard.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep margin

    def min_triangle_area(pts):
        """Compute the minimum area among all triangles formed by 3 points."""
        areas = []
        for i, j, k in combinations(range(len(pts)), 3):
            p1, p2, p3 = pts[i], pts[j], pts[k]
            area = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0
            areas.append(area)
        return min(areas) if areas else 0.0

    def hull_area(pts):
        """Compute the area of the convex hull."""
        try:
            hull = ConvexHull(pts)
            return hull.volume  # In 2D, volume = area
        except Exception:
            return 0.0

    def score(pts):
        """Compute A_min / A_hull."""
        ah = hull_area(pts)
        if ah <= 1e-12:
            return 0.0
        am = min_triangle_area(pts)
        return am / ah

    def soft_min_area(pts, temp=0.05):
        """Soft minimum of triangle areas for smoother optimization."""
        areas = []
        for i, j, k in combinations(range(len(pts)), 3):
            p1, p2, p3 = pts[i], pts[j], pts[k]
            area = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0
            areas.append(area)
        areas = np.array(areas)
        return temp * np.log(np.sum(np.exp(-areas / temp)))

    def objective_soft(x, temp=0.05):
        """Objective for optimization: minimize -soft_min_area / hull_area."""
        pts = x.reshape(-1, 2)
        ah = hull_area(pts)
        if ah <= 1e-12:
            return 1e6
        sm = soft_min_area(pts, temp)
        return -sm / ah

    def objective_exact(x):
        """Objective for exact optimization: minimize -min_triangle_area / hull_area."""
        pts = x.reshape(-1, 2)
        ah = hull_area(pts)
        if ah <= 1e-12:
            return 1e6
        am = min_triangle_area(pts)
        return -am / ah

    def lloyd_relax(pts, n_iter=50, interior_indices=None):
        """
        Lloyd's algorithm: move interior points to centroids of their Voronoi cells.
        Only interior points are moved; hull points stay fixed.
        """
        if interior_indices is None:
            interior_indices = list(range(len(pts)))
        
        pts = pts.copy()
        for _ in range(n_iter):
            try:
                vor = Voronoi(pts)
            except Exception:
                break
            
            for idx in interior_indices:
                # Find the Voronoi region for this point
                region_idx = vor.point_region[idx]
                region = vor.regions[region_idx]
                
                if -1 in region or len(region) == 0:
                    continue
                
                # Get vertices of the Voronoi cell
                cell_verts = vor.vertices[region]
                if len(cell_verts) == 0:
                    continue
                
                # Compute centroid of the cell
                centroid = np.mean(cell_verts, axis=0)
                pts[idx] = centroid
        
        return pts

    def generate_regular_polygon_with_interior(n_hull=7, n_interior=6):
        """Generate a regular polygon with interior points arranged in rings."""
        pts = []
        # Hull: regular polygon
        for i in range(n_hull):
            angle = 2 * np.pi * i / n_hull
            pts.append([np.cos(angle), np.sin(angle)])
        
        # Interior: concentric rings
        if n_interior >= 3:
            # Ring 1: 3 points at radius 0.5
            for i in range(3):
                angle = 2 * np.pi * i / 3 + np.pi / 6
                r = 0.5
                pts.append([r * np.cos(angle), r * np.sin(angle)])
        
        if n_interior >= 6:
            # Ring 2: 3 points at radius 0.25
            for i in range(3):
                angle = 2 * np.pi * i / 3 + np.pi / 6 + np.pi / 3
                r = 0.25
                pts.append([r * np.cos(angle), r * np.sin(angle)])
        
        return np.array(pts[:n])

    def generate_concentric_rings(radii=[0.0, 0.4, 0.8], counts=[1, 4, 8]):
        """Generate points on concentric rings."""
        pts = []
        for r, count in zip(radii, counts):
            for i in range(count):
                angle = 2 * np.pi * i / count
                if r == 0:
                    pts.append([0.0, 0.0])
                else:
                    pts.append([r * np.cos(angle), r * np.sin(angle)])
        
        return np.array(pts[:n])

    def generate_grid_based():
        """Generate a structured grid-like arrangement."""
        pts = []
        # 4x4 grid clipped to circle, take 13 points
        xs = np.linspace(-0.8, 0.8, 4)
        ys = np.linspace(-0.8, 0.8, 4)
        for x in xs:
            for y in ys:
                if x**2 + y**2 <= 1.0:
                    pts.append([x, y])
        
        if len(pts) < 13:
            # Add more points
            for x in xs:
                for y in ys:
                    if x**2 + y**2 <= 1.0 and [x, y] not in pts:
                        pts.append([x, y])
        
        # If still not enough, add center and nearby
        while len(pts) < 13:
            pts.append([0.0, 0.0])
        
        return np.array(pts[:13])

    def generate_random_structured(seed):
        """Generate a random but structured arrangement."""
        rng = np.random.default_rng(seed)
        pts = []
        # Outer ring
        n_outer = 7
        for i in range(n_outer):
            angle = 2 * np.pi * i / n_outer + rng.uniform(-0.2, 0.2)
            r = 1.0 + rng.uniform(-0.05, 0.05)
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        
        # Inner ring
        n_inner = 6
        for i in range(n_inner):
            angle = 2 * np.pi * i / n_inner + np.pi / n_inner + rng.uniform(-0.2, 0.2)
            r = 0.5 + rng.uniform(-0.1, 0.1)
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        
        return np.array(pts[:13])

    # Generate initial candidates from different structured topologies
    candidates = []
    
    # Regular polygon with interior
    c1 = generate_regular_polygon_with_interior(7, 6)
    candidates.append(c1)
    
    # Concentric rings
    c2 = generate_concentric_rings([0.0, 0.4, 0.8], [1, 4, 8])
    candidates.append(c2)
    
    # Grid-based
    c3 = generate_grid_based()
    candidates.append(c3)
    
    # Random structured with different seeds
    for seed in [1, 2, 3, 4, 5]:
        c = generate_random_structured(seed)
        candidates.append(c)

    # Evaluate initial candidates
    best_pts = None
    best_score = -1.0
    for cand in candidates:
        s = score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand.copy()

    # Apply Lloyd relaxation to the best candidate
    if best_pts is not None:
        # Identify interior points (those not on convex hull)
        try:
            hull = ConvexHull(best_pts)
            hull_indices = set(hull.vertices.tolist())
            interior_indices = [i for i in range(n) if i not in hull_indices]
        except Exception:
            interior_indices = list(range(n))
        
        relaxed = lloyd_relax(best_pts, n_iter=100, interior_indices=interior_indices)
        s_relaxed = score(relaxed)
        if s_relaxed > best_score:
            best_score = s_relaxed
            best_pts = relaxed.copy()

    # Optimize using Powell's method with soft-min objective
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:  # Leave 60s margin
        x0 = best_pts.flatten()
        try:
            result = minimize(
                objective_soft,
                x0,
                method='Powell',
                options={'maxiter': 200, 'ftol': 1e-8, 'xtol': 1e-8},
                args=(0.05,)
            )
            if result.success or result.fun < objective_soft(x0):
                new_pts = result.x.reshape(-1, 2)
                s_new = score(new_pts)
                if s_new > best_score:
                    best_score = s_new
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Further refine with exact objective
    elapsed = time.time() - start_time
    if elapsed < deadline - 30 and best_pts is not None:
        x0 = best_pts.flatten()
        try:
            result = minimize(
                objective_exact,
                x0,
                method='Powell',
                options={'maxiter': 100, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            if result.success or result.fun < objective_exact(x0):
                new_pts = result.x.reshape(-1, 2)
                s_new = score(new_pts)
                if s_new > best_score:
                    best_score = s_new
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Try multiple random restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 90:
        for seed in range(10, 30):
            elapsed = time.time() - start_time
            if elapsed > deadline - 60:
                break
            
            cand = generate_random_structured(seed)
            # Quick Lloyd relaxation
            try:
                hull = ConvexHull(cand)
                hull_indices = set(hull.vertices.tolist())
                interior_indices = [i for i in range(n) if i not in hull_indices]
            except Exception:
                interior_indices = list(range(n))
            
            relaxed = lloyd_relax(cand, n_iter=30, interior_indices=interior_indices)
            s = score(relaxed)
            if s > best_score:
                best_score = s
                best_pts = relaxed.copy()
                
                # Quick optimization
                x0 = best_pts.flatten()
                try:
                    result = minimize(
                        objective_soft,
                        x0,
                        method='Powell',
                        options={'maxiter': 50, 'ftol': 1e-7, 'xtol': 1e-7},
                        args=(0.05,)
                    )
                    new_pts = result.x.reshape(-1, 2)
                    s_new = score(new_pts)
                    if s_new > best_score:
                        best_score = s_new
                        best_pts = new_pts.copy()
                except Exception:
                    pass

    # Final check: ensure validity
    if best_pts is None:
        # Fallback to random
        rng = np.random.default_rng(42)
        best_pts = rng.random((n, 2))
    
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = np.nan_to_num(best_pts, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return best_pts


# EVOLVE-BLOCK-END
