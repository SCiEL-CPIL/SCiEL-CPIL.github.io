# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
from itertools import combinations


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 620  # Leave 10s margin from 630s

    # Pre-compute all 286 triangle index triples
    tri_indices = list(combinations(range(n), 3))
    tri_arr = np.array(tri_indices)
    i_idx = tri_arr[:, 0]
    j_idx = tri_arr[:, 1]
    k_idx = tri_arr[:, 2]

    def compute_all_areas(points):
        """Compute areas of all 286 triangles. Returns array of shape (286,)."""
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        areas = 0.5 * np.abs(pi[:, 0] * (pj[:, 1] - pk[:, 1]) +
                              pj[:, 0] * (pk[:, 1] - pi[:, 1]) +
                              pk[:, 0] * (pi[:, 1] - pj[:, 1]))
        return areas

    def compute_hull_area(points):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def compute_score(points):
        """Compute A_min / A_hull."""
        points = np.asarray(points)
        if not np.all(np.isfinite(points)):
            return -1.0
        areas = compute_all_areas(points)
        min_area = np.min(areas)
        if min_area < 1e-12:
            return -1.0
        hull_area = compute_hull_area(points)
        if hull_area < 1e-12:
            return -1.0
        return min_area / hull_area

    def make_triangular_lattice(seed=0):
        """
        Create a triangular lattice configuration for 13 points.
        Triangular lattice is known to be optimal for large n in Heilbronn's problem.
        """
        rng = np.random.default_rng(seed)
        
        # Start with a hexagonal/triangular lattice pattern
        # Center point + 6 points in first ring + 6 points in second ring
        pts = []
        
        # Center
        pts.append([0.0, 0.0])
        
        # First ring: 6 points at distance 1
        for k in range(6):
            angle = k * np.pi / 3
            pts.append([np.cos(angle), np.sin(angle)])
        
        # Second ring: 6 points at distance 2, offset by pi/6
        for k in range(6):
            angle = k * np.pi / 3 + np.pi / 6
            pts.append([2 * np.cos(angle), 2 * np.sin(angle)])
        
        points = np.array(pts, dtype=float)
        
        # Center and normalize
        points -= points.mean(axis=0)
        scale = np.max(np.abs(points))
        if scale > 0:
            points /= scale
        
        # Add small perturbation to break symmetry
        points += rng.normal(0, 0.01, (n, 2))
        
        return points

    def make_hexagonal_cluster(seed=0):
        """
        Alternative hexagonal cluster arrangement.
        """
        rng = np.random.default_rng(seed)
        
        pts = []
        # 3x3 grid with hexagonal offset
        for i in range(3):
            for j in range(3):
                x = j * 1.0
                y = i * np.sqrt(3) / 2
                if i % 2 == 1:
                    x += 0.5
                pts.append([x, y])
        
        # Add 4 more points to reach 13
        # Add corners
        pts.append([-0.5, -np.sqrt(3)/2])
        pts.append([3.5, -np.sqrt(3)/2])
        pts.append([-0.5, 3*np.sqrt(3)/2])
        pts.append([3.5, 3*np.sqrt(3)/2])
        
        points = np.array(pts, dtype=float)
        
        # Center and normalize
        points -= points.mean(axis=0)
        scale = np.max(np.abs(points))
        if scale > 0:
            points /= scale
        
        # Add small perturbation
        points += rng.normal(0, 0.01, (n, 2))
        
        return points

    def delaunay_guided_local_search(points, max_iter=2000, step_size=0.01):
        """
        Local search guided by Delaunay triangulation.
        Moves points to increase the minimum triangle area.
        """
        points = points.copy()
        rng = np.random.default_rng(123)
        
        best_score = compute_score(points)
        best_points = points.copy()
        
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Compute current score
            current_score = compute_score(points)
            
            # Get Delaunay triangulation
            try:
                tri = Delaunay(points)
            except Exception:
                break
            
            # Identify the smallest triangle
            areas = compute_all_areas(points)
            min_idx = np.argmin(areas)
            min_area = areas[min_idx]
            
            if min_area < 1e-10:
                break
            
            # Get the vertices of the smallest triangle
            tri_i, tri_j, tri_k = tri_arr[min_idx]
            
            # Try to move one of these vertices to increase the area
            # Use gradient-free local search
            for vertex_idx in [tri_i, tri_j, tri_k]:
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Try small perturbations
                for _ in range(10):
                    new_points = points.copy()
                    # Random small move
                    new_points[vertex_idx] += rng.normal(0, step_size, 2)
                    
                    new_score = compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                        points = new_points.copy()
                        break
        
        return best_points

    def simulated_annealing(points, max_iter=5000, t_start=0.05, t_end=0.001):
        """
        Simulated annealing to escape local minima.
        """
        points = points.copy()
        rng = np.random.default_rng(456)
        
        best_score = compute_score(points)
        best_points = points.copy()
        current_score = best_score
        
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Temperature schedule
            t = t_start * (t_end / t_start) ** (iteration / max_iter)
            
            # Generate a random move
            new_points = points.copy()
            vertex_idx = rng.integers(0, n)
            new_points[vertex_idx] += rng.normal(0, 0.02, 2)
            
            new_score = compute_score(new_points)
            
            if new_score > current_score:
                current_score = new_score
                points = new_points.copy()
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
            else:
                # Accept with probability exp((new_score - current_score) / t)
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / t)
                    if rng.random() < prob:
                        current_score = new_score
                        points = new_points.copy()
        
        return best_points

    best_score = -1.0
    best_points = None

    # Phase 1: Generate multiple lattice-based initializations
    initializations = []
    
    # Triangular lattice variations
    for seed in range(5):
        initializations.append(('triangular', seed))
    
    # Hexagonal cluster variations
    for seed in range(5):
        initializations.append(('hexagonal', seed))
    
    # Regular polygon as fallback
    for seed in range(3):
        initializations.append(('polygon', seed))

    for method, seed in initializations:
        if time.time() - start_time > time_limit - 120:
            break

        if method == 'triangular':
            init_points = make_triangular_lattice(seed)
        elif method == 'hexagonal':
            init_points = make_hexagonal_cluster(seed)
        else:
            # Regular polygon
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            init_points = np.column_stack([np.cos(angles), np.sin(angles)])
            rng = np.random.default_rng(seed)
            init_points += rng.normal(0, 0.02, (n, 2))
        
        # Evaluate initial score
        init_score = compute_score(init_points)
        if init_score > best_score:
            best_score = init_score
            best_points = init_points.copy()
        
        # Refine with Delaunay-guided local search
        if time.time() - start_time < time_limit - 60:
            refined_points = delaunay_guided_local_search(init_points, max_iter=500, step_size=0.005)
            refined_score = compute_score(refined_points)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points.copy()

    # Phase 2: Simulated annealing on the best solution
    if best_points is not None and time.time() - start_time < time_limit - 30:
        annealed_points = simulated_annealing(best_points, max_iter=3000, t_start=0.03, t_end=0.0005)
        annealed_score = compute_score(annealed_points)
        if annealed_score > best_score:
            best_score = annealed_score
            best_points = annealed_points.copy()

    # Phase 3: Final polish with Nelder-Mead
    if best_points is not None and time.time() - start_time < time_limit - 10:
        def neg_objective(flat):
            points = flat.reshape(n, 2)
            score = compute_score(points)
            if score < 0:
                return 1.0
            return -score
        
        try:
            result = minimize(neg_objective, best_points.flatten(),
                             method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if result.x is not None:
                final_points = result.x.reshape(n, 2)
                final_score = compute_score(final_points)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
        except Exception:
            pass

    # Ensure valid output
    if best_points is None or not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points


# EVOLVE-BLOCK-END
