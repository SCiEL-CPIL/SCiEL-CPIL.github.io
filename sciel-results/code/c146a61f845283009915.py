# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon using support function representation.
    The support function h(theta) defines the polygon as intersection of half-planes,
    guaranteeing convexity by construction. We optimize Fourier coefficients of h.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin before 600s limit

    # Support function approach:
    # h(theta) = 1 + sum_{k=1}^{K} (a_k cos(k*theta) + b_k sin(k*theta))
    # The polygon is the intersection of half-planes: x*cos(theta_i) + y*sin(theta_i) <= h(theta_i)
    # Vertices are at the intersections of consecutive half-planes.
    
    K = 4  # Number of Fourier harmonics for support function
    n_params = 2 * K
    
    # Sample angles for support function (use more samples for better approximation)
    n_samples = 200
    sample_angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    
    # The 20 vertex angles (evenly spaced)
    vertex_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def support_function(theta, params):
        """Evaluate support function h(theta) given Fourier parameters."""
        h = np.ones_like(theta)
        for k in range(1, K + 1):
            a = params[2 * (k - 1)]
            b = params[2 * (k - 1) + 1]
            h += a * np.cos(k * theta) + b * np.sin(k * theta)
        return h
    
    def params_to_polygon(params):
        """
        Convert support function parameters to polygon vertices.
        The polygon is defined by half-planes: x*cos(phi_i) + y*sin(phi_i) <= h(phi_i)
        where phi_i are the outward normal directions.
        Vertices are intersections of consecutive half-planes.
        """
        # Use the 20 vertex angles as normal directions
        h_values = support_function(vertex_angles, params)
        
        # Ensure h is positive (strictly convex requires h > 0)
        if np.any(h_values <= 0.05):
            return None
        
        # For each vertex i, it's the intersection of half-planes i-1 and i
        # Half-plane j: x*cos(phi_j) + y*sin(phi_j) = h(phi_j)
        # Solve the 2x2 system for each vertex
        
        vertices = np.zeros((n, 2))
        for i in range(n):
            # Vertex i is intersection of half-plane (i-1) and half-plane i
            phi_prev = vertex_angles[(i - 1) % n]
            phi_curr = vertex_angles[i]
            h_prev = h_values[(i - 1) % n]
            h_curr = h_values[i]
            
            # System:
            # cos(phi_prev) * x + sin(phi_prev) * y = h_prev
            # cos(phi_curr) * x + sin(phi_curr) * y = h_curr
            
            A = np.array([
                [np.cos(phi_prev), np.sin(phi_prev)],
                [np.cos(phi_curr), np.sin(phi_curr)]
            ])
            b = np.array([h_prev, h_curr])
            
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if abs(det) < 1e-10:
                return None
            
            try:
                vertices[i] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return None
        
        return vertices

    def is_strictly_convex(polygon, tol=1e-10):
        """Check if polygon is strictly convex (CCW, all cross products positive)."""
        n_pts = len(polygon)
        for i in range(n_pts):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % n_pts]
            p2 = polygon[(i + 2) % n_pts]
            cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
            if cross <= tol:
                return False
        return True

    def vertex_score(v_idx, polygon):
        """Compute the score for a single vertex."""
        v = polygon[v_idx]
        # Distances to all other vertices
        dists = np.sqrt(np.sum((polygon - v) ** 2, axis=1))
        # Remove self-distance
        mask = np.ones(n, dtype=bool)
        mask[v_idx] = False
        dists = dists[mask]
        dists_sorted = np.sort(dists)

        # Check minimum distance
        if dists_sorted[0] < 0.05:
            return 0.0

        # Find k: first index minimizing d_{k+1} - d_k
        diffs = np.diff(dists_sorted)
        k = int(np.argmin(diffs))
        D = (dists_sorted[k] + dists_sorted[k + 1]) / 2.0
        if D < 0.1:
            return 0.0

        # Select four distances
        if k == 0:
            selected = dists_sorted[0:4]
        elif k == 17:
            selected = dists_sorted[15:19]
        else:
            selected = dists_sorted[k - 1:k + 3]

        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        return min(1.0, 1.0 / raw)

    def min_vertex_score(polygon):
        """Minimum vertex score over all 20 vertices."""
        scores = [vertex_score(i, polygon) for i in range(n)]
        return min(scores)

    def objective(params):
        """Objective to minimize: negative of min vertex score."""
        polygon = params_to_polygon(params)
        if polygon is None:
            return 10.0 + np.sum(params ** 2)
        if not is_strictly_convex(polygon):
            return 1.0 + np.sum(params ** 2)
        score = min_vertex_score(polygon)
        return -score

    # Regular polygon as fallback (h = 1 gives regular polygon)
    angles = vertex_angles
    regular = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_score = min_vertex_score(regular)
    best_polygon = regular.copy()
    
    # Initial best params (zero perturbation = regular polygon)
    best_params = np.zeros(n_params)

    # Multi-start Nelder-Mead optimization
    np.random.seed(42)
    n_starts = 30
    bounds_range = 0.15  # Moderate perturbation range
    
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            start_params = best_params.copy()
        elif i == 1:
            start_params = np.zeros(n_params)
        else:
            start_params = np.random.uniform(-bounds_range, bounds_range, n_params)
        
        try:
            res = minimize(objective, start_params, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if res.fun < 0:  # Found valid convex polygon with positive score
                candidate_polygon = params_to_polygon(res.x)
                if candidate_polygon is not None and is_strictly_convex(candidate_polygon):
                    candidate_score = min_vertex_score(candidate_polygon)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_polygon = candidate_polygon
                        best_params = res.x
        except Exception:
            pass

    # High-precision polish on the best candidate
    if time.time() < deadline - 15 and best_params is not None:
        try:
            polish_res = minimize(objective, best_params, method='Nelder-Mead',
                                  options={'maxiter': 30000, 'xatol': 1e-15, 'fatol': 1e-15})
            if polish_res.fun < 0:
                candidate_polygon = params_to_polygon(polish_res.x)
                if candidate_polygon is not None and is_strictly_convex(candidate_polygon):
                    candidate_score = min_vertex_score(candidate_polygon)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_polygon = candidate_polygon
                        best_params = polish_res.x
        except Exception:
            pass

    return best_polygon

# EVOLVE-BLOCK-END
