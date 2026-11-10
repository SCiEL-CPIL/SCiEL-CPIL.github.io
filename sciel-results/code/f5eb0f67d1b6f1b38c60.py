# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon using support function representation.
    The support function h(theta) defines the polygon as intersection of half-planes,
    guaranteeing convexity by construction. We optimize Fourier coefficients of h.
    Uses a two-phase approach: smooth soft-min objective for basin finding, then
    exact min-objective for high-precision polish.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s limit

    # Support function approach:
    # h(theta) = 1 + sum_{k=1}^{K} (a_k cos(k*theta) + b_k sin(k*theta))
    # The polygon is the intersection of half-planes: x*cos(theta_i) + y*sin(theta_i) <= h(theta_i)
    # Vertices are at the intersections of consecutive half-planes.
    
    K = 4  # Number of Fourier harmonics for support function
    n_params = 2 * K
    
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

    def all_vertex_scores(polygon):
        """All vertex scores over all 20 vertices."""
        return np.array([vertex_score(i, polygon) for i in range(n)])

    def soft_min(scores, temperature=0.05):
        """
        Smooth approximation of min(scores) using log-sum-exp.
        soft_min(s) = -T * log(sum(exp(-s_i / T)))
        As T -> 0, this approaches min(s).
        """
        if len(scores) == 0:
            return 0.0
        scores = np.array(scores)
        
        # Standard soft-min with numerical stability:
        # soft_min = -T * log(sum_i exp(-s_i / T))
        # Let m = max(s_i), then:
        # sum_i exp(-s_i/T) = exp(-m/T) * sum_i exp(-(s_i-m)/T)
        # log(sum) = -m/T + log(sum_i exp(-(s_i-m)/T))
        # soft_min = -T * (-m/T + log(sum_i exp(-(s_i-m)/T)))
        #          = m - T * log(sum_i exp(-(s_i-m)/T))
        
        m = np.max(scores)
        shifted = scores - m
        exp_shifted = np.exp(-shifted / temperature)
        log_sum_exp = np.log(np.sum(exp_shifted))
        return m - temperature * log_sum_exp

    def objective(params):
        """Objective to minimize: negative of min vertex score."""
        polygon = params_to_polygon(params)
        if polygon is None:
            return 10.0 + np.sum(params ** 2)
        if not is_strictly_convex(polygon):
            return 1.0 + np.sum(params ** 2)
        score = min_vertex_score(polygon)
        return -score

    def objective_smooth(params, temperature=0.05):
        """Objective with smooth min to help gradient-free optimizers navigate the landscape."""
        polygon = params_to_polygon(params)
        if polygon is None:
            return 10.0 + np.sum(params ** 2)
        if not is_strictly_convex(polygon):
            return 1.0 + np.sum(params ** 2)
        
        # Compute individual vertex scores
        scores = all_vertex_scores(polygon)
        
        # If any score is 0, the min is 0
        if np.any(scores < 1e-6):
            return 0.0
        
        # Use soft-min to get a smooth objective
        soft_score = soft_min(scores, temperature)
        return -soft_score

    # Regular polygon as fallback (h = 1 gives regular polygon)
    angles = vertex_angles
    regular = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_score = min_vertex_score(regular)
    best_polygon = regular.copy()
    
    # Initial best params (zero perturbation = regular polygon)
    best_params = np.zeros(n_params)

    # Multi-start optimization
    np.random.seed(42)
    
    # Phase 1: Coarse search with smooth objective to find good basins
    n_starts_smooth = 25
    bounds_range = 0.15
    
    for i in range(n_starts_smooth):
        if time.time() > deadline - 150:
            break
        
        if i == 0:
            start_params = best_params.copy()
        elif i == 1:
            start_params = np.zeros(n_params)
        else:
            start_params = np.random.uniform(-bounds_range, bounds_range, n_params)
        
        try:
            res = minimize(objective_smooth, start_params, method='Nelder-Mead',
                          options={'maxiter': 6000, 'xatol': 1e-10, 'fatol': 1e-10})
            
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

    # Phase 2: Fine search with exact objective from best candidates
    n_starts_fine = 12
    
    for i in range(n_starts_fine):
        if time.time() > deadline - 60:
            break
        
        if i == 0:
            start_params = best_params.copy()
        else:
            # Perturb the best params
            start_params = best_params + np.random.normal(0, 0.02, n_params)
        
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

    # Phase 3: High-precision polish on the best candidate
    if time.time() < deadline - 30 and best_params is not None:
        try:
            polish_res = minimize(objective, best_params, method='Nelder-Mead',
                                  options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-15})
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

    # Phase 4: Very tight polish if time remains
    if time.time() < deadline - 10 and best_params is not None:
        try:
            ultra_polish = minimize(objective, best_params, method='Nelder-Mead',
                                    options={'maxiter': 100000, 'xatol': 1e-16, 'fatol': 1e-16})
            if ultra_polish.fun < 0:
                candidate_polygon = params_to_polygon(ultra_polish.x)
                if candidate_polygon is not None and is_strictly_convex(candidate_polygon):
                    candidate_score = min_vertex_score(candidate_polygon)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_polygon = candidate_polygon
                        best_params = ultra_polish.x
        except Exception:
            pass

    # Final validation and return
    if not is_strictly_convex(best_polygon):
        # Fallback to regular polygon
        angles = vertex_angles
        best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    return best_polygon

# EVOLVE-BLOCK-END
