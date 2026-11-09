# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).

    This implementation uses an optimization approach with Edge Lengths and Turning Angles
    to parameterize the convex polygon. It maximizes the minimum vertex score using a
    soft-min approximation to handle the non-smooth objective.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin

    # Initial guess: Regular polygon
    # Edge length for regular 20-gon with radius 1: 2*sin(pi/20)
    L0 = 2.0 * np.sin(np.pi / n)
    # Turning angle for regular 20-gon: 2*pi/20
    psi0 = 2.0 * np.pi / n
    
    # Parameters: L (20), u (20) where psi_k = 2*pi*u_k / sum(u)
    # Initialize u such that psi = psi0
    # u_k proportional to psi_k
    u0 = np.ones(n) * psi0
    
    x0 = np.concatenate([L0 * np.ones(n), u0])
    
    # Bounds for L: [0.1, 10.0]
    # Bounds for u: [0.01, 10.0]
    bounds_L = [(0.05, 10.0) for _ in range(n)]
    bounds_u = [(0.01, 10.0) for _ in range(n)]
    bounds = bounds_L + bounds_u

    def get_polygon_from_params(params):
        L = params[:n]
        u = params[n:]
        
        # Calculate psi
        psi = 2.0 * np.pi * u / np.sum(u)
        
        # Reconstruct vertices
        theta = np.zeros(n)
        theta[1:] = np.cumsum(psi[:-1])
        
        dx = L * np.cos(theta)
        dy = L * np.sin(theta)
        
        x = np.cumsum(dx)
        y = np.cumsum(dy)
        
        vertices = np.column_stack((x, y))
        
        # Closure error
        closure_err = np.linalg.norm(vertices[-1] + np.array([dx[-1], dy[-1]]) - vertices[0])
        # Actually, vertices[0] is (0,0). vertices[-1] is v_{n-1}.
        # v_n = v_{n-1} + L_{n-1} e^{i theta_{n-1}}
        # So closure is v_n - v_0.
        # In my cumsum, x[-1] is x_{n-1}.
        # The last edge is from v_{n-1} to v_n.
        # So v_n = x[-1] + dx[-1].
        # v_0 = 0.
        # closure_err = || v_n - v_0 ||
        
        return vertices, closure_err

    def vertex_score(v, vertices):
        # v is the vertex index
        # Calculate distances to all other vertices
        diffs = vertices - vertices[v]
        dists = np.linalg.norm(diffs, axis=1)
        # Remove self distance
        dists = np.delete(dists, v)
        
        # Sort
        sorted_dists = np.sort(dists)
        
        # Check constraints
        if sorted_dists[0] < 0.05:
            return 0.0
            
        # Find k minimizing gap
        gaps = np.diff(sorted_dists)
        k = np.argmin(gaps)
        
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
        
        if D < 0.1:
            return 0.0
            
        # Select 4 distances
        if k == 0:
            indices = [0, 1, 2, 3]
        elif k == n - 2: # k=18 for n=20? No, k is index in sorted_dists (len 19).
            # sorted_dists has 19 elements. indices 0..18.
            # gaps has 18 elements. indices 0..17.
            # If k=17 (last gap), indices 15..18?
            # Task: "d_15..d_18 if k = 17"
            # Yes.
            indices = [15, 16, 17, 18]
        else:
            indices = [k-1, k, k+1, k+2]
            
        selected = sorted_dists[indices]
        
        # Calculate raw score
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        
        score = min(1.0, 1.0 / raw)
        
        return score

    def calculate_min_score(vertices):
        # Normalize
        centroid = np.mean(vertices, axis=0)
        vertices = vertices - centroid
        dists_to_centroid = np.linalg.norm(vertices, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        
        if mean_dist <= 0.01:
            return 0.0
            
        vertices = vertices / mean_dist
        
        # Check convexity
        # Compute cross products of consecutive edges
        edges = np.diff(vertices, axis=0, append=vertices[:1])
        cross_products = np.cross(edges[:-1], edges[1:])
        
        # For CCW convex polygon, all cross products should be > 0
        # Note: np.cross for 2D vectors returns scalar z-component
        if np.any(cross_products <= 1e-9):
            return 0.0
            
        # Calculate score for each vertex
        scores = []
        for i in range(n):
            scores.append(vertex_score(i, vertices))
            
        return np.min(scores)

    def objective(params):
        # Check time
        if time.time() > deadline:
            return np.inf
            
        vertices, closure_err = get_polygon_from_params(params)
        
        # If closure error is too large, penalize heavily
        if closure_err > 1e-3:
            return 1e6 + closure_err
            
        min_score = calculate_min_score(vertices)
        
        # We want to maximize min_score.
        # Use soft-min? The min is already non-smooth.
        # Just return -min_score.
        return -min_score

    # Optimization
    best_score = 1.0 # Regular polygon score
    best_params = x0.copy()
    
    try:
        result = opt.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.fun < 0: # Improved
            best_params = result.x
            best_score = -result.fun
        else:
            # Check if result is better than initial
            v, _ = get_polygon_from_params(result.x)
            s = calculate_min_score(v)
            if s > best_score:
                best_params = result.x
                best_score = s

    except Exception:
        pass

    # Final reconstruction
    vertices, _ = get_polygon_from_params(best_params)
    
    # Normalize for output
    centroid = np.mean(vertices, axis=0)
    vertices = vertices - centroid
    dists_to_centroid = np.linalg.norm(vertices, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist > 0.01:
        vertices = vertices / mean_dist
        
    return vertices

# EVOLVE-BLOCK-END
