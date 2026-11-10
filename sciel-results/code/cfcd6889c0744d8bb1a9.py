import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def compute_scores(polygon):
    n = 20
    scores = np.zeros(n)
    for i in range(n):
        v = polygon[i]
        dists = np.linalg.norm(polygon - v, axis=1)
        dists[i] = np.inf
        dists = np.sort(dists)
        
        # Check minimum distance
        if dists[0] < 0.05:
            scores[i] = 0.0
            continue
            
        diffs = np.diff(dists)
        # Find the largest gap
        k = int(np.argmax(diffs))
        
        # The gap is between dists[k] and dists[k+1]
        # D is the midpoint of the gap
        D = (dists[k] + dists[k+1]) / 2.0
        
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        # Select 4 distances
        if k == 0:
            d_sel = dists[0:4]
        elif k == 17:
            d_sel = dists[15:19]
        else:
            d_sel = dists[k-1:k+3]
            
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
    return scores

def _normalize(polygon):
    centroid = np.mean(polygon, axis=0)
    polygon_norm = polygon - centroid
    mean_dist = np.mean(np.linalg.norm(polygon_norm, axis=1))
    if mean_dist <= 0.01:
        return None
    return polygon_norm / mean_dist

def _is_strictly_convex(polygon):
    # Check cross products of consecutive edges
    n = len(polygon)
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i+1)%n]
        p2 = polygon[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if cross <= 1e-8:
            return False
    # Check for repeated points
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(polygon[i] - polygon[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-4:
        return False
    return True

def _params_to_polygon(params):
    """
    params: [e_0..e_19, a_0..a_18]
    e: edge lengths
    a: exterior angles (turning angles)
    a_19 is determined by closure: 2*pi - sum(a_0..a_18)
    """
    n = 20
    edges = params[:n]
    angles = params[n:]
    
    # Calculate the last angle
    sum_angles = np.sum(angles)
    a_last = 2.0 * np.pi - sum_angles
    
    # If a_last is invalid (too small or negative), return a regular polygon or penalize heavily
    if a_last < 1e-6:
        # Return a degenerate or regular polygon to avoid crash, objective will penalize
        reg_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        reg_edges = np.ones(n)
        # Just return a regular polygon for now, optimizer should avoid this
        return _edges_angles_to_polygon(reg_edges, reg_angles)
        
    all_angles = np.append(angles, a_last)
    
    return _edges_angles_to_polygon(edges, all_angles)

def _edges_angles_to_polygon(edges, angles):
    """
    Construct polygon from edge lengths and exterior angles.
    Starts at origin, first edge along x-axis.
    """
    n = len(edges)
    vertices = np.zeros((n, 2))
    vertices[0] = [0, 0]
    
    current_angle = 0.0
    for i in range(n):
        # Move along edge i
        vertices[i+1] if i+1 < n else vertices[0] # This logic is flawed for closure, let's just build path
        
    # Correct construction:
    vertices = np.zeros((n+1, 2))
    current_angle = 0.0
    for i in range(n):
        dx = edges[i] * np.cos(current_angle)
        dy = edges[i] * np.sin(current_angle)
        vertices[i+1] = vertices[i] + np.array([dx, dy])
        current_angle += angles[i]
        
    # The last vertex should match the first (0,0) if angles sum to 2*pi
    # But we return the n vertices (excluding the duplicate start/end)
    return vertices[:n]

def _objective(params):
    polygon = _params_to_polygon(params)
    polygon_norm = _normalize(polygon)
    
    if polygon_norm is None:
        return 1000.0
        
    # Check strict convexity
    if not _is_strictly_convex(polygon_norm):
        return 500.0
        
    scores = compute_scores(polygon_norm)
    min_score = np.min(scores)
    
    # If any score is 0, penalize heavily
    if min_score == 0:
        return 100.0 - np.sum(scores) # Try to maximize sum of non-zero scores to find valid region
        
    return -min_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = time.time() + 600.0
    
    n = 20
    # Initial guess: Regular 20-gon
    # Edge length for regular 20-gon inscribed in unit circle is 2*sin(pi/20)
    # But we are free to scale. Let's start with edge length 1.
    e_reg = 2 * np.sin(np.pi / n)
    a_reg = 2 * np.pi / n
    
    e_init = np.full(n, e_reg)
    a_init = np.full(n-1, a_reg)
    
    best_params = np.concatenate([e_init, a_init])
    best_score = -1e9
    best_polygon = None
    
    # Regular polygon normalized
    reg_vertices = _edges_angles_to_polygon(e_init, np.full(n, a_reg))
    reg_norm = _normalize(reg_vertices)
    if reg_norm is not None and _is_strictly_convex(reg_norm):
        s = np.min(compute_scores(reg_norm))
        best_score = s
        best_polygon = reg_norm
        best_params = np.concatenate([e_init, a_init]).copy()
        
    bounds = [(1e-4, 5.0)] * n + [(1e-4, 0.5)] * (n-1)
    
    # Phase 1: Differential Evolution
    try:
        if time.time() < deadline - 120:
            de_result = differential_evolution(
                _objective, bounds,
                seed=42,
                maxiter=300,
                tol=1e-5,
                popsize=20,
                mutation=(0.4, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            
            if de_result.fun < 100:
                pn = _params_to_polygon(de_result.x)
                pn_norm = _normalize(pn)
                if pn_norm is not None and _is_strictly_convex(pn_norm):
                    s = np.min(compute_scores(pn_norm))
                    if s > best_score:
                        best_score = s
                        best_polygon = pn_norm
                        best_params = de_result.x.copy()
    except Exception:
        pass
    
    # Phase 2: Local Polish
    try:
        if time.time() < deadline - 30 and best_score > 0:
            polish_result = minimize(
                _objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if polish_result.fun < 100:
                pn = _params_to_polygon(polish_result.x)
                pn_norm = _normalize(pn)
                if pn_norm is not None and _is_strictly_convex(pn_norm):
                    s = np.min(compute_scores(pn_norm))
                    if s > best_score:
                        best_score = s
                        best_polygon = pn_norm
                        best_params = polish_result.x.copy()
    except Exception:
        pass
        
    # Phase 3: High Precision Polish
    try:
        if time.time() < deadline - 10 and best_score > 0.5:
            final_result = minimize(
                _objective, best_params,
                method='Powell',
                options={'maxiter': 20000, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            
            if final_result.fun < 100:
                pn = _params_to_polygon(final_result.x)
                pn_norm = _normalize(pn)
                if pn_norm is not None and _is_strictly_convex(pn_norm):
                    s = np.min(compute_scores(pn_norm))
                    if s > best_score:
                        best_score = s
                        best_polygon = pn_norm
    except Exception:
        pass
    
    if best_polygon is not None:
        return best_polygon
        
    # Fallback
    return reg_norm if reg_norm is not None else np.zeros((20,2))
