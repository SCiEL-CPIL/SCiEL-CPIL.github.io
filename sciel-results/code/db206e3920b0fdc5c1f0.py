# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _compute_score_from_points(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Assumes points form a convex polygon in order (or at least a valid hull).
    For the radial perturbation strategy, we assume the regular 13-gon hull is maintained 
    by small perturbations, so A_hull is constant (area of regular 13-gon with R=1).
    However, to be robust and correct for the evaluator, we should ideally compute A_hull 
    if the shape deviates significantly, but for the optimization loop, we can 
    pre-compute the hull area of the base polygon and assume it's roughly constant 
    for small perturbations. 
    BUT, the metric is A_min / A_hull. If we fix the hull to be the regular 13-gon, 
    A_hull is constant. 
    Let's compute A_min for the points and divide by the constant A_hull of the regular 13-gon.
    This is valid if the points remain inside the hull or on the boundary such that 
    the hull area doesn't change. 
    Wait, if points move radially, they might move outside the regular 13-gon if r > 1.
    If r < 1, the hull is still the regular 13-gon? No, if all r < 1, the hull is the 
    polygon formed by the points, which is smaller.
    
    Actually, the "regular 13-gon with radial perturbations" strategy usually implies 
    that the points are ON the boundary of the convex region. If we define the convex 
    region to be the regular 13-gon, then points with r_i <= 1 are inside. 
    The hull of the points will be a subset of the regular 13-gon.
    
    To maximize A_min / A_hull, we want A_min large and A_hull small.
    If we allow points to move radially, we can shrink the hull by moving points inward.
    However, moving points inward also reduces the triangle areas.
    
    The optimal solution for Heilbronn's problem is known to have points on the boundary 
    of the convex hull. So A_hull is determined by the outermost points.
    
    Let's stick to the strategy: Parametrize points as P_i = R_i * (cos(theta_i), sin(theta_i)) 
    where theta_i are angles of a regular 13-gon. 
    We optimize R_i (13 parameters).
    The convex hull of these points will be the polygon formed by them if they are in 
    convex position. For small perturbations around R=1, they will be in convex position 
    and the hull area will vary.
    
    To make the optimization fast, we can approximate A_hull. 
    Or, we can just compute the score using the exact formula but only for the 
    candidate points in the SA loop. 
    Computing ConvexHull is expensive. 
    However, for 13 points, it's not *that* expensive. 
    But we can do better: 
    If we assume the points are in convex position (which is likely for radial perturbations 
    of a regular polygon), the hull area can be computed via the shoelace formula on the 
    ordered points. 
    We must ensure the points are ordered by angle. 
    Since theta_i are fixed and distinct, the order is fixed. 
    We just need to ensure no self-intersection, which is unlikely for radial perturbations 
    of a convex polygon unless perturbations are huge.
    
    So, fast score:
    1. Compute areas of all 286 triangles.
    2. Find min area.
    3. Compute hull area using Shoelace formula on the 13 points (assuming convexity).
    4. Return min_area / hull_area.
    
    This avoids scipy.spatial.ConvexHull entirely in the inner loop.
    """
    n = points.shape[0]
    # Compute triangle areas
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 1e-10:
        return 0.0
        
    # Compute hull area using Shoelace formula
    # Points are in angular order
    x = points[:, 0]
    y = points[:, 1]
    # Shoelace: 0.5 * |sum(x_i y_{i+1} - x_{i+1} y_i)|
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    hull_area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
    
    if hull_area <= 1e-10:
        return 0.0
        
    return min_area / hull_area

def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    deadline = 600  # 600s limit, use 580s for safety
    
    # Base angles for regular 13-gon
    n = 13
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Initial radii: all 1.0
    best_radii = np.ones(n)
    best_score = _compute_score_from_points(np.column_stack([
        best_radii * np.cos(angles), 
        best_radii * np.sin(angles)
    ]))
    
    # Simulated Annealing parameters
    T = 0.1  # Initial temperature
    T_min = 1e-6
    alpha = 0.999  # Cooling rate
    max_iters = 50000
    
    current_radii = np.ones(n)
    current_points = np.column_stack([
        current_radii * np.cos(angles), 
        current_radii * np.sin(angles)
    ])
    current_score = best_score
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iters):
        if time.time() - start_time > deadline:
            break
            
        # Perturb radii
        # Randomly select a subset of radii to perturb to allow local adjustments
        # Or perturb all slightly. Let's perturb all with small noise.
        noise = rng.normal(0, 0.01, n)
        new_radii = current_radii + noise
        
        # Keep radii positive and bounded to avoid degenerate shapes
        new_radii = np.clip(new_radii, 0.5, 1.5)
        
        new_points = np.column_stack([
            new_radii * np.cos(angles), 
            new_radii * np.sin(angles)
        ])
        
        new_score = _compute_score_from_points(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_radii = new_radii
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_radii = current_radii.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_radii = new_radii
                    current_points = new_points
                    current_score = new_score
            else:
                # Should not happen if new_score > current_score is handled above, 
                # but just in case
                current_radii = new_radii
                current_points = new_points
                current_score = new_score
                
        # Cool
        T *= alpha
        if T < T_min:
            T = T_min
            # Optionally reheat if stuck? 
            # Let's just stop or restart with a different seed if we have time
            # For simplicity, just continue with low T
    
    # Final result
    final_points = np.column_stack([
        best_radii * np.cos(angles), 
        best_radii * np.sin(angles)
    ])
    
    # Sanity check: ensure finite and valid
    if not np.all(np.isfinite(final_points)):
        # Fallback to regular polygon
        final_points = np.column_stack([
            np.cos(angles), 
            np.sin(angles)
        ])
        
    return final_points

# EVOLVE-BLOCK-END
