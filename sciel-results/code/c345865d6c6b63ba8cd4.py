# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-10:
        return 0.0
        
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
    
    if min_area <= 0:
        return 0.0
        
    return min_area / A_hull

def _generate_initializations():
    """
    Generate diverse initial configurations for the optimizer.
    """
    inits = []
    
    # 1. Regular 13-gon (points on a circle)
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    pts = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    inits.append(pts)
    
    # 2. Hexagonal packing approximation (3-4-4-2 rows)
    pts = []
    # Row 0: 3 pts
    for x in [0.2, 0.5, 0.8]: pts.append([x, 0.8])
    # Row 1: 4 pts
    for x in [0.1, 0.36, 0.63, 0.9]: pts.append([x, 0.53])
    # Row 2: 4 pts
    for x in [0.1, 0.36, 0.63, 0.9]: pts.append([x, 0.26])
    # Row 3: 2 pts
    for x in [0.3, 0.7]: pts.append([x, 0.0])
    inits.append(np.array(pts))
    
    # 3. Concentric rings (1 center, 6 inner, 6 outer)
    pts = [[0.5, 0.5]]
    r1 = 0.2
    for i in range(6):
        a = np.pi / 3 * i
        pts.append([0.5 + r1 * np.cos(a), 0.5 + r1 * np.sin(a)])
    r2 = 0.4
    for i in range(6):
        a = np.pi / 3 * i + np.pi / 6
        pts.append([0.5 + r2 * np.cos(a), 0.5 + r2 * np.sin(a)])
    inits.append(np.array(pts))
    
    # 4. Random points (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.random((13, 2)) * 0.8 + 0.1
        inits.append(pts)
        
    return inits

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area using deterministic gradient-based ascent
    on the min-area bottleneck.
    """
    start_time = time.time()
    deadline = 630  # seconds, with a safety margin
    
    inits = _generate_initializations()
    
    best_points = None
    best_score = -np.inf
    
    # Parameters for gradient ascent
    lr_start = 0.05
    lr_decay = 0.9995
    n_steps_max = 20000
    
    for i, init_pts in enumerate(inits):
        if time.time() - start_time > deadline - 10:
            break
        
        current_points = init_pts.copy()
        current_score = _compute_score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        lr = lr_start
        
        # Precompute all triangle indices for vectorized operations
        n = 13
        idx = np.arange(n)
        triples = list(itertools.combinations(idx, 3))
        triples_arr = np.array(triples)
        m = len(triples_arr)
        
        p0_idx = triples_arr[:, 0]
        p1_idx = triples_arr[:, 1]
        p2_idx = triples_arr[:, 2]
        
        for step in range(n_steps_max):
            if time.time() - start_time > deadline - 5:
                break
            
            # 1. Compute areas for all triangles
            p0 = current_points[p0_idx]
            p1 = current_points[p1_idx]
            p2 = current_points[p2_idx]
            
            v1 = p1 - p0
            v2 = p2 - p0
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            
            # 2. Identify bottleneck triangles (those with area close to min)
            min_area = np.min(areas)
            if min_area <= 1e-10:
                # If area is too small, add noise to escape
                current_points += np.random.normal(0, 0.01, current_points.shape)
                current_points = np.clip(current_points, 0.01, 0.99)
                continue
                
            # Select triangles with area < min_area + epsilon
            epsilon = 0.1 * min_area
            bottleneck_mask = areas < (min_area + epsilon)
            
            if not np.any(bottleneck_mask):
                bottleneck_mask = np.zeros(m, dtype=bool)
                bottleneck_mask[np.argmin(areas)] = True
            
            # 3. Compute gradient of min area w.r.t points
            # For a triangle (i, j, k), area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
            # Let A = cross. dA/dp_i = -cross(p_j, p_k) / 2 ... actually easier to use vector calculus.
            # Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
            # Let's approximate the gradient for the bottleneck triangles.
            
            grad = np.zeros_like(current_points)
            
            # We want to increase the area of bottleneck triangles.
            # For each bottleneck triangle, we push points apart.
            # Specifically, for triangle (i,j,k), the area increases if we move i, j, k apart.
            # A simple heuristic: move each point in the direction away from the centroid of the triangle.
            # Or better: use the analytical gradient of the cross product magnitude.
            
            # Let's use the analytical gradient of the area for the active constraints.
            # Area = 0.5 * |det([p_j - p_i, p_k - p_i])|
            # Let D = det. dD/dp_i = - (p_j x p_k) ... wait.
            # D = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
            # dD/dx_i = -(y_k - y_i) - (y_j - y_i) * 0 ... 
            # dD/dx_i = -(y_k - y_i) - (y_j - y_i) = y_i - y_k - y_j + y_i = 2y_i - y_j - y_k
            # dD/dy_i = (x_j - x_i) - (x_k - x_i) = x_j - x_k
            # dD/dx_j = (y_k - y_i)
            # dD/dy_j = -(x_k - x_i)
            # dD/dx_k = -(y_j - y_i)
            # dD/dy_k = (x_j - x_i)
            
            # We want to maximize Area = 0.5 * |D|.
            # So we move in the direction of sign(D) * dD/dp.
            
            b_idx = np.where(bottleneck_mask)[0]
            for b in b_idx:
                i = p0_idx[b]
                j = p1_idx[b]
                k = p2_idx[b]
                
                pi = current_points[i]
                pj = current_points[j]
                pk = current_points[k]
                
                vi = pj - pi
                vk = pk - pi
                
                D = vi[0] * vk[1] - vi[1] * vk[0]
                
                if D == 0: continue
                
                sign_D = 1.0 if D > 0 else -1.0
                
                # Gradients for i
                grad[i, 0] += sign_D * (2 * pi[0] - pj[0] - pk[0])
                grad[i, 1] += sign_D * (pj[1] - pk[1])
                
                # Gradients for j
                grad[j, 0] += sign_D * (vk[1]) # (y_k - y_i)
                grad[j, 1] += sign_D * (-vk[0]) # -(x_k - x_i)
                
                # Gradients for k
                grad[k, 0] += sign_D * (-vi[1]) # -(y_j - y_i)
                grad[k, 1] += sign_D * (vi[0]) # (x_j - x_i)
            
            # 4. Compute gradient for hull area (we want to minimize it, so subtract)
            # Approximation: pull points towards the centroid to reduce hull area.
            # This is a simple heuristic that works well in conjunction with the area maximization.
            centroid = np.mean(current_points, axis=0)
            hull_grad = current_points - centroid
            # Scale hull gradient to be comparable to area gradient
            # The area gradient is roughly O(1) for unit points. Hull gradient is O(0.1-0.5).
            # We want to balance them.
            hull_weight = 0.1 
            total_grad = grad - hull_weight * hull_grad
            
            # Normalize gradient to avoid instability
            norm = np.linalg.norm(total_grad)
            if norm > 1e-10:
                step_vec = lr * total_grad / norm
            else:
                step_vec = np.zeros_like(current_points)
                
            new_points = current_points + step_vec
            new_points = np.clip(new_points, 0.01, 0.99)
            
            new_score = _compute_score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # If score drops, reduce learning rate and try a smaller step or random jitter
                lr *= 0.9
                # Add small random noise to escape local minima if stuck
                if step % 100 == 0:
                    current_points += np.random.normal(0, 0.005, current_points.shape)
                    current_points = np.clip(current_points, 0.01, 0.99)
                    current_score = _compute_score(current_points)
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
            
            lr *= lr_decay
            if lr < 1e-5:
                break

    # Fallback
    if best_points is None:
        best_points = inits[0]
        
    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = np.array([
            [0.2, 0.8], [0.5, 0.8], [0.8, 0.8],
            [0.1, 0.53], [0.36, 0.53], [0.63, 0.53], [0.9, 0.53],
            [0.1, 0.26], [0.36, 0.26], [0.63, 0.26], [0.9, 0.26],
            [0.3, 0.0], [0.7, 0.0]
        ])
        
    return best_points

# EVOLVE-BLOCK-END
