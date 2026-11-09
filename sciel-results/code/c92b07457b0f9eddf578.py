import numpy as np
import time
import sys
from scipy.spatial import ConvexHull
from itertools import combinations

def _all_triangle_indices(n):
    """Precompute all C(n,3) triangle index triples."""
    return list(combinations(range(n), 3))

def _triangle_areas_vectorized(pts):
    """Compute areas of all C(n,3) triangles efficiently using vectorization."""
    n = len(pts)
    tris = _all_triangle_indices(n)
    idx = np.array(tris)  # shape (286, 3)
    p0 = pts[idx[:, 0]]  # (286, 2)
    p1 = pts[idx[:, 1]]
    p2 = pts[idx[:, 2]]
    # Area = 0.5 * |cross(p1-p0, p2-p0)|
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) / 2.0
    return areas, idx

def _score(pts):
    """Compute A_min / A_hull score."""
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas, _ = _triangle_areas_vectorized(pts)
        a_min = areas.min()
        if a_min < 1e-15:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _soft_min_area(pts, temperature):
    """
    Smooth approximation of min triangle area using log-sum-exp.
    Returns (soft_min_value, gradient) where gradient is d(soft_min)/d(pts).
    """
    n = len(pts)
    areas, idx = _triangle_areas_vectorized(pts)
    # soft min = -T * log(sum(exp(-area/T)))
    # For numerical stability, subtract max
    neg_areas = -areas
    max_neg = neg_areas.max()
    exp_terms = np.exp(neg_areas - max_neg)
    log_sum = max_neg + np.log(exp_terms.sum())
    soft_min = -log_sum
    
    # Gradient: d(soft_min)/d(pts[i]) = sum_k w_k * d(soft_min)/d(area_k) * d(area_k)/d(pts[i])
    # where w_k = exp(-area_k/T) / sum(exp(-area_j/T))
    # and d(soft_min)/d(area_k) = -w_k (since soft_min = -T*log(sum(exp(-area/T))))
    # Actually: soft_min = -T * log(sum_j exp(-area_j/T))
    # d(soft_min)/d(area_k) = -T * (1/sum) * exp(-area_k/T) * (-1/T) = exp(-area_k/T)/sum = w_k
    # So d(soft_min)/d(pts[i]) = sum_k w_k * d(area_k)/d(pts[i])
    
    weights = exp_terms / exp_terms.sum()  # shape (286,)
    
    grad = np.zeros_like(pts)
    # For each triangle k with vertices (i,j,l), area_k = 0.5*|cross|
    # We need d(area_k)/d(pts[i]) for each vertex
    # area_k = 0.5 * |x_i(y_j - y_l) + x_j(y_l - y_i) + x_l(y_i - y_j)|
    # Let signed = x_i(y_j - y_l) + x_j(y_l - y_i) + x_l(y_i - y_j)
    # area = |signed|/2
    # d(area)/d(x_i) = sign(signed) * (y_j - y_l) / 2
    # d(area)/d(y_i) = sign(signed) * (x_l - x_j) / 2
    # d(area)/d(x_j) = sign(signed) * (y_l - y_i) / 2
    # d(area)/d(y_j) = sign(signed) * (x_i - x_l) / 2
    # d(area)/d(x_l) = sign(signed) * (y_i - y_j) / 2
    # d(area)/d(y_l) = sign(signed) * (x_j - x_i) / 2
    
    p0 = pts[idx[:, 0]]
    p1 = pts[idx[:, 1]]
    p2 = pts[idx[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    signed = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    signs = np.sign(signed)
    signs[signs == 0] = 1.0
    
    for k in range(len(idx)):
        w = weights[k]
        if w < 1e-15:
            continue
        i, j, l = idx[k]
        s = signs[k]
        xi, yi = pts[i]
        xj, yj = pts[j]
        xl, yl = pts[l]
        
        # d(area)/d(xi) = s * (yj - yl) / 2
        grad[i, 0] += w * s * (yj - yl) / 2.0
        # d(area)/d(yi) = s * (xl - xj) / 2
        grad[i, 1] += w * s * (xl - xj) / 2.0
        # d(area)/d(xj) = s * (yl - yi) / 2
        grad[j, 0] += w * s * (yl - yi) / 2.0
        # d(area)/d(yj) = s * (xi - xl) / 2
        grad[j, 1] += w * s * (xi - xl) / 2.0
        # d(area)/d(xl) = s * (yi - yj) / 2
        grad[l, 0] += w * s * (yi - yj) / 2.0
        # d(area)/d(yl) = s * (xj - xi) / 2
        grad[l, 1] += w * s * (xj - xi) / 2.0
    
    return soft_min, grad

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Start from regular n-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float64)
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Phase 1: Soft-min gradient ascent with annealing
    # Temperature controls how many triangles contribute to the gradient
    T_start = 0.05
    T_end = 0.001
    total_steps = 8000
    
    # Precompute for speed
    tris_list = _all_triangle_indices(n)
    
    for step in range(total_steps):
        if time.time() > deadline - 30.0:
            break
        
        # Anneal temperature
        progress = step / total_steps
        T = T_start * (T_end / T_start) ** progress
        
        soft_min_val, grad = _soft_min_area(pts, T)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            continue
        
        # Adaptive learning rate
        lr = 0.002 * (1.0 - progress * 0.5)
        
        # Gradient ascent step
        candidate = pts + lr * grad / grad_norm
        
        cand_score = _score(candidate)
        if cand_score > best_score + 1e-14:
            pts = candidate
            best_pts = pts.copy()
            best_score = cand_score
        else:
            # Try smaller steps
            for factor in [0.5, 0.25, 0.1]:
                candidate = pts + lr * factor * grad / grad_norm
                cand_score = _score(candidate)
                if cand_score > best_score + 1e-14:
                    pts = candidate
                    best_pts = pts.copy()
                    best_score = cand_score
                    break
            else:
                # Simulated annealing: accept with probability exp(-delta/T_anneal)
                delta = best_score - cand_score
                if delta > 0:
                    T_anneal = 0.01 * (1.0 - progress) + 0.001
                    if np.random.random() < np.exp(-delta / T_anneal):
                        pts = candidate
    
    # Phase 2: Local refinement with exact min subgradient
    # Fine-tune using the hard minimum
    for restart in range(50):
        if time.time() > deadline - 60.0:
            break
        
        # Perturb best solution
        rng = np.random.default_rng(restart)
        noise = rng.normal(0, 0.05, size=(n, 2))
        pts = best_pts + noise
        curr_score = _score(pts)
        
        if curr_score > best_score:
            best_pts = pts.copy()
            best_score = curr_score
        
        # Gradient ascent on hard min
        lr = 0.005
        for step in range(2000):
            if time.time() > deadline - 60.0:
                break
            
            # Find minimum area triangle
            areas, idx = _triangle_areas_vectorized(pts)
            min_idx = np.argmin(areas)
            min_area = areas[min_idx]
            if min_area < 1e-15:
                break
            
            i, j, k = idx[min_idx]
            xi, yi = pts[i]
            xj, yj = pts[j]
            xl, yl = pts[k]
            signed = (xj - xi) * (yl - yi) - (yj - yi) * (xl - xi)
            s = np.sign(signed)
            if s == 0:
                s = 1.0
            
            # Subgradient of min area
            grad = np.zeros_like(pts)
            grad[i, 0] += s * (yj - yl) / 2.0
            grad[i, 1] += s * (xl - xj) / 2.0
            grad[j, 0] += s * (yl - yi) / 2.0
            grad[j, 1] += s * (xi - xl) / 2.0
            grad[k, 0] += s * (yi - yj) / 2.0
            grad[k, 1] += s * (xj - xi) / 2.0
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            candidate = pts + lr * grad / grad_norm
            cand_score = _score(candidate)
            if cand_score > best_score + 1e-14:
                pts = candidate
                best_pts = pts.copy()
                best_score = cand_score
                lr = min(lr * 1.05, 0.01)
            else:
                lr *= 0.9
                if lr < 1e-7:
                    break
    
    # Final validation
    final_score = _score(best_pts)
    if final_score < best_score - 1e-10:
        # Something went wrong, but best_pts should still be valid
        pass
    
    return best_pts
