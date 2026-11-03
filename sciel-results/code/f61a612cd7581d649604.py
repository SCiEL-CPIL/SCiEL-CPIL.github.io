"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples of points."""
    n = len(pts)
    # Precompute all pairs of points
    min_area = np.inf
    # For efficiency, we can use the fact that area = 0.5 * |cross product|
    # We iterate over all triples
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pij = pts[j] - pi
            for k in range(j + 1, n):
                pik = pts[k] - pi
                cross = pij[0] * pik[1] - pij[1] * pik[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback if hull fails
        return 1.0

def _score(pts):
    """Compute the score: min triangle area / convex hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_a

def _negative_score(pts):
    """Negative score for minimization."""
    return -_score(pts)

def _grad_negative_score(pts):
    """
    Compute gradient of negative score.
    This is complex; we'll use numerical differentiation or a simplified approach.
    For now, let's use a finite difference approximation if needed, 
    but since scipy minimize can use BFGS which approximates gradient,
    we can just provide the function.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Generate initial points using Fermat spiral (Vogel's model) for uniformity
    n = N
    i = np.arange(n)
    r = np.sqrt(i / n)
    theta = i * 2.399963229728653  # Golden angle in radians
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    best_pts = np.column_stack((x, y))
    best_pts -= np.mean(best_pts, axis=0)  # Center
    best_score = _score(best_pts)
    
    # 2. Optimization using L-BFGS-B
    # We optimize in 2D coordinates. To maintain convexity and avoid degenerate hulls,
    # we rely on the fact that the Fermat spiral starts with a good convex hull.
    
    def objective(v):
        pts = v.reshape(n, 2)
        return _negative_score(pts)
    
    x0 = best_pts.flatten()
    
    # Try a few restarts with different initializations
    initializations = [x0]
    
    # Add a few random perturbations to the initial points
    rng = np.random.default_rng(42)
    for _ in range(3):
        perturbed = best_pts + rng.normal(0, 0.05, (n, 2))
        perturbed -= np.mean(perturbed, axis=0)
        initializations.append(perturbed.flatten())
    
    for init in initializations:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective, 
                init, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < _negative_score(best_pts):
                new_pts = res.x.reshape(n, 2)
                new_pts -= np.mean(new_pts, axis=0)
                new_score = _score(new_pts)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts
                    
        except Exception:
            continue
    
    # 3. Final polish: Local search with small perturbations
    # This helps fine-tune the position
    if time.time() < deadline - 5:
        # Use a simple coordinate ascent or small random steps
        current_pts = best_pts.copy()
        current_score = best_score
        step_size = 0.01
        improvements = 0
        
        for iter in range(200):
            if time.time() > deadline - 5:
                break
            if improvements > 10:
                step_size *= 0.5
                improvements = 0
                
            # Try moving each point in 4 directions
            improved = False
            for i in range(n):
                for dx, dy in [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size)]:
                    new_pts = current_pts.copy()
                    new_pts[i] += np.array([dx, dy])
                    new_score = _score(new_pts)
                    if new_score > current_score + 1e-10:
                        current_pts = new_pts
                        current_score = new_score
                        improved = True
                        improvements += 1
                        break
                if improved:
                    break
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts
    
    return best_pts

# EVOLVE-BLOCK-END
