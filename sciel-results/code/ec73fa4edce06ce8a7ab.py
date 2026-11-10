# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import itertools


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """

    n = 14
    d = 3
    deadline = time.time() + 620  # Leave 10s margin for safety

    # Helper to calculate the score: (d_min / d_max)^2
    def get_score(pts):
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max <= 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Helper to calculate negative score for minimization
    def neg_score(pts):
        return -get_score(pts)

    # Generate initial points using various strategies
    def generate_initial_points(seed):
        np.random.seed(seed)
        # Try different initial configurations
        configs = []
        
        # Random points
        pts = np.random.randn(n, d)
        configs.append(pts)
        
        # Points on a sphere
        pts = np.random.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        configs.append(pts)
        
        # Points on a grid-like structure
        pts = []
        for i in range(n):
            angle1 = 2 * np.pi * i / n
            angle2 = np.pi * i / n
            r = 1.0
            pts.append([r * np.sin(angle2) * np.cos(angle1),
                       r * np.sin(angle2) * np.sin(angle1),
                       r * np.cos(angle2)])
        configs.append(np.array(pts))
        
        return configs

    # Simulated annealing
    def anneal(initial_points, max_iters=50000, start_temp=1.0, end_temp=0.0001):
        current_points = initial_points.copy()
        current_score = get_score(current_points)
        best_points = current_points.copy()
        best_score = current_score
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Generate neighbor by moving one point
            idx = np.random.randint(n)
            step_size = 0.1 * (1 - i / max_iters) + 0.01
            new_points = current_points.copy()
            new_points[idx] += np.random.randn(d) * step_size
            
            new_score = get_score(new_points)
            
            # Accept if better or with probability
            if new_score > current_score or np.random.rand() < np.exp((new_score - current_score) / start_temp):
                current_points = new_points
                current_score = new_score
                start_temp *= 0.99995
                
                if new_score > best_score:
                    best_points = new_points.copy()
                    best_score = new_score
        
        return best_points, best_score

    # Gradient descent on the sphere
    def gradient_descent_sphere(initial_points, max_iters=10000):
        points = initial_points.copy()
        best_points = points.copy()
        best_score = get_score(best_points)
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient of sum(1/d^2)
            grad = np.zeros_like(points)
            for i_idx in range(n):
                for j_idx in range(i_idx + 1, n):
                    diff = points[i_idx] - points[j_idx]
                    dist_sq = np.sum(diff ** 2)
                    if dist_sq > 1e-10:
                        grad[i_idx] += 2 * diff / (dist_sq ** 2)
                        grad[j_idx] -= 2 * diff / (dist_sq ** 2)
            
            # Project gradient onto tangent space of sphere
            for i_idx in range(n):
                norm_sq = np.sum(points[i_idx] ** 2)
                if norm_sq > 1e-10:
                    grad[i_idx] -= np.dot(grad[i_idx], points[i_idx]) / norm_sq * points[i_idx]
            
            # Update points
            lr = 0.01 * (1 - i / max_iters) + 0.001
            points -= lr * grad
            
            # Renormalize to sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
            score = get_score(points)
            if score > best_score:
                best_points = points.copy()
                best_score = score
        
        return best_points, best_score

    # Nelder-Mead polish
    def nm_polish(initial_points, max_iters=1000):
        if time.time() > deadline - 10:
            return initial_points, get_score(initial_points)
        
        result = minimize(neg_score, initial_points, method='Nelder-Mead',
                         options={'maxiter': max_iters, 'xatol': 1e-8, 'fatol': 1e-8})
        return result.x, -result.fun

    # Main optimization loop
    best_points = None
    best_score = -1.0
    
    # Try multiple initial configurations
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        
        configs = generate_initial_points(seed)
        
        for config in configs:
            if time.time() > deadline - 20:
                break
            
            # Phase 1: Gradient descent on sphere
            pts, score = gradient_descent_sphere(config, max_iters=5000)
            if score > best_score:
                best_points = pts.copy()
                best_score = score
            
            # Phase 2: Nelder-Mead polish
            if time.time() > deadline - 15:
                break
            pts, score = nm_polish(pts, max_iters=500)
            if score > best_score:
                best_points = pts.copy()
                best_score = score
            
            # Phase 3: Simulated annealing from current best
            if time.time() > deadline - 10:
                break
            pts, score = anneal(pts, max_iters=20000)
            if score > best_score:
                best_points = pts.copy()
                best_score = score

    # If we didn't find anything, return random points
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    return best_points


# EVOLVE-BLOCK-END
