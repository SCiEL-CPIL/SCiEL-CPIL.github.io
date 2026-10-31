# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 13

def _triangle_areas_fast(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    # Vectorized cross product for all pairs? No, triples are O(N^3).
    # For N=13, 13*12*11/6 = 286 triangles. Very fast to loop.
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                areas.append(0.5 * abs(cross))
    return np.array(areas)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_tri = np.min(_triangle_areas_fast(points))
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_tri / hull_area

def _generate_points_from_params(params):
    """
    Generate 13 points on the unit circle with perturbed angles.
    params: [A1, phi1, A2, phi2]
    theta_i = 2*pi*i/13 + A1*cos(2*pi*i/13 + phi1) + A2*cos(4*pi*i/13 + phi2)
    """
    n = 13
    A1, phi1, A2, phi2 = params
    
    # Base angles
    base_angles = np.arange(n) * 2 * np.pi / n
    
    # Perturbations
    # Harmonic 1: frequency 1 relative to index
    # Harmonic 2: frequency 2 relative to index
    # Note: The argument to cos is the "phase" of the perturbation wave.
    # We use the index position to define the wave.
    wave1 = A1 * np.cos(base_angles + phi1)
    wave2 = A2 * np.cos(2 * base_angles + phi2)
    
    angles = base_angles + wave1 + wave2
    
    # Ensure angles are sorted and within [0, 2*pi)
    # Since base angles are sorted and perturbations are small, they should remain sorted.
    # However, to be safe, we can wrap and sort, but sorting might break the "regular polygon" topology if perturbation is large.
    # We assume small perturbations.
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points

def _objective(params):
    """Negative score for minimization."""
    points = _generate_points_from_params(params)
    
    # Check if points are in convex position (all on hull)
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < 13:
            # Penalize non-convex or collinear points
            return -0.01
    except Exception:
        return -0.01
    
    # Check angular order preservation (convexity on circle)
    angles = np.arctan2(points[:, 1], points[:, 0])
    # Unwrap angles to handle 0/2pi boundary
    angles_unwrapped = np.unwrap(angles)
    diffs = np.diff(angles_unwrapped)
    # If any diff is negative or too small, order is broken or points are too close
    if np.any(diffs < 0.1): # 0.1 rad is ~5 deg, base is ~27 deg. Safe threshold.
        return -0.01
        
    return -_score(points)

def _cma_es(objective, x0, sigma, bounds, max_iter=1000, seed=42):
    """
    Minimal CMA-ES implementation.
    x0: initial mean
    sigma: initial step size
    bounds: list of (min, max)
    """
    n = len(x0)
    rng = np.random.RandomState(seed)
    
    # Parameters
    mu = max(1, n // 2)  # number of recombination vectors
    mu_eff = mu
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1) - 1)) + c_sigma
    chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n**2))
    
    # State
    x_mean = np.array(x0, dtype=float)
    C = np.eye(n)
    B = np.eye(n)
    D = np.ones(n)
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    sigma = sigma
    count_evals = 0
    best_x = x_mean.copy()
    best_f = objective(x_mean)
    
    for it in range(max_iter):
        # Sample offspring
        offspring = []
        for i in range(mu):
            z = rng.standard_normal(n)
            # Apply CMA transform: B * D * z
            y = B @ (D * z)
            x_new = x_mean + sigma * y
            
            # Clip to bounds
            for j in range(n):
                if x_new[j] < bounds[j][0]:
                    x_new[j] = bounds[j][0]
                elif x_new[j] > bounds[j][1]:
                    x_new[j] = bounds[j][1]
            
            offspring.append(x_new)
        
        # Evaluate
        fitnesses = [objective(x) for x in offspring]
        
        # Sort by fitness (minimization)
        indices = np.argsort(fitnesses)
        offspring_sorted = [offspring[i] for i in indices]
        fitnesses_sorted = [fitnesses[i] for i in indices]
        
        # Update best
        if fitnesses_sorted[0] < best_f:
            best_f = fitnesses_sorted[0]
            best_x = offspring_sorted[0].copy()
        
        # Recombination
        x_new_mean = np.sum(np.array([weights[i] * offspring_sorted[i] for i in range(mu)]), axis=0)
        
        # Step-size control
        # Calculate y_mean
        y_mean = (x_new_mean - x_mean) / sigma
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ y_mean) / D
        h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1)))) / chi_n < 1.4 + 2 / (n + 1)
        
        if h_sigma:
            p_c = (1 - c_c) * p_c + c_c * y_mean
        else:
            p_c = (1 - c_c) * p_c
            
        # Update C
        # Rank-1 and Rank-mu update
        v = np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2 * (it + 1))) / chi_n
        C = (1 + c_sigma * d_sigma * c_sigma * (h_sigma - 1) * np.outer(p_sigma, p_sigma) + c_c * (v - 1) * np.outer(p_c, p_c)) * C
        
        # Rank-mu
        y_sorted = [(x_new_mean - x_mean) / sigma] # Wait, need individual y's
        # Recalculate y's for sorted offspring
        y_sorted = []
        for i in range(mu):
            y = (offspring_sorted[i] - x_mean) / sigma
            y_sorted.append(y)
            
        C += c_sigma * mu_eff * np.sum([weights[i] * np.outer(y_sorted[i], y_sorted[i]) for i in range(mu)], axis=0)
        
        # Decompose C
        B, D = np.linalg.eigh(C)
        # Ensure D is positive
        D = np.sqrt(np.abs(D))
        C = B @ (D[:, None] * D[None, :]) @ B.T
        
        # Update mean
        x_mean = x_new_mean
        
        # Adapt step size
        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
        
        # Check convergence
        if np.all(np.abs(x_mean - best_x) < 1e-6):
            break

    return best_x, -best_f

def construct():
    """
    Construct 13 points using CMA-ES optimization of angular Fourier perturbations.
    """
    # Parameters: A1, phi1, A2, phi2
    # Bounds:
    # A1, A2: small amplitudes to maintain convexity. 0.1 rad is safe.
    # phi1, phi2: 0 to 2*pi
    
    bounds = [
        (-0.15, 0.15),  # A1
        (0, 2 * np.pi), # phi1
        (-0.15, 0.15),  # A2
        (0, 2 * np.pi), # phi2
    ]
    
    # Initial guess: regular polygon (A=0)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Run CMA-ES
    try:
        best_params, best_score = _cma_es(_objective, x0, 0.05, bounds, max_iter=500, seed=42)
    except Exception:
        best_params = x0
        best_score = _score(_generate_points_from_params(x0))
        
    points = _generate_points_from_params(best_params)
    
    # Final verification
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < 13:
             # Fallback to regular polygon
             angles = np.arange(N) * 2 * np.pi / N
             points = np.column_stack([np.cos(angles), np.sin(angles)])
    except Exception:
         angles = np.arange(N) * 2 * np.pi / N
         points = np.column_stack([np.cos(angles), np.sin(angles)])
         
    return points
# EVOLVE-BLOCK-END
