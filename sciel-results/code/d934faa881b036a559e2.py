"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

N = 13

def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                areas[i, j, k] = 0.5 * abs(cross)
    return areas

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    try:
        min_tri = np.min(_triangle_areas(points))
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
        return min_tri / hull_area
    except Exception:
        return 0.0

def _generate_points(params):
    """
    Generate 13 points from parameters.
    params: [a1, phi1, a2, phi2, b1, psi1, b2, psi2]
    Angular perturbation: delta_theta = a1*sin(theta + phi1) + a2*sin(2*theta + phi2)
    Radial perturbation: delta_r = b1*cos(theta + psi1) + b2*cos(2*theta + psi2)
    """
    n = 13
    a1, phi1, a2, phi2, b1, psi1, b2, psi2 = params
    
    theta_base = np.arange(n) * 2 * np.pi / n
    
    # Angular perturbations
    delta_theta = a1 * np.sin(theta_base + phi1) + a2 * np.sin(2 * theta_base + phi2)
    theta = theta_base + delta_theta
    
    # Radial perturbations
    radii = 1.0 + b1 * np.cos(theta_base + psi1) + b2 * np.cos(2 * theta_base + psi2)
    
    # Ensure radii are positive to avoid invalid geometry
    radii = np.maximum(radii, 0.1)
    
    points = np.column_stack([radii * np.cos(theta), radii * np.sin(theta)])
    return points

def _objective(params):
    """Negative score for minimization."""
    points = _generate_points(params)
    return -_score(points)

def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    Uses differential evolution for global optimization of Fourier parameters.
    """
    # Bounds for parameters
    # Angular amplitudes (radians)
    a_bounds = (-0.2, 0.2)
    # Radial amplitudes
    b_bounds = (-0.2, 0.2)
    # Phases
    phase_bounds = (0, 2 * np.pi)
    
    bounds = [
        a_bounds, # a1
        phase_bounds, # phi1
        a_bounds, # a2
        phase_bounds, # phi2
        b_bounds, # b1
        phase_bounds, # psi1
        b_bounds, # b2
        phase_bounds, # psi2
    ]
    
    # Use differential evolution for robust global optimization
    # maxiter=100, popsize=15 is a reasonable balance for 120s limit
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=True
        )
        best_params = result.x
        best_points = _generate_points(best_params)
        best_score = _score(best_points)
    except Exception:
        best_params = np.zeros(8)
        best_points = _generate_points(best_params)
        best_score = _score(best_points)
    
    # Fallback if score is 0 (should not happen with valid params)
    if best_score <= 0:
        angles = np.arange(N) * 2 * np.pi / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
