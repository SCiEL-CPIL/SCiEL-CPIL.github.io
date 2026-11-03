"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Place 12 unit hexagons in a compact hexagonal lattice arrangement
    # Hexagonal lattice: adjacent centers are sqrt(3) apart for edge-to-edge contact
    # Use axial coordinates (q, r) -> Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    
    sqrt3 = np.sqrt(3)
    
    # Define 12 positions in a compact hexagonal cluster
    # Pattern: rows of 2, 4, 4, 2 in axial coordinates
    axial_coords = [
        (-1, -2), (0, -2),
        (-2, -1), (-1, -1), (0, -1), (1, -1),
        (-2, 0), (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1),
    ]
    
    centers = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Center the cluster at origin
    center_of_mass = np.mean(centers, axis=0)
    centers = centers - center_of_mass
    
    # Find minimal regular hexagon containing all centers
    # A regular hexagon with center at origin, side s, rotation theta
    # Point p is inside if |p · n_k| <= s*sqrt(3)/2 for all 6 normals n_k
    
    def outer_side_length(theta):
        """Find minimum side length s for hexagon with rotation theta to contain all centers."""
        max_apothem = 0.0
        for k in range(6):
            angle = theta + np.pi / 6 + k * np.pi / 3
            nx = np.cos(angle)
            ny = np.sin(angle)
            dots = centers[:, 0] * nx + centers[:, 1] * ny
            max_abs = np.max(np.abs(dots))
            max_apothem = max(max_apothem, max_abs)
        s = max_apothem * 2.0 / np.sqrt(3)
        return s
    
    # Search over rotation angle (hexagon has 60-degree symmetry, so search [0, pi/3))
    best_s = float('inf')
    best_theta = 0.0
    
    # Coarse grid search
    for theta in np.linspace(0, np.pi / 3, 73):
        s = outer_side_length(theta)
        if s < best_s:
            best_s = s
            best_theta = theta
    
    # Refine with Nelder-Mead
    def objective(theta_arr):
        return outer_side_length(theta_arr[0])
    
    res = minimize(objective, [best_theta], method='Nelder-Mead',
                   options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000, 'maxfev': 2000})
    if res.fun < best_s:
        best_s = res.fun
        best_theta = res.x[0]
    
    # Construct inner hexagon positions (all axis-aligned)
    inner = []
    for cx, cy in centers:
        inner.append([float(cx), float(cy), 0.0])
    
    # Outer hexagon: center at origin, side length best_s, rotation best_theta (in degrees)
    return inner, (0.0, 0.0), float(best_s), float(np.degrees(best_theta))
# EVOLVE-BLOCK-END
