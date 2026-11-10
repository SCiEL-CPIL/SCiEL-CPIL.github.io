# EVOLVE-BLOCK-START
import numpy as np
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n_outer = 11
    n_inner = 5

    # Precompute outer angles
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    cos_outer = np.cos(angles_outer)
    sin_outer = np.sin(angles_outer)

    # For given r and phi, compute min and max distances analytically
    def compute_min_max(r, phi):
        # Inner angles
        angles_inner = phi + np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
        cos_inner = r * np.cos(angles_inner)
        sin_inner = r * np.sin(angles_inner)

        min_d2 = np.inf
        max_d2 = 0.0

        # Outer-outer distances
        for i in range(n_outer):
            for j in range(i + 1, n_outer):
                dx = cos_outer[i] - cos_outer[j]
                dy = sin_outer[i] - sin_outer[j]
                d2 = dx * dx + dy * dy
                if d2 < min_d2:
                    min_d2 = d2
                if d2 > max_d2:
                    max_d2 = d2

        # Inner-inner distances
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                dx = cos_inner[i] - cos_inner[j]
                dy = sin_inner[i] - sin_inner[j]
                d2 = dx * dx + dy * dy
                if d2 < min_d2:
                    min_d2 = d2
                if d2 > max_d2:
                    max_d2 = d2

        # Inner-outer distances
        for i in range(n_inner):
            for j in range(n_outer):
                dx = cos_inner[i] - cos_outer[j]
                dy = sin_inner[i] - sin_outer[j]
                d2 = dx * dx + dy * dy
                if d2 < min_d2:
                    min_d2 = d2
                if d2 > max_d2:
                    max_d2 = d2

        return min_d2, max_d2

    def compute_score(r, phi):
        min_d2, max_d2 = compute_min_max(r, phi)
        if max_d2 <= 0:
            return 0.0
        d_min = math.sqrt(min_d2)
        d_max = math.sqrt(max_d2)
        return (d_min / d_max) ** 2

    # Grid search over r and phi
    best_score = 0
    best_r = 0.47
    best_phi = math.pi / 11

    # Search over r in [0.3, 0.7] and phi in [0, 2*pi/11]
    n_r = 100
    n_phi = 100
    r_vals = np.linspace(0.30, 0.70, n_r)
    phi_vals = np.linspace(0, 2 * math.pi / n_outer, n_phi)

    for r in r_vals:
        for phi in phi_vals:
            score = compute_score(r, phi)
            if score > best_score:
                best_score = score
                best_r = r
                best_phi = phi

    # Local refinement
    n_r_fine = 50
    n_phi_fine = 50
    r_fine = np.linspace(best_r - 0.02, best_r + 0.02, n_r_fine)
    phi_fine = np.linspace(max(0, best_phi - 0.1), min(2 * math.pi / n_outer, best_phi + 0.1), n_phi_fine)

    for r in r_fine:
        for phi in phi_fine:
            score = compute_score(r, phi)
            if score > best_score:
                best_score = score
                best_r = r
                best_phi = phi

    # Construct final points
    angles_inner = best_phi + np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    
    outer_points = np.column_stack((cos_outer, sin_outer))
    inner_points = best_r * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
    
    points = np.vstack((outer_points, inner_points))
    
    return points

# EVOLVE-BLOCK-END
