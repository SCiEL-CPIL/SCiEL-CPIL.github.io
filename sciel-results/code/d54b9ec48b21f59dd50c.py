import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    n = 12
    start_time = time.time()
    time_limit = 550  # Leave 50s margin
    
    # Normals for the 6 sides of a regular hexagon (angle 0)
    # Vertices at 0, 60, ... degrees.
    # Normals are perpendicular to edges.
    # Edge between 0 and 60 deg. Midpoint angle 30 deg. Normal angle 30+90=120?
    # Let's use the standard definition:
    # Hexagon defined by |n_k . x| <= apothem.
    # For a hexagon with vertex at (1,0), the edge is between (1,0) and (0.5, sqrt(3)/2).
    # The normal to this edge points in the direction of the bisector of the angle between the vertices?
    # No, the normal points to the midpoint of the edge.
    # Midpoint of edge 0-60: (0.75, sqrt(3)/4). Angle is 30 deg.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    apothem_unit = np.sqrt(3) / 2.0
    min_dist_centers = np.sqrt(3) # Distance between centers of touching aligned hexagons

    def get_s(x):
        """Calculate the minimum outer hexagon side length s required to contain the hexagons at centers x."""
        # x is shape (12, 2)
        # Projection of centers onto normals
        # proj = x @ normals.T (12, 6)
        proj = x @ normals.T
        max_proj = np.max(np.abs(proj))
        # s * apothem_outer >= max_proj + apothem_inner
        # s * (sqrt(3)/2) >= max_proj + sqrt(3)/2
        # s >= 1 + (2/sqrt(3)) * max_proj
        return 1.0 + (2.0 / np.sqrt(3.0)) * max_proj

    def penalty(x, penalty_weight):
        """Calculate the penalty for overlapping hexagons."""
        # x is (12, 2)
        # Compute pairwise differences
        # diff = x[:, np.newaxis, :] - x[np.newaxis, :, :] (12, 12, 2)
        # This is memory heavy but 12 is small.
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        # Project differences onto normals
        # proj_diff = diff @ normals.T (12, 12, 6)
        proj_diff = diff @ normals.T
        # For each pair (i,j), we need max_k |proj_diff[i,j,k]| >= min_dist_centers
        # max_proj_diff = np.max(np.abs(proj_diff), axis=2) (12, 12)
        max_proj_diff = np.max(np.abs(proj_diff), axis=2)
        # We want max_proj_diff[i,j] >= min_dist_centers for i != j
        # Violation is max(0, min_dist_centers - max_proj_diff[i,j])
        # Zero out diagonal
        mask = ~np.eye(12, dtype=bool)
        violations = np.where(mask, np.maximum(0, min_dist_centers - max_proj_diff), 0)
        return penalty_weight * np.sum(violations**2)

    def objective(x):
        return get_s(x) + penalty(x, 100.0)

    def objective_grad(x):
        # Not implementing gradient, using numerical or BFGS
        return None

    # Heuristic Start: Flower (7) + 5 outer
    sqrt3 = np.sqrt(3)
    centers = np.zeros((12, 2))
    # Center
    centers[0] = [0, 0]
    # 6 Neighbors
    for i in range(6):
        angle = np.deg2rad(60 * i)
        centers[i+1] = [sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)]
    # 5 Outer (at angles 0, 60, 120, 180, 240)
    for i in range(5):
        angle = np.deg2rad(60 * i)
        # Distance 2*sqrt(3) to touch the neighbor
        centers[i+7] = [2*sqrt3 * np.cos(angle), 2*sqrt3 * np.sin(angle)]
    
    best_x = centers
    best_s = get_s(centers)
    best_penalty = penalty(centers, 1.0)
    
    # Optimization
    # Use L-BFGS-B for local refinement
    try:
        res = minimize(objective, centers, method='L-BFGS-B', options={'maxiter': 2000})
        if res.success:
            s_val = get_s(res.x)
            pen_val = penalty(res.x, 1.0)
            if pen_val < 1e-8 and s_val < best_s:
                best_x = res.x
                best_s = s_val
    except Exception:
        pass

    # Try Differential Evolution for a global search if time permits
    if time.time() - start_time < time_limit - 100:
        try:
            bounds = [(-5, 5)] * 24
            def de_objective(x_flat):
                x = x_flat.reshape(12, 2)
                return get_s(x) + penalty(x, 1000.0)
            
            # DE is slow, limit iterations
            res_de = differential_evolution(de_objective, bounds, maxiter=50, popsize=10, tol=1e-4, seed=42, polish=False)
            if res_de.success:
                x_de = res_de.x.reshape(12, 2)
                # Refine DE result
                res_refine = minimize(objective, x_de, method='L-BFGS-B', options={'maxiter': 2000})
                if res_refine.success:
                    s_val = get_s(res_refine.x)
                    pen_val = penalty(res_refine.x, 1.0)
                    if pen_val < 1e-8 and s_val < best_s:
                        best_x = res_refine.x
                        best_s = s_val
        except Exception:
            pass

    # Final check and format
    # Ensure finite values
    if not np.all(np.isfinite(best_x)):
        # Fallback to seed-like arrangement
        best_x = np.array([
            [0, 0], [-2.5, 0], [2.5, 0],
            [-1.25, 2.17], [1.25, 2.17], [-1.25, -2.17], [1.25, -2.17],
            [-3.75, 2.17], [3.75, 2.17], [-3.75, -2.17], [3.75, -2.17],
            [0, -4]
        ])
        best_s = 8.0

    # Construct return arrays
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = best_x[:, 0]
    inner_hex_data[:, 1] = best_x[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
