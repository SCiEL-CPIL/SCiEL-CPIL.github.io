import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s limit

    # Centers of the 11 hexagons in the rigid cluster
    # 1 center
    centers = np.array([[0.0, 0.0]])

    # 6 neighbors at dist sqrt(3)
    r1 = np.sqrt(3)
    angles_r1 = np.deg2rad(np.arange(6) * 60)
    neighbors_r1 = np.column_stack([r1 * np.cos(angles_r1), r1 * np.sin(angles_r1)])
    centers = np.vstack([centers, neighbors_r1])

    # 4 more at dist 3
    r2 = 3.0
    angles_r2 = np.deg2rad(30 + np.arange(4) * 60)
    neighbors_r2 = np.column_stack([r2 * np.cos(angles_r2), r2 * np.sin(angles_r2)])
    centers = np.vstack([centers, neighbors_r2])

    n_hex = len(centers)

    # Vertices of a unit hexagon (side 1)
    base_angles = np.deg2rad(np.arange(6) * 60)
    base_vertices = np.column_stack([np.cos(base_angles), np.sin(base_angles)])

    # Precompute base vertex offsets for each hexagon (shape: n_hex, 6, 2)
    # This allows vectorized computation of all vertices
    center_offsets = centers.reshape(n_hex, 1, 2)  # (n_hex, 1, 2)
    
    def get_all_vertices(theta_cluster, theta_inner):
        """
        Returns: Array of shape (n_hex * 6, 2) containing all vertices
        Vectorized version for speed.
        """
        c = np.cos(theta_cluster)
        s = np.sin(theta_cluster)
        # Rotate centers: (n_hex, 2) -> (n_hex, 2)
        rotated_centers = np.column_stack([
            centers[:, 0] * c - centers[:, 1] * s,
            centers[:, 0] * s + centers[:, 1] * c
        ])

        c2 = np.cos(theta_inner)
        s2 = np.sin(theta_inner)
        # Rotate base vertices: (6, 2) -> (6, 2)
        rotated_verts = np.column_stack([
            base_vertices[:, 0] * c2 - base_vertices[:, 1] * s2,
            base_vertices[:, 0] * s2 + base_vertices[:, 1] * c2
        ])

        # All vertices: (n_hex, 6, 2)
        all_vertices = rotated_verts.reshape(1, 6, 2) + rotated_centers.reshape(n_hex, 1, 2)
        return all_vertices.reshape(n_hex * 6, 2)

    def compute_side_length(theta_outer, theta_cluster, theta_inner):
        """
        Compute the exact minimum outer hexagon side length using projection.
        """
        verts = get_all_vertices(theta_cluster, theta_inner)

        # Normals of the outer hexagon sides
        outer_angles = np.deg2rad(30 + np.arange(6) * 60) + theta_outer
        normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])  # (6, 2)

        # Project all vertices onto all normals
        dots = verts @ normals.T  # (n_hex*6, 6)
        max_proj = np.max(dots, axis=0)  # (6,)

        # Required S for each direction
        S_values = max_proj / (np.sqrt(3) / 2)
        return np.max(S_values)

    def smooth_objective(params, beta_inner=100.0, beta_outer=50.0):
        """
        Smooth objective using log-sum-exp.
        params: [theta_outer, theta_cluster, theta_inner]
        """
        theta_outer = params[0]
        theta_cluster = params[1]
        theta_inner = params[2]

        verts = get_all_vertices(theta_cluster, theta_inner)

        outer_angles = np.deg2rad(30 + np.arange(6) * 60) + theta_outer
        normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])

        dots = verts @ normals.T  # (n_hex*6, 6)
        
        # Log-sum-exp for max projection per normal
        max_dots = np.max(dots, axis=0)
        smooth_max = max_dots + np.log(np.sum(np.exp(beta_inner * (dots - max_dots)), axis=0)) / beta_inner

        S_values = smooth_max / (np.sqrt(3) / 2)

        # Log-sum-exp for max S
        max_S = np.max(S_values)
        S_smooth = max_S + np.log(np.sum(np.exp(beta_outer * (S_values - max_S)))) / beta_outer

        return S_smooth

    def exact_objective(params):
        """Exact (nonsmooth) objective for final polish."""
        theta_outer = params[0]
        theta_cluster = params[1]
        theta_inner = params[2]
        return compute_side_length(theta_outer, theta_cluster, theta_inner)

    # Initial guess and bounds
    bounds = [(0, np.pi/3), (0, np.pi/3), (0, np.pi/3)]

    best_score = float('inf')
    best_params = None

    # Multi-start optimization with strategic starting points
    start_configs = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, np.pi/6],
        [0.0, np.pi/6, 0.0],
        [0.0, np.pi/6, np.pi/6],
        [np.pi/6, 0.0, 0.0],
        [np.pi/6, 0.0, np.pi/6],
        [np.pi/6, np.pi/6, 0.0],
        [np.pi/6, np.pi/6, np.pi/6],
        [np.pi/12, np.pi/12, np.pi/12],
        [np.pi/4, np.pi/4, np.pi/4],
        [np.pi/3, 0.0, 0.0],
        [0.0, np.pi/3, np.pi/3],
        [np.pi/6, np.pi/12, np.pi/12],
        [np.pi/12, np.pi/6, np.pi/12],
        [np.pi/12, np.pi/12, np.pi/6],
    ]

    n_starts = len(start_configs)
    for i in range(n_starts):
        if time.time() > deadline - 20:
            break

        x_start = np.array(start_configs[i])

        try:
            # Powell with smooth objective
            res = minimize(smooth_objective, x_start, method='Powell', bounds=bounds,
                          options={'maxiter': 300, 'disp': False, 'ftol': 1e-10, 'xtol': 1e-10})

            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # If we found a good solution, do a high-precision polish
    if best_params is not None and time.time() < deadline - 8:
        try:
            # Nelder-Mead on exact objective
            res2 = minimize(exact_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 
                                   'adaptive': True, 'disp': False})

            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    # Final high-precision polish with very smooth surrogate
    if best_params is not None and time.time() < deadline - 3:
        try:
            def very_smooth_objective(params):
                return smooth_objective(params, beta_inner=500.0, beta_outer=200.0)

            res3 = minimize(very_smooth_objective, best_params, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-14})

            exact_val = exact_objective(res3.x)
            if exact_val < best_score:
                best_score = exact_val
                best_params = res3.x.copy()
        except Exception:
            pass

    # One more polish pass if time allows
    if best_params is not None and time.time() < deadline - 1:
        try:
            res4 = minimize(exact_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-15, 'fatol': 1e-15,
                                   'adaptive': True, 'disp': False})
            if res4.fun < best_score:
                best_score = res4.fun
                best_params = res4.x.copy()
        except Exception:
            pass

    if best_params is None:
        return _fallback_solution()

    # Extract best solution
    theta_outer = best_params[0]
    theta_cluster = best_params[1]
    theta_inner = best_params[2]

    # Compute final centers
    c = np.cos(theta_cluster)
    s = np.sin(theta_cluster)
    final_centers = np.column_stack([
        centers[:, 0] * c - centers[:, 1] * s,
        centers[:, 0] * s + centers[:, 1] * c
    ])

    inner = []
    for i in range(n_hex):
        inner.append([final_centers[i][0], final_centers[i][1], np.rad2deg(theta_inner)])

    # Compute exact side length with safety margin
    side = compute_side_length(theta_outer, theta_cluster, theta_inner)
    side += 1e-9

    return inner, (0.0, 0.0), side, np.rad2deg(theta_outer)

def _fallback_solution():
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))

    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])

    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    return inner, (0.0, 0.0), side, 30.0

if __name__ == "__main__":
    construct()
