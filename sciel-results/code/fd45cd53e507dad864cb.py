import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Centers of the 11 hexagons in the rigid cluster
    # 1 center
    centers = np.array([
        [0.0, 0.0]
    ])

    # 6 neighbors at dist sqrt(3)
    r1 = np.sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])

    # 4 more at dist 3
    # Angles 30, 90, 150, 210
    r2 = 3.0
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])

    n_hex = len(centers)

    # Vertices of a unit hexagon (side 1)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))] for k in range(6)
    ])

    def get_all_vertices(theta_cluster, theta_inner):
        """
        theta_cluster: rotation of the entire cluster
        theta_inner: rotation of inner hexagons relative to cluster
        Returns: Array of shape (n_hex * 6, 2) containing all vertices
        """
        c = np.cos(theta_cluster)
        s = np.sin(theta_cluster)
        rot_cluster = np.array([[c, -s], [s, c]])
        rotated_centers = centers @ rot_cluster.T

        c2 = np.cos(theta_inner)
        s2 = np.sin(theta_inner)
        rot_inner = np.array([[c2, -s2], [s2, c2]])
        rotated_verts = base_vertices @ rot_inner.T

        all_vertices = []
        for i in range(n_hex):
            verts = rotated_verts + rotated_centers[i]
            all_vertices.append(verts)

        return np.vstack(all_vertices)

    def compute_side_length(theta_outer, theta_cluster, theta_inner):
        """
        Compute the exact minimum outer hexagon side length using projection.
        The outer hexagon has vertices at angles 0, 60, 120, 180, 240, 300 + theta_outer
        and the apothem (distance from center to side) is S * sqrt(3) / 2
        """
        verts = get_all_vertices(theta_cluster, theta_inner)

        # Normals of the outer hexagon sides (pointing inward)
        # For a hexagon with vertices at angle alpha, the side normals are at alpha + 30
        normals = []
        for k in range(6):
            angle = np.deg2rad(30 + k * 60) + theta_outer
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)  # shape (6, 2)

        # Project all vertices onto all normals
        dots = verts @ normals.T  # shape (n_hex*6, 6)
        max_proj = np.max(dots)  # shape (6,)

        # Required S for each direction: S = max_proj / (sqrt(3)/2)
        S_values = max_proj / (np.sqrt(3) / 2)

        # The side length must satisfy all constraints
        S = np.max(S_values)
        return S

    def smooth_objective(params, beta=50.0):
        """
        Smooth objective using log-sum-exp for differentiability.
        params: [theta_outer, theta_cluster, theta_inner]
        """
        theta_outer = params[0]
        theta_cluster = params[1]
        theta_inner = params[2]

        verts = get_all_vertices(theta_cluster, theta_inner)

        normals = []
        for k in range(6):
            angle = np.deg2rad(30 + k * 60) + theta_outer
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)

        dots = verts @ normals.T
        smooth_max = np.log(np.sum(np.exp(beta * dots), axis=0)) / beta

        S_values = smooth_max / (np.sqrt(3) / 2)

        beta_outer = 30.0
        S_smooth = np.log(np.sum(np.exp(beta_outer * S_values))) / beta_outer

        return S_smooth

    def exact_objective(params):
        """Exact (nonsmooth) objective for final polish."""
        theta_outer = params[0]
        theta_cluster = params[1]
        theta_inner = params[2]
        return compute_side_length(theta_outer, theta_cluster, theta_inner)

    # Initial guess and bounds
    x0 = np.array([0.0, 0.0, 0.0])
    bounds = [(0, np.pi/3), (0, np.pi/3), (0, np.pi/3)]

    best_score = float('inf')
    best_params = None

    # Multi-start optimization
    n_starts = 12
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
    ]

    for i in range(n_starts):
        if time.time() > deadline - 25:
            break

        if i < len(start_configs):
            x_start = np.array(start_configs[i])
        else:
            x_start = np.random.uniform(0, np.pi/3, 3)

        try:
            # First pass with smooth objective using Powell
            res = minimize(smooth_objective, x_start, method='Powell', bounds=bounds,
                          options={'maxiter': 500, 'disp': False, 'ftol': 1e-12, 'xtol': 1e-12})

            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # If we found a good solution, do a high-precision polish with exact objective
    if best_params is not None and time.time() < deadline - 10:
        try:
            res2 = minimize(exact_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 
                                   'adaptive': True, 'disp': False})

            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    # Additional polish with L-BFGS-B on smooth objective near the solution
    if best_params is not None and time.time() < deadline - 5:
        try:
            # Create a slightly smoothed version for gradient-based refinement
            def refined_smooth_objective(params, beta=200.0):
                theta_outer = params[0]
                theta_cluster = params[1]
                theta_inner = params[2]

                verts = get_all_vertices(theta_cluster, theta_inner)

                normals = []
                for k in range(6):
                    angle = np.deg2rad(30 + k * 60) + theta_outer
                    normals.append([np.cos(angle), np.sin(angle)])
                normals = np.array(normals)

                dots = verts @ normals.T
                smooth_max = np.log(np.sum(np.exp(beta * dots), axis=0)) / beta

                S_values = smooth_max / (np.sqrt(3) / 2)

                beta_outer = 100.0
                S_smooth = np.log(np.sum(np.exp(beta_outer * S_values))) / beta_outer

                return S_smooth

            res3 = minimize(refined_smooth_objective, best_params, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})

            exact_val = exact_objective(res3.x)
            if exact_val < best_score:
                best_score = exact_val
                best_params = res3.x.copy()
        except Exception:
            pass

    if best_params is None:
        return _fallback_solution()

    # Extract best solution
    theta_outer = best_params[0]
    theta_cluster = best_params[1]
    theta_inner = best_params[2]

    # Compute final centers and angles
    c = np.cos(theta_cluster)
    s = np.sin(theta_cluster)
    rot_cluster = np.array([[c, -s], [s, c]])
    final_centers = centers @ rot_cluster.T

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
