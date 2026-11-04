import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s kill

    # Centers of the 11 hexagons in the rigid cluster
    # 1 center
    centers = np.array([[0.0, 0.0]])

    # 6 neighbors at dist sqrt(3)
    r1 = np.sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])

    # 4 more at dist 3
    r2 = 3.0
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])

    n_hex = len(centers)

    # Vertices of a unit hexagon (side 1), pointy-top orientation
    base_vertices = np.array([
        [np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))] for k in range(6)
    ])

    def get_all_vertices(theta_cluster, theta_inner):
        c = np.cos(theta_cluster)
        s = np.sin(theta_cluster)
        rot_cluster = np.array([[c, -s], [s, c]])
        rotated_centers = centers @ rot_cluster.T

        c2 = np.cos(theta_inner)
        s2 = np.sin(theta_inner)
        rot_inner = np.array([[c2, -s2], [s2, c2]])
        rotated_verts = base_vertices @ rot_inner.T

        all_vertices = np.empty((n_hex * 6, 2))
        for i in range(n_hex):
            all_vertices[i*6:(i+1)*6] = rotated_verts + rotated_centers[i]
        return all_vertices

    def min_side_for_theta(theta_cluster, theta_inner):
        """Find minimum outer side length over all outer rotations."""
        verts = get_all_vertices(theta_cluster, theta_inner)

        def side_for_outer(theta_outer):
            normals = np.array([
                [np.cos(np.deg2rad(30 + k*60) + theta_outer),
                 np.sin(np.deg2rad(30 + k*60) + theta_outer)]
                for k in range(6)
            ])
            dots = verts @ normals.T
            return np.max(dots) / (np.sqrt(3) / 2)

        # Sample 12 angles for initial best
        best_s = float('inf')
        best_t = 0.0
        for i in range(12):
            t = i * np.pi / 6
            s = side_for_outer(t)
            if s < best_s:
                best_s = s
                best_t = t

        # Refine with Nelder-Mead
        res = minimize(side_for_outer, best_t, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12, 'disp': False})
        if res.fun < best_s:
            best_s = res.fun
            best_t = res.x[0]

        return best_s, best_t

    def objective(params):
        """Smooth objective over cluster and inner rotations."""
        theta_cluster = params[0]
        theta_inner = params[1]

        verts = get_all_vertices(theta_cluster, theta_inner)

        # Use multiple outer angle samples for smoothness
        best_s = 0.0
        for i in range(6):
            theta_outer = i * np.pi / 6
            normals = np.array([
                [np.cos(np.deg2rad(30 + k*60) + theta_outer),
                 np.sin(np.deg2rad(30 + k*60) + theta_outer)]
                for k in range(6)
            ])
            dots = verts @ normals.T
            s = np.max(dots) / (np.sqrt(3) / 2)
            if s > best_s:
                best_s = s
        return best_s

    def smooth_objective(params):
        """Log-sum-exp smoothed objective."""
        theta_cluster = params[0]
        theta_inner = params[1]

        verts = get_all_vertices(theta_cluster, theta_inner)

        beta = 80.0
        smooth_max = 0.0
        for i in range(6):
            theta_outer = i * np.pi / 6
            normals = np.array([
                [np.cos(np.deg2rad(30 + k*60) + theta_outer),
                 np.sin(np.deg2rad(30 + k*60) + theta_outer)]
                for k in range(6)
            ])
            dots = verts @ normals.T
            s = np.log(np.sum(np.exp(beta * dots), axis=0)) / beta
            s = s / (np.sqrt(3) / 2)
            smooth_max = max(smooth_max, s)
        return smooth_max

    best_score = float('inf')
    best_params = None
    best_outer = 0.0

    # Multi-start optimization
    n_starts = 15
    for i in range(n_starts):
        if time.time() > deadline - 25:
            break

        if i == 0:
            x_start = np.array([0.0, 0.0])
        elif i == 1:
            x_start = np.array([0.0, np.pi/6])
        elif i == 2:
            x_start = np.array([np.pi/6, 0.0])
        elif i == 3:
            x_start = np.array([np.pi/6, np.pi/6])
        elif i == 4:
            x_start = np.array([np.pi/12, np.pi/12])
        elif i == 5:
            x_start = np.array([np.pi/3, 0.0])
        elif i == 6:
            x_start = np.array([0.0, np.pi/3])
        elif i == 7:
            x_start = np.array([np.pi/3, np.pi/3])
        elif i == 8:
            x_start = np.array([np.pi/4, np.pi/4])
        elif i == 9:
            x_start = np.array([np.pi/2, np.pi/2])
        elif i == 10:
            x_start = np.array([np.pi/5, np.pi/10])
        elif i == 11:
            x_start = np.array([np.pi/10, np.pi/5])
        elif i == 12:
            x_start = np.array([2*np.pi/5, np.pi/15])
        elif i == 13:
            x_start = np.array([np.pi/15, 2*np.pi/5])
        else:
            x_start = np.random.uniform(0, np.pi/3, 2)

        try:
            res = minimize(smooth_objective, x_start, method='Powell',
                          bounds=[(0, np.pi/3), (0, np.pi/3)],
                          options={'maxiter': 400, 'disp': False})

            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final precise evaluation and outer rotation optimization
    if best_params is not None:
        if time.time() < deadline - 10:
            # Evaluate precisely
            s, t = min_side_for_theta(best_params[0], best_params[1])
            best_score = s
            best_outer = t

            # Try small perturbations of cluster/inner rotations
            for dc in [-0.01, 0.01, -0.02, 0.02]:
                for di in [-0.01, 0.01, -0.02, 0.02]:
                    tc = (best_params[0] + dc) % (np.pi/3)
                    ti = (best_params[1] + di) % (np.pi/3)
                    s2, t2 = min_side_for_theta(tc, ti)
                    if s2 < best_score:
                        best_score = s2
                        best_params = np.array([tc, ti])
                        best_outer = t2

            # Nelder-Mead polish on exact objective
            def exact_obj(params):
                s, _ = min_side_for_theta(params[0], params[1])
                return s

            if time.time() < deadline - 5:
                try:
                    res2 = minimize(exact_obj, best_params, method='Nelder-Mead',
                                   options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12, 'disp': False})
                    if res2.fun < best_score:
                        best_score = res2.fun
                        best_params = res2.x.copy()
                        s, t = min_side_for_theta(best_params[0], best_params[1])
                        best_score = s
                        best_outer = t
                except Exception:
                    pass

    if best_params is None:
        return _fallback_solution()

    theta_cluster = best_params[0]
    theta_inner = best_params[1]

    c = np.cos(theta_cluster)
    s = np.sin(theta_cluster)
    rot_cluster = np.array([[c, -s], [s, c]])
    final_centers = centers @ rot_cluster.T

    inner = []
    for i in range(n_hex):
        inner.append([final_centers[i][0], final_centers[i][1], np.rad2deg(theta_inner)])

    side = best_score + 1e-9
    return inner, (0.0, 0.0), side, np.rad2deg(best_outer)

def _fallback_solution():
    eps = 1e-9
    centers = [(0.0, 0.0)]
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append(((np.sqrt(3) + eps) * np.cos(angle), (np.sqrt(3) + eps) * np.sin(angle)))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append(((3.0 + 2*eps) * np.cos(angle), (3.0 + 2*eps) * np.sin(angle)))

    inner = [[x, y, 30.0] for x, y in centers]
    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    return inner, (0.0, 0.0), side, 30.0

if __name__ == "__main__":
    construct()
