import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform

N = 100

def construct():
    # Define lattice generators
    def get_fcc_points(L):
        pts = []
        for i in range(-L, L + 1):
            for j in range(-L, L + 1):
                for k in range(-L, L + 1):
                    if (i + j + k) % 2 == 0:
                        pts.append([i, j, k])
        return np.array(pts, dtype=float)

    def get_bcc_points(L):
        pts = []
        # Corners and centers
        for i in range(-L, L + 1):
            for j in range(-L, L + 1):
                for k in range(-L, L + 1):
                    pts.append([i, j, k])
                    pts.append([i + 0.5, j + 0.5, k + 0.5])
        return np.array(pts, dtype=float)

    def get_sc_points(L):
        pts = []
        for i in range(-L, L + 1):
            for j in range(-L, L + 1):
                for k in range(-L, L + 1):
                    pts.append([i, j, k])
        return np.array(pts, dtype=float)

    lattices = {
        'fcc': get_fcc_points(4),
        'bcc': get_bcc_points(3),
        'sc': get_sc_points(4)
    }

    best_radius = -1
    best_centers = None

    for name, points in lattices.items():
        # Center the points
        center = np.mean(points, axis=0)
        points -= center

        # Select N points with smallest L-infinity norm (max coordinate)
        # This favors a cubic shape over a spherical one
        l_inf_norms = np.max(np.abs(points), axis=1)
        idx = np.argsort(l_inf_norms)
        points = points[idx[:N]]

        # Recenter to ensure symmetry
        center = np.mean(points, axis=0)
        points -= center

        # Calculate D_min
        dist_matrix = pdist(points)
        D_min = np.min(dist_matrix)
        
        if D_min == 0:
            continue

        # Define objective function with rotation and translation
        def rot_matrix(angles):
            roll, pitch, yaw = angles
            c_r, s_r = np.cos(roll), np.sin(roll)
            c_p, s_p = np.cos(pitch), np.sin(pitch)
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            
            Rx = np.array([
                [1, 0, 0],
                [0, c_r, -s_r],
                [0, s_r, c_r]
            ])
            Ry = np.array([
                [c_p, 0, s_p],
                [0, 1, 0],
                [-s_p, 0, c_p]
            ])
            Rz = np.array([
                [c_y, -s_y, 0],
                [s_y, c_y, 0],
                [0, 0, 1]
            ])
            return Rz @ Ry @ Rx

        def objective(params):
            roll, pitch, yaw, tx, ty, tz = params
            R = rot_matrix([roll, pitch, yaw])
            t = np.array([tx, ty, tz])
            rotated = points @ R.T
            translated = rotated + t
            W_max = np.max(np.abs(translated))
            return W_max

        # Optimize
        # Bounds for angles
        angle_bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        # Bounds for translation (roughly within the bounding box)
        trans_bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        bounds = angle_bounds + trans_bounds

        # Use multiple starts with Powell or Nelder-Mead
        # Powell is often faster and more robust for this type of problem
        best_w_max = np.inf
        best_params = None
        
        # Initial guesses
        rng = np.random.RandomState(42)
        n_starts = 20
        for i in range(n_starts):
            x0 = np.array([
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-np.pi, np.pi),
                0.0, 0.0, 0.0
            ])
            # Try a few different optimizers or just Powell
            res = minimize(objective, x0, method='Powell', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
            if res.fun < best_w_max:
                best_w_max = res.fun
                best_params = res.x
        
        # Refine the best result with Nelder-Mead for higher precision
        res = minimize(objective, best_params, method='Nelder-Mead', bounds=bounds, options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res.fun < best_w_max:
            best_w_max = res.fun
            best_params = res.x

        roll, pitch, yaw, tx, ty, tz = best_params
        R_opt = rot_matrix([roll, pitch, yaw])
        t_opt = np.array([tx, ty, tz])
        
        # Calculate scale and final positions
        # r = 0.5 * D_min / (2 * W_max + D_min)
        # s = r / (0.5 - W_max)
        # Alternatively, s = 0.5 / (W_max + D_min / 2)
        s = 0.5 / (best_w_max + D_min / 2.0)
        centers = s * (points @ R_opt.T + t_opt)
        
        # Calculate actual radius
        dists = pdist(centers)
        r_centers = np.min(dists) / 2 if len(dists) > 0 else 0
        r_bounds = 0.5 - np.max(np.abs(centers))
        r = min(r_centers, r_bounds)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers

    return best_centers
