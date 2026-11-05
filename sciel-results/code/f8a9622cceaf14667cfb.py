import numpy as np
import time
import itertools
from scipy.optimize import minimize, basinhopping, minimize_scalar

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s limit

    def elapsed():
        return time.time() - start_time

    eps = 1e-12

    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    def compute_outer_side(theta_out, phis, centers):
        """
        Vectorized computation of minimum outer hexagon side length.
        """
        n_hex = len(centers)
        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]  # (6, 2)
            dots = verts @ normals.T  # (6, 6)
            max_proj = max(max_proj, np.max(np.abs(dots)))

        return 2.0 * max_proj / np.sqrt(3)

    def smooth_objective(params, centers, beta=50.0):
        """
        Smooth surrogate using soft-max for the max over vertices.
        params: [theta_out, phi_0, ..., phi_{n-1}]
        """
        theta_out = params[0]
        phis = params[1:]
        n_hex = len(centers)
        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

        all_abs_dots = []
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]  # (6, 2)
            dots = verts @ normals.T  # (6, 6)
            all_abs_dots.append(np.abs(dots))

        all_abs_dots = np.concatenate(all_abs_dots, axis=1)  # (6*n_hex, 6)
        # Soft max over the first axis for each normal direction
        scaled = beta * all_abs_dots
        # Use log-sum-exp for numerical stability
        max_vals = np.max(scaled, axis=0, keepdims=True)
        log_sum_exp = max_vals + np.log(np.sum(np.exp(scaled - max_vals), axis=0, keepdims=True))
        soft_max = log_sum_exp / beta  # (1, 6)

        max_proj = np.max(soft_max)
        return 2.0 * max_proj / np.sqrt(3)

    def objective(params, centers):
        """
        params: [theta_out, phi_0, ..., phi_{n-1}]
        Returns the outer hexagon side length.
        """
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, phis, centers)

    best_score = float('inf')
    best_params = None
    best_centers = None

    # Systematic search over combinations of 4 outer hexagons from 6 positions
    all_angles = [0, 60, 120, 180, 240, 300]
    combinations = list(itertools.combinations(all_angles, 4))

    for combo in combinations:
        if elapsed() > deadline - 20:
            break

        # Build centers
        centers = np.array([[0.0, 0.0]])

        # Inner ring (6) at 30, 90, ... to touch flat sides of center
        r1 = np.sqrt(3) + eps
        for k in range(6):
            angle = np.deg2rad(30 + k * 60)
            centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])

        # Outer ring (4) at chosen angles
        r2 = 3.0 + 2 * eps
        for angle_deg in combo:
            angle = np.deg2rad(angle_deg)
            centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])

        n_hex = len(centers)

        # Analytical initial guess: all inner hexagons at 30 degrees (flat-top), outer at 30 degrees
        x0 = np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex)

        local_best_score = objective(x0, centers)
        local_best_params = x0.copy()

        # Basin Hopping optimization with smooth surrogate
        if elapsed() < deadline - 15:
            try:
                def local_minimizer(x):
                    # Use smooth objective for gradient-like optimization
                    res = minimize(lambda p: smooth_objective(p, centers, beta=50.0), x,
                                   method='Powell',
                                   options={'maxiter': 80, 'disp': False, 'ftol': 1e-14, 'xtol': 1e-14})
                    # Evaluate with exact objective
                    exact_score = objective(res.x, centers)
                    return type('Result', (), {'x': res.x, 'fun': exact_score, 'success': True})()

                def step_taker(x, stepsize):
                    x_new = x + stepsize * np.random.randn(len(x))
                    return x_new

                bh = basinhopping(lambda p: objective(p, centers), x0,
                                  minimizer=local_minimizer,
                                  stepsize=0.3, T=500.0,
                                  niter=30, accept_test=None,
                                  step_taker=step_taker)

                if bh.fun < local_best_score:
                    local_best_score = bh.fun
                    local_best_params = bh.x.copy()
            except Exception:
                pass

        # Final high-precision polish with Nelder-Mead
        if elapsed() < deadline - 5:
            try:
                res = minimize(lambda p: objective(p, centers), local_best_params,
                               method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
                if res.fun < local_best_score:
                    local_best_score = res.fun
                    local_best_params = res.x.copy()
            except Exception:
                pass

        # Update global best
        if local_best_score < best_score:
            best_score = local_best_score
            best_params = local_best_params.copy()
            best_centers = centers.copy()

    # Fallback if no valid configuration found
    if best_centers is None:
        combo = combinations[0]
        centers = np.array([[0.0, 0.0]])
        r1 = np.sqrt(3) + eps
        for k in range(6):
            angle = np.deg2rad(30 + k * 60)
            centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        r2 = 3.0 + 2 * eps
        for angle_deg in combo:
            angle = np.deg2rad(angle_deg)
            centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])

        n_hex = len(centers)
        x0 = np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex)
        best_score = objective(x0, centers)
        best_params = x0
        best_centers = centers

    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]

    inner = []
    for i in range(len(best_centers)):
        inner.append([best_centers[i][0], best_centers[i][1], np.rad2deg(phis[i])])

    side = best_score + 1e-10  # Tiny margin for floating point safety

    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

if __name__ == "__main__":
    construct()
