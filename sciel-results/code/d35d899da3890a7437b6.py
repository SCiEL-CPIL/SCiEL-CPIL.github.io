import numpy as np
from scipy.optimize import minimize
import time

N = 11
START_TIME = time.time()
DEADLINE = 145.0  # 5s safety margin before 150s kill

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon as (vx_array, vy_array)."""
    a = np.radians(angle_deg)
    thetas = a + np.arange(6) * np.pi / 3
    return cx + side * np.cos(thetas), cy + side * np.sin(thetas)

def construct():
    # Initial hexagonal packing: 1 center, 6 around, 4 more
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    positions[0] = [0.0, 0.0]
    for k in range(6):
        theta = k * np.pi / 3
        positions[k + 1] = [2.0 * np.cos(theta), 2.0 * np.sin(theta)]
    for k in range(4):
        theta = np.pi / 6 + k * np.pi / 2
        positions[k + 7] = [3.5 * np.cos(theta), 3.5 * np.sin(theta)]

    x0 = np.concatenate([positions.flatten(), angles, [0.0, 0.0, 0.0, 5.5]])

    def objective(params):
        if time.time() > DEADLINE:
            return 1e10
        pos = params[:2 * N].reshape(N, 2)
        ang = params[2 * N:3 * N]
        ox, oy = params[3 * N], params[3 * N + 1]
        oang = params[3 * N + 2]
        os = params[3 * N + 3]

        # Compute all inner hex vertices
        all_vx = np.zeros((N, 6))
        all_vy = np.zeros((N, 6))
        for i in range(N):
            vx, vy = hex_vertices(pos[i, 0], pos[i, 1], ang[i])
            all_vx[i] = vx
            all_vy[i] = vy

        # Outer containment penalty
        a_outer = np.radians(oang)
        apothem = os * np.sqrt(3) / 2
        penalty = 0.0
        for i in range(N):
            for k in range(6):
                n_angle = a_outer + np.pi / 6 + k * np.pi / 3
                nx, ny = np.cos(n_angle), np.sin(n_angle)
                proj = nx * (all_vx[i] - ox) + ny * (all_vy[i] - oy)
                max_proj = np.max(proj)
                if max_proj > apothem:
                    penalty += (max_proj - apothem) ** 2

        # Overlap penalty using center distance (soft constraint)
        overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 1.5:
                    overlap += (1.5 - d) ** 2

        return os + 5.0 * penalty + 50.0 * overlap

    bounds = [(-10, 10)] * 2 * N + [(-180, 180)] * N + [(-5, 5), (-5, 5), (-180, 180), (1, 20)]

    best_x = x0.copy()
    best_f = objective(x0)

    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-7, 'gtol': 1e-5})
        if res.fun < best_f:
            best_x = res.x
            best_f = res.fun
    except:
        pass

    pos_best = best_x[:2 * N].reshape(N, 2)
    ang_best = best_x[2 * N:3 * N]

    # Find minimum enclosing outer hexagon for the optimized inner configuration
    def min_enclosing(pos_arr, ang_arr, oang_deg):
        a = np.radians(oang_deg)
        apothem_factor = np.sqrt(3) / 2

        all_vx = np.zeros((N, 6))
        all_vy = np.zeros((N, 6))
        for i in range(N):
            vx, vy = hex_vertices(pos_arr[i, 0], pos_arr[i, 1], ang_arr[i])
            all_vx[i] = vx
            all_vy[i] = vy

        flat_vx = all_vx.flatten()
        flat_vy = all_vy.flatten()

        def outer_obj(o):
            ox, oy = o
            max_val = 0.0
            for k in range(6):
                n_angle = a + np.pi / 6 + k * np.pi / 3
                nx, ny = np.cos(n_angle), np.sin(n_angle)
                proj = nx * (flat_vx - ox) + ny * (flat_vy - oy)
                mv = np.max(proj)
                if mv > max_val:
                    max_val = mv
            return max_val / apothem_factor

        res = minimize(outer_obj, [0.0, 0.0], method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
        return res.fun, res.x

    best_os = float('inf')
    best_ox, best_oy = 0.0, 0.0
    best_oang = 0.0

    for oang_test in np.linspace(0, 60, 7):
        os_val, (ox_val, oy_val) = min_enclosing(pos_best, ang_best, oang_test)
        if os_val < best_os:
            best_os = os_val
            best_ox, best_oy = ox_val, oy_val
            best_oang = oang_test

    inner = [[pos_best[i, 0], pos_best[i, 1], ang_best[i]] for i in range(N)]
    return inner, (best_ox, best_oy), best_os, best_oang
