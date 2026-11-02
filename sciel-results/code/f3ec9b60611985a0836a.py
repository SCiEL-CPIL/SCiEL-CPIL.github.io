import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def _get_normals(angle_deg):
    normals = []
    for k in range(6):
        theta = math.radians(angle_deg + 30 + 60 * k)
        normals.append((math.cos(theta), math.sin(theta)))
    return normals

def solve_convex_subproblem(O_x, O_y, theta_deg, initial_centers, maxiter=50):
    n_vars = 2 * N + 1
    def objective(vars):
        return vars[-1]
        
    constraints = []
    normals = _get_normals(theta_deg)
    
    for i in range(N):
        for nx, ny in normals:
            idx_x = 2 * i
            idx_y = 2 * i + 1
            idx_a = 2 * N
            c = np.zeros(n_vars)
            c[idx_x] = nx
            c[idx_y] = ny
            c[idx_a] = -1.0
            b = -(nx * O_x + ny * O_y)
            def make_containment(c_vec, b_val):
                def fun(vars):
                    return -(np.dot(c_vec, vars) + b_val)
                return fun
            constraints.append({'type': 'ineq', 'fun': make_containment(c, b)})
            
    for i in range(N):
        for j in range(i + 1, N):
            idx_xi = 2 * i
            idx_yi = 2 * i + 1
            idx_xj = 2 * j
            idx_yj = 2 * j + 1
            def make_nonoverlap(xi, yi, xj, yj):
                def fun(vars):
                    dx = vars[xi] - vars[xj]
                    dy = vars[yi] - vars[yj]
                    return dx * dx + dy * dy - 3.0
                return fun
            constraints.append({'type': 'ineq', 'fun': make_nonoverlap(idx_xi, idx_yi, idx_xj, idx_yj)})
            
    bounds = []
    for i in range(N):
        bounds.append((None, None))
        bounds.append((None, None))
    bounds.append((0.5, None))
    
    x0 = np.zeros(n_vars)
    for i in range(N):
        x0[2*i] = initial_centers[i][0]
        x0[2*i+1] = initial_centers[i][1]
    max_dist = 0
    for c in initial_centers:
        d = math.hypot(c[0]-O_x, c[1]-O_y)
        if d > max_dist: max_dist = d
    x0[-1] = max_dist * 1.1
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': maxiter, 'ftol': 1e-5})
        if res.success:
            a_opt = res.x[-1]
            centers_opt = [(res.x[2*i], res.x[2*i+1]) for i in range(N)]
            return a_opt, centers_opt
        else:
            return None
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    sqrt3 = math.sqrt(3)
    
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
             
    # Try a few combinations of ring2
    # We'll just use the first 4 for simplicity, as the convex solver should move them if needed
    selected_r2 = ring2[:4]
    full_axial = pos_axial + selected_r2
    initial_centers = [axial_to_xy(q, r) for q, r in full_axial]
    
    cx_mean = sum(x for x, y in initial_centers) / N
    cy_mean = sum(y for x, y in initial_centers) / N
    
    best_side = float('inf')
    best_centers = None
    best_O = (0,0)
    best_theta = 0
    
    def objective_outer(params):
        if time.time() > deadline - 5.0:
            return 1e6
        O_x, O_y, theta = params
        res = solve_convex_subproblem(O_x, O_y, theta, initial_centers, maxiter=20)
        if res is None:
            return 1e6
        a, centers = res
        side = 2.0 * a / sqrt3
        return side
        
    x0_outer = [cx_mean, cy_mean, 0.0]
    
    try:
        res_outer = minimize(objective_outer, x0_outer, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-5})
        if res_outer.success:
            best_side = res_outer.fun
            O_x, O_y, theta = res_outer.x
            best_O = (O_x, O_y)
            best_theta = theta
            res_sub = solve_convex_subproblem(O_x, O_y, theta, initial_centers, maxiter=100)
            if res_sub:
                best_centers = res_sub[1]
    except Exception:
        pass
        
    if best_centers is None:
        # Fallback to fixed angle optimization
        for theta in [0.0, 30.0]:
            res = solve_convex_subproblem(cx_mean, cy_mean, theta, initial_centers, maxiter=100)
            if res:
                a, centers = res
                side = 2.0 * a / sqrt3
                if side < best_side:
                    best_side = side
                    best_centers = centers
                    best_O = (cx_mean, cy_mean)
                    best_theta = theta
                    
    if best_centers is None:
        # Absolute fallback
        best_centers = initial_centers
        cx = sum(x for x, y in best_centers) / N
        cy = sum(y for x, y in best_centers) / N
        # Estimate side
        max_d = 0
        for c in best_centers:
            d = math.hypot(c[0]-cx, c[1]-cy)
            if d > max_d: max_d = d
        best_side = max_d * 2.0 / sqrt3
        best_O = (cx, cy)
        best_theta = 0.0

    inner = [[c[0], c[1], 0.0] for c in best_centers]
    return inner, best_O, best_side, best_theta
