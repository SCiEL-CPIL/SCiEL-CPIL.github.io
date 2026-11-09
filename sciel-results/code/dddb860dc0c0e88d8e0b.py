import numpy as np
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    base_verts = np.array([[np.cos(np.pi/6 + k*np.pi/3), np.sin(np.pi/6 + k*np.pi/3)] for k in range(6)])
    
    def get_vertices(cx, cy, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return base_verts @ R.T + np.array([cx, cy])

    def calc_overlap_theta(theta1, theta2, verts1, verts2):
        n1 = np.array([[np.cos(theta1 + np.pi/6 + k*np.pi/3), np.sin(theta1 + np.pi/6 + k*np.pi/3)] for k in range(6)])
        n2 = np.array([[np.cos(theta2 + np.pi/6 + k*np.pi/3), np.sin(theta2 + np.pi/6 + k*np.pi/3)] for k in range(6)])
        min_sep = np.inf
        for n in np.vstack([n1, n2]):
            proj1 = np.max(verts1 @ n)
            proj2 = np.min(verts2 @ n)
            sep = proj2 - proj1
            if sep < min_sep:
                min_sep = sep
        return min_sep

    def calc_containment_margin(verts, L):
        n_c = np.array([[np.cos(np.pi/6 + k*np.pi/3), np.sin(np.pi/6 + k*np.pi/3)] for k in range(6)])
        d = L * np.sqrt(3) / 2
        margins = d - (verts @ n_c.T)
        return np.min(margins)

    def objective(x):
        return x[-1]

    def constraint_values(x):
        L = x[-1]
        positions = x[:-1].reshape(N, 3)
        verts_list = [get_vertices(p[0], p[1], p[2]) for p in positions]
        c_vals = []
        for i in range(N):
            for j in range(i+1, N):
                sep = calc_overlap_theta(positions[i][2], positions[j][2], verts_list[i], verts_list[j])
                c_vals.append(sep)
        for i in range(N):
            margin = calc_containment_margin(verts_list[i], L)
            c_vals.append(margin)
        return np.array(c_vals)

    def constraints_dict(x):
        return {'type': 'ineq', 'fun': constraint_values}

    def make_initial_guess(pattern, seed=0):
        np.random.seed(seed)
        x = np.zeros(N*3 + 1)
        if pattern == 0:
            x[0:3] = [0, 0, 0]
            for k in range(6):
                ang = k * np.pi / 3
                dist = np.sqrt(3)
                x[(k+1)*3:(k+2)*3] = [dist*np.cos(ang), dist*np.sin(ang), 0]
            for k, ang in enumerate([0, np.pi/2, np.pi, 3*np.pi/2]):
                dist = 2 * np.sqrt(3)
                idx = 7 + k
                x[idx*3:(idx+1)*3] = [dist*np.cos(ang), dist*np.sin(ang), 0]
            x[-1] = 3.5
        else:
            for i in range(N):
                x[i*3] = np.random.uniform(-2, 2)
                x[i*3+1] = np.random.uniform(-2, 2)
                x[i*3+2] = np.random.uniform(0, np.pi)
            x[-1] = 4.0
        return x

    best_L = np.inf
    best_x = None
    
    attempts = [(0, 0), (1, 1), (1, 2), (1, 3), (0, 1)]
    
    for pattern, seed in attempts:
        if time.time() > deadline:
            break
        x0 = make_initial_guess(pattern, seed)
        try:
            res = minimize(objective, x0, method='SLSQP', constraints=constraints_dict, options={'maxiter': 300, 'ftol': 1e-8})
            if res.fun < best_L:
                c_vals = constraint_values(res.x)
                if np.all(c_vals >= -1e-5):
                    best_L = res.fun
                    best_x = res.x
        except Exception as e:
            pass

    if best_x is None:
        return [[0,0,0]]*N, (0,0), 10.0, 0.0

    positions = best_x[:-1].reshape(N, 3)
    L = best_x[-1]
    inner = [[p[0], p[1], np.degrees(p[2])] for p in positions]
    return inner, (0.0, 0.0), L, 0.0
