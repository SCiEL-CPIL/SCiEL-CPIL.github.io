import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_support_vec(alpha):
    alpha = np.mod(alpha, 2 * np.pi)
    ks = np.arange(6)
    vertex_angles = ks * np.pi / 3
    dots = np.cos(alpha[:, None] - vertex_angles[None, :])
    return np.max(dots, axis=1)

def build_dist_table():
    ds = np.linspace(0, np.pi/3, 361)
    min_dists = []
    for d in ds:
        vs = np.linspace(0, 2*np.pi, 720)
        dists = hex_support_vec(vs) + hex_support_vec(-vs + d)
        min_dists.append(np.min(dists))
    return ds, np.array(min_dists)

DS, MIN_DISTS = build_dist_table()

def get_min_dist(d):
    d = np.mod(d, np.pi/3)
    return np.interp(d, DS, MIN_DISTS)

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    v1 = np.array([np.sqrt(3), 0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    indices = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (0,2), (-2,0), (0,-2)
    ]
    
    centers = []
    angles = []
    for i, j in indices:
        c = i * v1 + j * v2
        centers.append(c)
        angles.append(0.0)
        
    centers = np.array(centers)
    angles = np.array(angles)
    x0 = np.concatenate([centers.ravel(), angles])
    
    def unpack(x):
        c = x[:22].reshape(11, 2)
        a = x[22:]
        return c, a

    def get_S_and_Phi(x):
        c, a = unpack(x)
        best_S = np.inf
        best_phi = 0.0
        phis = np.linspace(0, np.pi/3, 361)
        for phi in phis:
            nk_angles = phi + np.arange(6) * np.pi/3
            max_proj = 0
            for k in range(6):
                n_angle = nk_angles[k]
                n_vec = np.array([np.cos(n_angle), np.sin(n_angle)])
                dots = c @ n_vec
                rel_angles = n_angle - a
                radii = hex_support_vec(rel_angles)
                projs = dots + radii
                max_proj = max(max_proj, np.max(projs))
            S = (2 / np.sqrt(3)) * max_proj
            if S < best_S:
                best_S = S
                best_phi = phi
        return best_S, best_phi

    def objective_vec(x):
        S, _ = get_S_and_Phi(x)
        return S

    def penalized_objective(x, penalty_weight):
        base_obj, _ = get_S_and_Phi(x)
        c, a = unpack(x)
        penalty = 0
        for i in range(11):
            for j in range(i+1, 11):
                d_angle = a[i] - a[j]
                min_d = get_min_dist(d_angle)
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < min_d:
                    penalty += (min_d - dist) ** 2
        return base_obj + penalty_weight * penalty

    best_x = x0
    best_S, best_phi = get_S_and_Phi(x0)
    
    restarts = 5
    for r in range(restarts):
        if time.time() - start_time > time_limit:
            break
        jitter = np.random.normal(0, 0.1, x0.shape)
        x_start = x0 + jitter
        
        res = minimize(penalized_objective, x_start, args=(10.0,), method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        S_res, phi_res = get_S_and_Phi(res.x)
        
        # Check validity
        c, a = unpack(res.x)
        valid = True
        for i in range(11):
            for j in range(i+1, 11):
                d_angle = a[i] - a[j]
                min_d = get_min_dist(d_angle)
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < min_d - 1e-5:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and S_res < best_S:
            best_x = res.x
            best_S = S_res
            best_phi = phi_res

    if time.time() - start_time < time_limit - 20:
        def make_constraint(i, j):
            def cons(x):
                c, a = unpack(x)
                d_angle = a[i] - a[j]
                min_d = get_min_dist(d_angle)
                diff = c[i] - c[j]
                dist_sq = np.dot(diff, diff)
                return dist_sq - min_d**2
            return cons

        cons_list = []
        for i in range(11):
            for j in range(i+1, 11):
                cons_list.append({'type': 'ineq', 'fun': make_constraint(i, j)})
                
        res = minimize(objective_vec, best_x, method='SLSQP', constraints=cons_list,
                       options={'maxiter': 100, 'ftol': 1e-6})
        
        if res.success:
            S_res, phi_res = get_S_and_Phi(res.x)
            c, a = unpack(res.x)
            valid = True
            for i in range(11):
                for j in range(i+1, 11):
                    d_angle = a[i] - a[j]
                    min_d = get_min_dist(d_angle)
                    diff = c[i] - c[j]
                    dist = np.linalg.norm(diff)
                    if dist < min_d - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            if valid and S_res < best_S:
                best_x = res.x
                best_S = S_res
                best_phi = phi_res

    c, a = unpack(best_x)
    inner = []
    for i in range(11):
        inner.append([c[i][0], c[i][1], np.degrees(a[i])])
        
    return inner, (0.0, 0.0), best_S, np.degrees(best_phi)
