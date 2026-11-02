import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def h_unit(phi):
        phi = phi % (2 * np.pi)
        angles = np.arange(6) * np.pi / 3
        return np.max(np.cos(phi - angles))

    outer_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))] for k in range(6)
    ])

    def compute_penalty(vars):
        S = vars[-1]
        if S <= 0:
            return 1e10
        
        centers = vars[:2*N].reshape(N, 2)
        thetas = vars[2*N:3*N]
        
        penalty = 0.0
        
        # 1. Containment Penalty
        for i in range(N):
            c_i = centers[i]
            t_i = thetas[i]
            for k in range(6):
                n = outer_normals[k]
                lhs = np.dot(c_i, n) + h_unit(np.arctan2(n[1], n[0]) - t_i)
                rhs = S * SQRT3 / 2.0
                if lhs > rhs:
                    penalty += (lhs - rhs) ** 2
        
        # 2. Non-Overlap Penalty (Smoothed SAT)
        for i in range(N):
            for j in range(i + 1, N):
                c_i = centers[i]
                c_j = centers[j]
                t_i = thetas[i]
                t_j = thetas[j]
                
                diff = c_i - c_j
                
                # Collect all potential separating axes (normals of both hexagons)
                # Normals for i
                normals_i = np.array([
                    [np.cos(t_i + np.radians(30 + 60*l)), np.sin(t_i + np.radians(30 + 60*l))] 
                    for l in range(6)
                ])
                # Normals for j
                normals_j = np.array([
                    [np.cos(t_j + np.radians(30 + 60*l)), np.sin(t_j + np.radians(30 + 60*l))] 
                    for l in range(6)
                ])
                
                all_normals = np.vstack([normals_i, normals_j])
                
                gaps = []
                for n in all_normals:
                    angle_n = np.arctan2(n[1], n[0])
                    h_i_val = h_unit(angle_n - t_i)
                    h_j_val = h_unit(angle_n - t_j)
                    
                    gap = np.dot(diff, n) - h_i_val - h_j_val
                    gaps.append(gap)
                
                gaps = np.array(gaps)
                # Smooth minimum: -logsumexp(-gaps / tau) * tau
                # This approximates min(gaps)
                tau = 0.1
                min_gap_smooth = -np.log(np.sum(np.exp(-gaps / tau))) * tau
                
                if min_gap_smooth < 0:
                    penalty += (min_gap_smooth) ** 2
        
        return penalty

    def objective(vars):
        if time.time() > deadline:
            return 1e10
        penalty = compute_penalty(vars)
        S = vars[-1]
        return S + 1000.0 * penalty

    # Initial Configuration: Triangular Lattice
    # Spacing slightly larger than minimum to ensure initial validity
    # Min spacing x: sqrt(3), y: 1.5
    spacing_x = SQRT3 * 1.1
    spacing_y = 1.5 * 1.1
    centers_init = []
    for r in range(3):
        for c in range(4):
            x = c * spacing_x
            y = r * spacing_y
            if r % 2 == 1:
                x += spacing_x / 2.0
            centers_init.append([x, y])
    centers_init = np.array(centers_init)
    centers_init -= centers_init.mean(axis=0)
    
    thetas_init = np.zeros(N)
    
    # Calculate initial S
    max_req = 0
    for i in range(N):
        c_i = centers_init[i]
        t_i = thetas_init[i]
        for k in range(6):
            n = outer_normals[k]
            lhs = np.dot(c_i, n) + h_unit(np.arctan2(n[1], n[0]) - t_i)
            req_s = lhs / (SQRT3 / 2.0)
            if req_s > max_req:
                max_req = req_s
                
    S_init = max_req * 1.05
    vars_init = np.concatenate([centers_init.flatten(), thetas_init, [S_init]])
    
    best_vars = vars_init.copy()
    best_score = S_init

    try:
        res = minimize(
            objective,
            vars_init,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        if res.fun < best_score:
            # Check validity
            p = compute_penalty(res.x)
            if p < 1e-6:
                best_score = res.x[-1]
                best_vars = res.x.copy()
    except Exception:
        pass

    centers, thetas, S = best_vars[:2*N].reshape(N, 2), best_vars[2*N:3*N], best_vars[-1]
    
    inner = []
    for i in range(N):
        inner.append([centers[i, 0], centers[i, 1], np.degrees(thetas[i])])
    
    return inner, (0.0, 0.0), S, 0.0
