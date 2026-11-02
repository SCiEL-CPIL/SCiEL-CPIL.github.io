import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def generate_hex_lattice():
        spacing = 0.11
        points = []
        k = 15
        for j in range(-k, k+1):
            for i in range(-k, k+1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 0.95**2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
             spacing *= 0.9
             return generate_hex_lattice()
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
        return points[idx[:N]]

    inits = []
    p_hex = generate_hex_lattice()
    inits.append(p_hex)
    
    angle = np.pi / 6
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    p_rot = p_hex @ rot.T
    inits.append(p_rot)
    
    np.random.seed(42)
    for _ in range(3):
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        inits.append(np.column_stack((x, y)))

    best_score = -1
    best_points = None

    def get_score(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e10)
        min_dist = dists.min() / 2.0
        radii = np.linalg.norm(p, axis=1)
        min_bound = (1.0 - radii).min()
        return min(min_dist, min_bound)

    def optimize(points, max_time):
        current_p = points.copy()
        best_local_score = get_score(current_p)
        best_local_p = current_p.copy()
        
        betas = [10, 50, 200, 1000, 5000]
        
        for beta in betas:
            if time.time() > max_time:
                break
                
            def loss_and_grad(p_flat):
                p = p_flat.reshape(-1, 2)
                diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, 1e-10)
                
                radii = np.linalg.norm(p, axis=1)
                b_dists = 1.0 - radii
                b_dists = np.maximum(b_dists, 1e-10)
                
                w_ij = np.exp(-beta * dists)
                np.fill_diagonal(w_ij, 0)
                
                grad_d = diff / dists[:, :, np.newaxis]
                
                grad_S_pairs = -beta * (w_ij * grad_d).sum(axis=1)
                
                w_b = np.exp(-beta * b_dists)
                grad_b = p / radii[:, np.newaxis]
                grad_S_bound = beta * (w_b[:, np.newaxis] * grad_b)
                
                grad_S = grad_S_pairs + grad_S_bound
                S = w_ij.sum() + w_b.sum()
                
                grad_L = -1.0 / (beta * S) * grad_S
                L = -1.0 / beta * np.log(S)
                
                return L, grad_L.reshape(-1)

            # Use L-BFGS-B
            res = minimize(loss_and_grad, current_p.flatten(), method='L-BFGS-B', 
                           jac=True, options={'maxiter': 100, 'gtol': 1e-5})
            
            current_p = res.x.reshape(-1, 2)
            score = get_score(current_p)
            if score > best_local_score:
                best_local_score = score
                best_local_p = current_p.copy()

        return best_local_p, best_local_score

    for init in inits:
        if time.time() > deadline:
            break
        p, score = optimize(init, deadline)
        if score > best_score:
            best_score = score
            best_points = p

    if best_points is not None and time.time() < deadline:
        p_final, s_final = optimize(best_points, deadline)
        if s_final > best_score:
            best_score = s_final
            best_points = p_final
            
    return best_points
