import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for safety
    
    # 1. Initialization: Hex Lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            c = points[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98
    
    def get_score(c):
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        
        # Boundary distances
        boundary_dists = 1.0 - np.linalg.norm(c, axis=1)
        min_boundary_dist = np.min(boundary_dists)
        
        return min(min_pair_dist, min_boundary_dist)

    best_c = c.copy()
    best_score = get_score(best_c)
    
    # Precompute indices
    iu, ju = np.triu_indices(N, k=1)
    
    def make_objective(beta):
        def objective_and_grad(x_flat):
            c = x_flat.reshape(N, 2)
            
            # Compute pairwise distances
            diff = c[iu] - c[ju]
            dists = np.linalg.norm(diff, axis=1)
            
            # Boundary distances
            boundary_dists = 1.0 - np.linalg.norm(c, axis=1)
            
            # Combine all values
            all_vals = np.concatenate([dists / 2.0, boundary_dists])
            
            # Soft min parameters
            min_val = np.min(all_vals)
            exp_vals = np.exp(-beta * (all_vals - min_val))
            sum_exp = np.sum(exp_vals)
            
            # Objective value: minimize -soft_min
            obj = (1.0 / beta) * np.log(sum_exp)
            
            # Weights for gradient
            weights = exp_vals / sum_exp
            
            # Gradient w.r.t coordinates
            grad = np.zeros_like(c)
            
            # Boundary part
            b_weights = weights[N:]
            norms = np.linalg.norm(c, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            grad += b_weights[:, np.newaxis] * (-c / norms)
            
            # Pair part
            p_weights = weights[:len(iu)]
            diff_ij = c[iu] - c[ju]
            dist_ij = dists
            dist_ij[dist_ij == 0] = 1e-9
            unit_vec = diff_ij / dist_ij[:, np.newaxis]
            contrib = p_weights[:, np.newaxis] * unit_vec * 0.5
            
            np.add.at(grad, iu, contrib)
            np.add.at(grad, ju, -contrib)
            
            return obj, grad.flatten()
        return objective_and_grad

    # Stage 1: Coarse optimization
    try:
        res = minimize(
            make_objective(20.0), 
            c.flatten(), 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        c_new = res.x.reshape(N, 2)
        score_new = get_score(c_new)
        if score_new > best_score:
            best_score = score_new
            best_c = c_new
    except Exception:
        pass

    # Stage 2: Fine optimization (if time permits)
    if time.time() < deadline - 5:
        try:
            res2 = minimize(
                make_objective(50.0), 
                best_c.flatten(), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c_new2 = res2.x.reshape(N, 2)
            score_new2 = get_score(c_new2)
            if score_new2 > best_score:
                best_score = score_new2
                best_c = c_new2
        except Exception:
            pass

    return best_c
