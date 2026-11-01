import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 32

def sum_radii(c):
    p = c.reshape(N, 2)
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    return np.sum(radii)

def grad_sum_radii(c):
    p = c.reshape(N, 2)
    grad = np.zeros_like(p)
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1)
    min_neighbor_idx = np.argmin(dists, axis=1)
    
    for i in range(N):
        wall_r = min_wall_dist[i]
        neigh_r = min_neighbor_dist[i] / 2.0
        
        if wall_r < neigh_r - 1e-10:
            px, py = p[i]
            walls = [px, 1-px, py, 1-py]
            wall_grads = [[1,0], [-1,0], [0,1], [0,-1]]
            min_val = min(walls)
            active = [g for w, g in zip(walls, wall_grads) if abs(w - min_val) < 1e-10]
            if active:
                avg_grad = np.mean(active, axis=0)
                grad[i] += avg_grad
        elif neigh_r < wall_r - 1e-10:
            j = min_neighbor_idx[i]
            d = min_neighbor_dist[i]
            if d > 1e-10:
                vec = p[i] - p[j]
                grad[i] += 0.5 * vec / d
                grad[j] += -0.5 * vec / d
        else:
            px, py = p[i]
            walls = [px, 1-px, py, 1-py]
            wall_grads = [[1,0], [-1,0], [0,1], [0,-1]]
            min_val = min(walls)
            active_walls = [g for w, g in zip(walls, wall_grads) if abs(w - min_val) < 1e-10]
            wall_grad = np.mean(active_walls, axis=0) if active_walls else np.zeros(2)
            j = min_neighbor_idx[i]
            d = min_neighbor_dist[i]
            if d > 1e-10:
                vec = p[i] - p[j]
                neigh_grad_i = 0.5 * vec / d
                neigh_grad_j = -0.5 * vec / d
            else:
                neigh_grad_i = np.zeros(2)
                neigh_grad_j = np.zeros(2)
            grad[i] += 0.5 * (wall_grad + neigh_grad_i)
            grad[j] += 0.5 * neigh_grad_j
    return grad.flatten()

def construct():
    start_time = time.time()
    timeout = 115
    
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    
    # Global search using Differential Evolution
    # This is robust to the non-smooth min() in the objective
    try:
        res_global = differential_evolution(
            lambda c: -sum_radii(c), 
            bounds=bounds, 
            seed=42,
            maxiter=100, 
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.9,
            polish=False, # We will polish manually for better control
            init='sobol' # Better coverage than 'latinhypercube' or 'random'
        )
        best_c = res_global.x
        best_val = -res_global.fun
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to random start + LBFGS
        best_c = np.random.uniform(0.05, 0.95, size=2*N)
        best_val = sum_radii(best_c)

    # Local refinement using L-BFGS-B
    # We try a few different local refinements from the best global solution
    # and potentially from the top few DE solutions if we kept them (DE only returns best)
    
    # Polish 1: Standard LBFGS
    try:
        res_local = minimize(lambda c: -sum_radii(c), best_c, jac=lambda c: -grad_sum_radii(c),
                             method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
        val_local = -res_local.fun
        if val_local > best_val:
            best_val = val_local
            best_c = res_local.x.copy()
    except Exception as e:
        pass

    # Polish 2: Coordinate Descent (robust to non-smoothness, can escape LBFGS stalls)
    if time.time() - start_time < timeout - 5:
        try:
            p = best_c.reshape(N, 2).copy()
            # Simple coordinate descent: for each point, move it to maximize its radius
            # We use a simple grid search or Nelder-Mead for each point? 
            # Given time, let's do a simple "push apart" relaxation which is common in circle packing
            # Or just run LBFGS again with a different starting point?
            
            # Let's try a simple "relaxation" step:
            # If two circles overlap (r_i + r_j > dist), push them apart.
            # But r_i and r_j are coupled.
            
            # Better: Run LBFGS again but with a slightly perturbed start?
            # Or just accept the LBFGS result.
            
            # Let's try one more LBFGS pass with stricter tolerance or different params
            res_local2 = minimize(lambda c: -sum_radii(c), best_c, jac=lambda c: -grad_sum_radii(c),
                             method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
            val_local2 = -res_local2.fun
            if val_local2 > best_val:
                best_val = val_local2
                best_c = res_local2.x.copy()
        except Exception as e:
            pass

    if best_c is None:
        best_c = np.random.uniform(0.05, 0.95, size=2*N)
        
    p = best_c.reshape(N, 2)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    result = np.column_stack((p, radii))
    return result
