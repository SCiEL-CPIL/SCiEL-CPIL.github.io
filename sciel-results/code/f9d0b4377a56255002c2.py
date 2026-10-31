import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2
DEADLINE = 140.0

def _pairwise_dists(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    return d2[iu]

def _soft_objective(params, temp=0.01):
    pts = params.reshape(N, D)
    d2 = _pairwise_dists(pts)
    d2 = np.maximum(d2, 1e-12)
    m = np.max(d2)
    # Numerically stable log-sum-exp for soft max
    shifted = (d2 - m) / temp
    soft_max = m + temp * np.log(np.sum(np.exp(shifted)))
    m2 = np.min(d2)
    # Numerically stable log-sum-exp for soft min
    shifted_min = -(d2 - m2) / temp
    soft_min = m2 - temp * np.log(np.sum(np.exp(shifted_min)))
    return soft_max / soft_min

def _exact_score(pts):
    d2 = _pairwise_dists(pts)
    d2 = np.maximum(d2, 1e-15)
    return np.max(d2) / np.min(d2)

def _anneal_optimize(init_pts, t0, deadline):
    """Multi-stage annealing optimization with gradually decreasing temperature."""
    temps = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    current = init_pts.flatten()
    
    for temp in temps:
        if time.time() - t0 > deadline - 2:
            break
        try:
            res = minimize(
                lambda p, t=temp: _soft_objective(p, t),
                current,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            current = res.x
        except Exception:
            break
    
    return current.reshape(N, D)

def construct():
    t0 = time.time()
    
    # Initial candidates - more diverse starting points
    candidates = []
    
    # 1. Standard 4x4 Grid
    gx = np.array([0, 1, 2, 3], dtype=float)
    gy = np.array([0, 1, 2, 3], dtype=float)
    X, Y = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([X.flatten(), Y.flatten()])
    candidates.append(grid_pts)
    
    # 2. Compressed 4x4 Grid (corners moved in more aggressively)
    comp_pts = grid_pts.copy()
    center = np.array([1.5, 1.5])
    for idx in [0, 3, 12, 15]:
        comp_pts[idx] = comp_pts[idx] + 0.4 * (center - comp_pts[idx])
    candidates.append(comp_pts)
    
    # 3. Triangular Lattice
    tri_pts = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0.5, np.sqrt(3)/2], [1.5, np.sqrt(3)/2], [2.5, np.sqrt(3)/2], [3.5, np.sqrt(3)/2],
        [1, np.sqrt(3)], [2, np.sqrt(3)], [3, np.sqrt(3)], [4, np.sqrt(3)],
        [1.5, 1.5*np.sqrt(3)], [2.5, 1.5*np.sqrt(3)], [3.5, 1.5*np.sqrt(3)], [4.5, 1.5*np.sqrt(3)]
    ])
    candidates.append(tri_pts)
    
    # 4. Hexagonal-ish arrangement
    hex_pts = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0.5, 0.866], [1.5, 0.866], [2.5, 0.866], [3.5, 0.866],
        [0, 1.732], [1, 1.732], [2, 1.732], [3, 1.732],
        [0.5, 2.598], [1.5, 2.598], [2.5, 2.598], [3.5, 2.598]
    ])
    candidates.append(hex_pts)
    
    # 5-14. Random perturbations with varying scales
    rng = np.random.RandomState(42)
    for scale in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        perturbed = grid_pts + rng.randn(N, D) * scale
        candidates.append(perturbed)
    
    # 15-20. Random perturbations of triangular lattice
    for scale in [0.05, 0.1, 0.15, 0.2, 0.3]:
        perturbed = tri_pts + rng.randn(N, D) * scale
        candidates.append(perturbed)

    best_pts = None
    best_score = np.inf

    for i, init_pts in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 8:
            break
        
        try:
            # Use annealing optimization
            result_pts = _anneal_optimize(init_pts, t0, DEADLINE)
            score = _exact_score(result_pts)
            
            if score < best_score:
                best_score = score
                best_pts = result_pts
                
        except Exception:
            continue

    # Final high-precision polish on the best
    if best_pts is not None and time.time() - t0 < DEADLINE - 1:
        try:
            # Very fine annealing
            final_temps = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            current = best_pts.flatten()
            for temp in final_temps:
                if time.time() - t0 > DEADLINE - 0.5:
                    break
                res = minimize(
                    lambda p, t=temp: _soft_objective(p, t),
                    current,
                    method='L-BFGS-B',
                    options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                current = res.x
            
            pts_final = current.reshape(N, D)
            score_final = _exact_score(pts_final)
            if score_final < best_score:
                best_pts = pts_final
        except Exception:
            pass

    if best_pts is None:
        best_pts = grid_pts

    return best_pts

if __name__ == '__main__':
    pts = construct()
    print("Score:", _exact_score(pts))
