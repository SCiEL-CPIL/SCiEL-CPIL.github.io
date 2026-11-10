import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 50

def compute_score(points):
    """
    Compute the score: min(min_pair_dist/2, min_wall_dist)
    """
    norms = np.linalg.norm(points, axis=1)
    min_wall = np.min(1.0 - norms)
    
    # Vectorized pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e9)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, min_wall)

def objective(params):
    """
    Objective function for minimization.
    Returns -score + penalty.
    """
    points = params.reshape(-1, 2)
    
    # Penalty for points outside the container
    norms = np.linalg.norm(points, axis=1)
    outside_penalty = np.sum(np.maximum(0.0, norms - 1.0)) * 10.0
    
    score = compute_score(points)
    return -score + outside_penalty

def construct():
    """
    Global optimization using Differential Evolution followed by local refinement.
    """
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_points = None
    best_score = -np.inf

    # 1. Differential Evolution (Global Search)
    try:
        res = differential_evolution(objective, bounds, 
                                     maxiter=150, 
                                     popsize=12, 
                                     tol=1e-5, 
                                     seed=42,
                                     polish=True,
                                     mutation=(0.5, 1.5),
                                     recombination=0.7)
        
        if res.x is not None:
            cand_pts = res.x.reshape(-1, 2)
            if np.all(np.isfinite(cand_pts)):
                cand_score = compute_score(cand_pts)
                # Check validity (inside container)
                if np.max(np.linalg.norm(cand_pts, axis=1)) < 1.0 + 1e-9:
                    if cand_score > best_score:
                        best_score = cand_score
                        best_points = cand_pts
    except Exception:
        pass

    # 2. Local Refinement (Polish)
    if best_points is not None:
        try:
            # Use Nelder-Mead for local polish as it handles non-smooth min well
            res2 = minimize(objective, best_points.flatten(), method='Nelder-Mead', 
                            options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9})
            cand_pts = res2.x.reshape(-1, 2)
            if np.all(np.isfinite(cand_pts)):
                cand_score = compute_score(cand_pts)
                if np.max(np.linalg.norm(cand_pts, axis=1)) < 1.0 + 1e-9:
                    if cand_score > best_score:
                        best_score = cand_score
                        best_points = cand_pts
        except Exception:
            pass

    # Fallback: Hexagonal grid if DE failed or returned invalid
    if best_points is None:
        points = []
        spacing = 0.22
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) < 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            best_points = points[idx]
        else:
            # Random fallback
            best_points = np.random.uniform(-0.9, 0.9, size=(N, 2))

    return best_points
