# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import random

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _fibonacci_sphere(n):
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

def _local_polish(points, time_limit=10.0):
    """
    Local polish using a smooth approximation of the objective.
    We minimize -log(d_min) + log(d_max) approximated by logsumexp.
    """
    n = 14
    start_time = time.time()
    current = points.copy()
    current_score = _score(current)
    
    # Try a few beta values for the logsumexp approximation
    betas = [0.5, 0.2, 0.1, 0.05]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        def obj_and_grad(x):
            pts = x.reshape(n, 3)
            # Compute pairwise squared distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, 0)
            
            D = dist_sq / beta
            
            # logsumexp(D) - logsumexp(-D)
            max_pos = np.max(D)
            if max_pos == 0:
                lse_pos = np.log(np.sum(np.exp(D)))
            else:
                lse_pos = max_pos + np.log(np.sum(np.exp(D - max_pos)))
                
            D_neg = -D
            max_neg = np.max(D_neg)
            if max_neg == 0:
                lse_neg = np.log(np.sum(np.exp(D_neg)))
            else:
                lse_neg = max_neg + np.log(np.sum(np.exp(D_neg - max_neg)))
                
            val = lse_pos - lse_neg
            
            # Gradient
            if max_pos == 0:
                w_pos = np.ones_like(D) / (n*(n-1))
            else:
                exp_pos = np.exp(D - max_pos)
                sum_exp_pos = np.sum(exp_pos)
                w_pos = exp_pos / sum_exp_pos
                
            if max_neg == 0:
                w_neg = np.ones_like(D) / (n*(n-1))
            else:
                exp_neg = np.exp(D_neg - max_neg)
                sum_exp_neg = np.sum(exp_neg)
                w_neg = exp_neg / sum_exp_neg
                
            W = w_pos + w_neg
            S = np.sum(W, axis=1)
            P = W @ pts
            grad_pts = 4.0 * (pts * S[:, np.newaxis] - P)
            grad = (1.0 / beta) * grad_pts.flatten()
            
            return val, grad

        x0 = current.flatten()
        res = minimize(
            obj_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.status == 0:
            new_pts = res.x.reshape(n, 3)
            # Center and normalize to prevent drift
            new_pts -= np.mean(new_pts, axis=0)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            if np.any(norms == 0):
                continue
            new_pts /= norms
            
            new_score = _score(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
                
    return current, current_score

def _simulated_annealing(points, time_limit=60.0):
    """
    Discrete Simulated Annealing with targeted local polish.
    Moves: random perturbation of one point, or swap of two points.
    """
    n = 14
    start_time = time.time()
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    T = 1.0
    T_min = 1e-6
    alpha = 0.995
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Generate a move
        move_type = random.random()
        candidate = current.copy()
        
        if move_type < 0.7:
            # Perturb one point
            idx = random.randint(0, n-1)
            noise = np.random.normal(0, T, 3)
            candidate[idx] += noise
        else:
            # Swap two points
            i, j = random.sample(range(n), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        
        # Ensure finite
        if not np.all(np.isfinite(candidate)):
            T *= alpha
            continue
            
        cand_score = _score(candidate)
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            # Accept with probability exp((cand_score - current_score) / T)
            delta = cand_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current = candidate
                    current_score = cand_score
        
        T *= alpha
        
        # Periodic local polish
        if random.random() < 0.05 and time.time() - start_time < time_limit - 10:
            polished, pol_score = _local_polish(current, time_limit=5.0)
            if pol_score > current_score:
                current = polished
                current_score = pol_score
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
                    
    return best, best_score

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    n = 14
    
    # Initial guess: Fibonacci sphere
    pts = _fibonacci_sphere(n)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts /= norms
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Run Simulated Annealing
    if time.time() < deadline - 20:
        sa_pts, sa_score = _simulated_annealing(pts, time_limit=deadline - time.time() - 20)
        if sa_score > best_score:
            best_pts = sa_pts
            best_score = sa_score
    
    # Final local polish on the best result
    if time.time() < deadline - 5:
        polished, pol_score = _local_polish(best_pts, time_limit=deadline - time.time() - 5)
        if pol_score > best_score:
            best_pts = polished
            best_score = pol_score
            
    return best_pts

# EVOLVE-BLOCK-END
