"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def _hex_lattice_init(n):
    """Generate a hexagonal lattice initialization."""
    # Estimate spacing
    # Area of unit circle is pi. N circles of radius r.
    # r ~ sqrt(pi / (N * 2 * sqrt(3))) approx
    # Let's just use a grid and filter
    spacing = 0.25
    while True:
        k = int(1.0 / spacing) + 2
        points = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x**2 + y**2 <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            # Sort by radius to keep the most central ones
            radii = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(radii)
            return points[indices[:n]]
        spacing *= 0.95

def _soft_min_objective(x, beta):
    """
    x: flattened array of 2*N coordinates
    beta: temperature parameter for soft-min
    Returns: -soft_min(distances)
    """
    n = len(x) // 2
    c = x.reshape(n, 2)
    
    # Calculate pairwise distances
    # d_ij = ||c_i - c_j||
    # We want min(d_ij / 2, 1 - ||c_i||)
    
    # Vectorized pairwise distances
    # d[i, j] = ||c[i] - c[j]||
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    d = np.sqrt(d_sq)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    d = d[mask]
    d = d / 2.0 # radius contribution from pair
    
    # Boundary distances
    r = np.hypot(c[:, 0], c[:, 1])
    d_b = 1.0 - r
    
    # Combine all distances into one array
    all_d = np.concatenate([d, d_b])
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow/underflow, use log-sum-exp trick
    # max_d = np.max(all_d)
    # log_sum_exp = max_d + np.log(np.sum(np.exp(-beta * (all_d - max_d))))
    # soft_min = -log_sum_exp / beta
    
    # Simpler and stable enough for moderate beta:
    # If beta is large, exp(-beta * d) underflows for small d, which is good.
    # But if d is large, exp(-beta * d) is 0.
    # We need the min, so the smallest d dominates.
    # exp(-beta * min_d) is the largest term.
    
    # Let's use the stable formulation
    min_d = np.min(all_d)
    # Shift by min_d to keep exponents <= 0
    shifted = all_d - min_d
    # exp(-beta * shifted) is <= 1
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = min_d - log_sum / beta
    
    return -soft_min_val

def _gradient(x, beta):
    """
    Gradient of the soft-min objective.
    J = -soft_min(d)
    soft_min(d) = -1/beta * log(sum(exp(-beta * d_k)))
    J = 1/beta * log(sum(exp(-beta * d_k)))
    dJ/dx = 1/beta * 1/sum(exp(-beta * d_k)) * sum( exp(-beta * d_k) * (-beta * dd_k/dx) )
    dJ/dx = - sum( w_k * dd_k/dx ) where w_k = exp(-beta * d_k) / sum(exp(-beta * d_j))
    """
    n = len(x) // 2
    c = x.reshape(n, 2)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    d = np.sqrt(d_sq)
    mask = ~np.eye(n, dtype=bool)
    d = d[mask]
    d = d / 2.0
    
    # Boundary distances
    r = np.hypot(c[:, 0], c[:, 1])
    d_b = 1.0 - r
    
    all_d = np.concatenate([d, d_b])
    
    # Weights
    min_d = np.min(all_d)
    shifted = all_d - min_d
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Gradient of distances w.r.t x
    # For pairs: d_ij = 0.5 * ||c_i - c_j||
    # dd_ij / dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # dd_ij / dc_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
    
    grad = np.zeros((n, 2))
    
    # Process pairs
    # We need to map back from flattened d to pairs
    # This is inefficient. Let's re-calculate gradients directly in a loop or vectorized way.
    # For N=50, N^2 = 2500 pairs. A loop is fine.
    
    # Reset grad
    grad = np.zeros((n, 2))
    
    # Pairwise contributions
    for i in range(n):
        for j in range(i + 1, n):
            diff_ij = c[i] - c[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < 1e-12:
                continue
            # d_ij term in all_d is at index ?
            # We need the weight for this specific pair.
            # Re-calculate weight for this pair
            d_val = dist_ij / 2.0
            w = np.exp(-beta * (d_val - min_d)) / sum_exp
            
            # Gradient of d_val w.r.t c_i
            grad_c_i = 0.5 * diff_ij / dist_ij
            # Gradient of d_val w.r.t c_j
            grad_c_j = -0.5 * diff_ij / dist_ij
            
            grad[i] += -w * grad_c_i  # Note: J = -soft_min, so dJ/dx = - d(soft_min)/dx? 
            # Let's re-verify signs.
            # Objective to minimize: J = -soft_min(d)
            # soft_min(d) approx min(d). We want to maximize min(d), so minimize -min(d).
            # d(soft_min)/dx = - sum(w_k * dd_k/dx)
            # dJ/dx = - d(soft_min)/dx = sum(w_k * dd_k/dx)
            
            grad[i] += w * grad_c_i
            grad[j] += w * grad_c_j

    # Boundary contributions
    for i in range(n):
        r_i = np.linalg.norm(c[i])
        if r_i < 1e-12:
            continue
        d_val = 1.0 - r_i
        w = np.exp(-beta * (d_val - min_d)) / sum_exp
        
        # d(1 - r)/dc = - c / r
        grad_c_i = -c[i] / r_i
        
        grad[i] += w * grad_c_i

    return grad.flatten()

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    # Initial guess
    x0 = _hex_lattice_init(N).flatten()
    
    # Multi-stage optimization
    betas = [10, 50, 200, 1000, 5000]
    
    current_x = x0.copy()
    best_score = -np.inf
    best_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
            
        try:
            res = minimize(
                _soft_min_objective,
                current_x,
                args=(beta,),
                jac=_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            current_x = res.x
            
            # Evaluate true score
            c = current_x.reshape(N, 2)
            min_pair_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(c[i] - c[j])
                    if d < min_pair_dist:
                        min_pair_dist = d
            min_boundary_dist = np.min(1.0 - np.linalg.norm(c, axis=1))
            r = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if r > best_score:
                best_score = r
                best_x = current_x.copy()
                
        except Exception:
            pass
            
    # Final polish with high beta if time permits
    if time.time() < deadline - 2.0:
        try:
            res = minimize(
                _soft_min_objective,
                best_x,
                args=(10000,),
                jac=_gradient,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            c = res.x.reshape(N, 2)
            min_pair_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(c[i] - c[j])
                    if d < min_pair_dist:
                        min_pair_dist = d
            min_boundary_dist = np.min(1.0 - np.linalg.norm(c, axis=1))
            r = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if r > best_score:
                best_score = r
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
