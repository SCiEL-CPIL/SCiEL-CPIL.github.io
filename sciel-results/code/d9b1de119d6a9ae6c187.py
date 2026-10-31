import numpy as np
import time
from scipy.optimize import minimize, basinhopping
from scipy.special import logsumexp

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _normalize(P):
    """
    Normalize points: center at origin, scale such that max pairwise distance is 1.
    This reduces scale invariance issues.
    """
    P = P - np.mean(P, axis=0)
    d_sq = _pairwise_dists_sq(P)
    max_d = np.sqrt(np.max(d_sq))
    if max_d < 1e-12:
        return P
    return P / max_d

def _smooth_log_ratio(P, p=50.0):
    """
    Smooth approximation of log(max_dist/min_dist).
    Minimizing this is equivalent to minimizing the ratio.
    p is the sharpness parameter. Higher p -> closer to true min/max.
    """
    P = _normalize(P)
    d_sq = _pairwise_dists_sq(P)
    
    # Avoid log(0)
    d_sq = np.maximum(d_sq, 1e-15)
    
    # log(D_max) approx = 1/p * logsumexp(p * d_sq)
    # Note: D_max is sqrt(max(d_sq)). So log(D_max) = 0.5 * log(max(d_sq)).
    # log(D_min) = 0.5 * log(min(d_sq)).
    # Ratio R = D_max / D_min. log(R) = 0.5 * (log(max(d_sq)) - log(min(d_sq))).
    
    # log(max(d_sq)) approx = 1/p * logsumexp(p * d_sq)
    # log(min(d_sq)) approx = -1/p * logsumexp(-p * d_sq)
    
    log_max_d_sq = logsumexp(p * d_sq) / p
    log_min_d_sq = -logsumexp(-p * d_sq) / p
    
    return 0.5 * (log_max_d_sq - log_min_d_sq)

def _local_objective(params):
    """
    Objective for local optimization (L-BFGS-B).
    """
    P = params.reshape(-1, 2)
    return _smooth_log_ratio(P, p=50.0)

def _local_minimize(x0):
    """
    Run L-BFGS-B on the smooth objective.
    """
    res = minimize(
        _local_objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    return res.x, res.fun

def _perturb(x, T):
    """
    Perturbation step for Basin Hopping.
    """
    rng = np.random.default_rng()
    return x + rng.normal(0, T, x.size)

def _accept_step(x_new, x_old, f_new, f_old, T):
    """
    Metropolis acceptance criterion for Basin Hopping.
    """
    if f_new < f_old:
        return True
    if T == 0:
        return False
    return np.random.default_rng().random() < np.exp(-(f_new - f_old) / T)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Initial guess: Hexagonal lattice subset
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 9.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    d_sqs = np.sum(hex_points**2, axis=1)
    sorted_indices = np.argsort(d_sqs)
    P_init = hex_points[sorted_indices][:16]
    
    x0 = P_init.flatten()
    
    best_score = _score(P_init)
    best_P = P_init
    
    # Run Basin Hopping
    # Basin Hopping parameters
    # We use a custom loop to control time and T cooling
    T = 0.5
    min_T = 0.01
    cooling_factor = 0.95
    
    x = x0.copy()
    f = _local_objective(x)
    
    # Initial local optimization
    x, f = _local_minimize(x)
    P_curr = x.reshape(-1, 2)
    score_curr = _score(P_curr)
    if score_curr < best_score:
        best_score = score_curr
        best_P = P_curr.copy()
        
    step = 0
    while time.time() < deadline - 10:
        # Perturb
        x_new = _perturb(x, T)
        
        # Local optimize from perturbed state
        x_opt, f_opt = _local_minimize(x_new)
        
        # Acceptance
        if _accept_step(x_opt, x, f_opt, f, T):
            x = x_opt
            f = f_opt
            P_curr = x.reshape(-1, 2)
            score_curr = _score(P_curr)
            if score_curr < best_score:
                best_score = score_curr
                best_P = P_curr.copy()
        else:
            # If rejected, we can still try to update best if the perturbed point 
            # happened to be good before optimization? No, optimization is better.
            pass
            
        # Cool down
        T *= cooling_factor
        if T < min_T:
            T = min_T
            
        step += 1
        
    # Final polish
    if time.time() < deadline - 2:
        x_final, _ = _local_minimize(best_P.flatten())
        P_final = x_final.reshape(-1, 2)
        score_final = _score(P_final)
        if score_final < best_score:
            best_score = score_final
            best_P = P_final

    return best_P

# EVOLVE-BLOCK-END
