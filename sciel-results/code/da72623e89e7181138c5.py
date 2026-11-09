import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 590  # Safety margin before 600s kill

    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='euclidean')
        if np.any(dists == 0):
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_soft(x, beta):
        """
        Minimize negative log ratio of soft-min to soft-max.
        Uses logsumexp for numerical stability.
        """
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
            
        dists = pdist(p, metric='euclidean')
        
        # Avoid log(0) issues in soft_min if points coincide
        min_d = np.min(dists)
        if min_d < 1e-12:
            return 1e10 + 1.0 / min_d

        # soft_min = -1/beta * log(sum(exp(-beta * d)))
        # soft_max =  1/beta * log(sum(exp( beta * d)))
        
        # logsumexp(a) = log(sum(exp(a)))
        # So log(soft_min) = log(-1/beta) + log(sum(exp(-beta*d)))
        # We want to maximize log(soft_min) - log(soft_max)
        # Which is equivalent to minimizing:
        # log(soft_max) - log(soft_min)
        
        # log(soft_max) = log(1/beta) + logsumexp(beta * d)
        # log(soft_min) = log(-1/beta) + logsumexp(-beta * d)
        # Note: log(-1/beta) is undefined for positive beta? 
        # Wait, soft_min is a positive distance.
        # soft_min = (1/beta) * (sum(exp(-beta*d))) / exp(-beta*max_d) ... no, standard formula:
        # min(y) approx -1/beta * log(sum(exp(-beta*y)))
        # This value is positive if y are positive.
        
        # Let's just compute the values directly to be safe, though logsumexp is better.
        # log(soft_min) = log(1/beta) + logsumexp(-beta * dists) - log(n)? No, the 1/beta is outside.
        # soft_min = -1/beta * logsumexp(-beta * dists)
        # log(soft_min) = log(1/beta) + logsumexp(-beta * dists)
        
        # soft_max = 1/beta * logsumexp(beta * dists)
        # log(soft_max) = log(1/beta) + logsumexp(beta * dists)
        
        # Objective to minimize: log(soft_max) - log(soft_min)
        # = [log(1/beta) + logsumexp(beta * d)] - [log(1/beta) + logsumexp(-beta * d)]
        # = logsumexp(beta * d) - logsumexp(-beta * d)
        
        # This is a very clean formulation.
        try:
            log_sum_max = logsumexp(beta * dists)
            log_sum_min = logsumexp(-beta * dists)
        except:
            return 1e10

        return log_sum_max - log_sum_min

    # Initial configurations
    initializations = []
    
    # 1. Icosahedron + 2 poles
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
    poles = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    initializations.append(np.vstack([ico_verts, poles]))
    
    # 2. Icosahedron + 2 points slightly off poles
    offset = 0.1
    poles_off = np.array([[offset, 0, np.sqrt(1-offset**2)], [0, offset, -np.sqrt(1-offset**2)]])
    initializations.append(np.vstack([ico_verts, poles_off]))

    # 3. Fibonacci Sphere (from knowledge: effective for Tammes-like problems)
    # Fibonacci sphere gives a very uniform distribution
    i = np.arange(n)
    phi_fib = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi / (phi_fib * phi_fib)
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(1 - z**2)
    theta = golden_angle * i
    fib_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    initializations.append(fib_pts)

    # 4. Random points on sphere
    np.random.seed(42)
    for _ in range(10):
        rand_pts = np.random.randn(n, d)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1)[:, np.newaxis]
        initializations.append(rand_pts)

    best_points = initializations[0].copy()
    best_score = get_score(best_points)
    
    # Optimization loop
    # Strategy: 
    # 1. Coarse search with lower beta (smoother) to escape local minima
    # 2. Fine search with higher beta (sharper) to polish
    
    # We will iterate through initializations and apply a sequence of betas
    beta_schedule = [10, 50, 100, 200, 500]
    
    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > time_limit - 10:
            break
            
        current_pts = init_pts.copy()
        
        # Run optimization for each beta in the schedule
        for beta in beta_schedule:
            if time.time() - start_time > time_limit - 5:
                break
                
            try:
                # Use L-BFGS-B for speed and gradient-based efficiency
                # Note: objective_soft is differentiable
                res = minimize(objective_soft, current_pts.ravel(), method='L-BFGS-B', 
                               args=(beta,),
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e9:
                    opt_points = res.x.reshape((n, d))
                    if np.all(np.isfinite(opt_points)):
                        # Normalize scale to prevent numerical issues, though ratio is invariant
                        # Scaling to unit diameter helps with gradient magnitudes
                        dists_opt = pdist(opt_points)
                        max_d = np.max(dists_opt)
                        if max_d > 0:
                            opt_points = opt_points / max_d
                            
                        opt_score = get_score(opt_points)
                        if opt_score > best_score:
                            best_score = opt_score
                            best_points = opt_points.copy()
                        # Update current points for the next (higher) beta
                        current_pts = opt_points.copy()
                        
            except Exception:
                # If L-BFGS fails, try Powell as a fallback? 
                # Powell is slower but more robust for non-smoothness if gradients are bad.
                # But logsumexp is smooth. L-BFGS should be fine.
                pass

    # Final safety check
    if not np.all(np.isfinite(best_points)):
        best_points = initializations[0].copy()
        
    return best_points

# EVOLVE-BLOCK-END
