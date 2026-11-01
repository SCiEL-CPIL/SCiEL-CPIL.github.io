import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def objective_qcqp(x, upper_bound, penalty):
    """
    Objective for QCQP formulation:
    Maximize min pairwise sq dist, subject to all pairwise sq dist <= upper_bound.
    We minimize: -min_sq_dist + penalty * sum(max(0, sq_dist - upper_bound)^2)
    Plus a normalization term to fix scale.
    """
    points = x.reshape(N, D)
    
    # Center
    points = points - np.mean(points, axis=0, keepdims=True)
    
    # Normalize: fix the sum of squared distances to a constant to remove scale invariance
    # We want to maximize min dist while keeping max dist <= U.
    # To fix scale, let's fix the mean squared distance to 1.
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    points = points / np.sqrt(mean_sq_dist)
    
    sq_dists = get_pairwise_sq_dists(points)
    
    # Soft min of squared distances (we want to maximize this, so we minimize negative soft min)
    # Using log-sum-exp trick for min: min(x) = -log(sum(exp(-x)))/n ... wait.
    # min(x) approx = - (1/tau) * log(sum(exp(-tau * x)))
    # Let's use a simpler penalty for min: just take the actual min? No, not smooth.
    # Let's use soft min with tau.
    tau = 50.0
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    # This is the soft min. We want to maximize it, so we subtract it.
    obj = -log_sum_exp_min
    
    # Penalty for violating upper bound
    violations = np.maximum(0, sq_dists - upper_bound)
    penalty_term = penalty * np.sum(violations**2)
    
    return obj + penalty_term

def construct():
    start_time = time.time()
    timeout = 145.0
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Strategy: Binary search on the ratio R.
    # For a given R, check if there exist points such that max_dist^2 / min_dist^2 <= R.
    # This is equivalent to: max_dist^2 <= R * min_dist^2.
    # Let min_dist^2 = d_min. Then max_dist^2 <= R * d_min.
    # We can normalize d_min = 1. Then we need max_dist^2 <= R.
    # So we want to find points with min_dist^2 = 1 and max_dist^2 <= R.
    # If we can find such points, R is feasible.
    # We want the minimum feasible R.
    
    # Let's try to solve for a specific R using the QCQP objective.
    # Actually, the ratio R is the target.
    # Let's define the problem: Minimize max_dist^2 subject to min_dist^2 >= 1.
    # Or: Maximize min_dist^2 subject to max_dist^2 <= R.
    
    # Let's use the second one.
    # We will perform a binary search on R.
    # For a given R, we solve: Maximize min_dist^2 subject to max_dist^2 <= R.
    # If the optimal min_dist^2 > 1, then R is feasible (we can scale down to make min_dist^2 = 1, and max_dist^2 becomes R * (1/min_dist^2) < R).
    # Wait, if min_dist^2 > 1, scaling by 1/sqrt(min_dist^2) makes min_dist^2 = 1 and max_dist^2 = R/min_dist^2 < R.
    # So if we can find points with max_dist^2 <= R and min_dist^2 > 1, then R is feasible.
    # Actually, we just need min_dist^2 >= 1.
    
    # Let's reformulate:
    # Given R, find points such that all pairwise distances d_ij satisfy 1 <= d_ij^2 <= R.
    # This is a feasibility problem.
    # We can solve: Minimize (max(d_ij^2 - R)^2 + max(1 - d_ij^2)^2)
    # Or use the QCQP objective with upper_bound = R and target_min = 1.
    
    def objective_feasibility(x, R, penalty=100.0):
        points = x.reshape(N, D)
        points = points - np.mean(points, axis=0, keepdims=True)
        sq_dists = get_pairwise_sq_dists(points)
        
        # We want 1 <= sq_dists <= R
        # Penalty for sq_dists > R
        viol_max = np.maximum(0, sq_dists - R)
        # Penalty for sq_dists < 1
        viol_min = np.maximum(0, 1 - sq_dists)
        
        return penalty * (np.sum(viol_max**2) + np.sum(viol_min**2))
    
    # Binary search on R
    # Lower bound: 1 (all distances equal, impossible for 14 points in 3D)
    # Upper bound: 100 (generous)
    # We want the smallest R such that the problem is feasible.
    
    R_low = 1.0
    R_high = 100.0
    best_R = R_high
    best_points_for_R = None
    
    # We will try to find a feasible R.
    # For each R, we run a few optimizations.
    
    # Initial guess for R: let's start with a high R and try to lower it.
    # Or binary search.
    
    # Let's generate some initial points to get a baseline R.
    # Icosahedron + 2
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    p1 = rng.normal(size=D); p1 /= np.linalg.norm(p1)
    p2 = rng.normal(size=D); p2 /= np.linalg.norm(p2)
    pts_init = np.vstack([ico, p1, p2])
    R_init = raw_score(pts_init)
    R_high = R_init * 1.5
    R_low = 1.0
    
    # We will iterate.
    # For a given R, we try to find points with 1 <= d^2 <= R.
    # We use L-BFGS-B to minimize the penalty.
    
    def solve_for_R(R, x0, max_iter=200):
        res = minimize(
            objective_feasibility,
            x0,
            args=(R,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return res.x, res.fun
    
    # Binary search
    for _ in range(10):
        if time.time() - start_time > timeout - 10:
            break
        R_mid = (R_low + R_high) / 2
        feasible = False
        best_x_for_R = None
        best_fun_for_R = np.inf
        
        # Try multiple restarts for this R
        for _ in range(3):
            x0 = rng.uniform(-1, 1, size=(N, D))
            # Maybe start from a good point?
            if best_points is not None:
                x0 = best_points.flatten()
            else:
                x0 = pts_init.flatten()
            
            try:
                x_res, fun = solve_for_R(R_mid, x0)
                if fun < best_fun_for_R:
                    best_fun_for_R = fun
                    best_x_for_R = x_res
            except:
                pass
        
        # Check if feasible
        if best_x_for_R is not None:
            pts = best_x_for_R.reshape(N, D)
            sq_dists = get_pairwise_sq_dists(pts)
            if np.min(sq_dists) >= 0.99 and np.max(sq_dists) <= R_mid * 1.01:
                feasible = True
                # Update best score
                score = raw_score(pts)
                if score < best_score:
                    best_score = score
                    best_points = pts.copy()
        
        if feasible:
            R_high = R_mid
        else:
            R_low = R_mid
    
    # After binary search, R_high is the best R we found.
    # Let's do a final polish on the best points found.
    
    if best_points is not None:
        # Polish with the original objective
        def objective_smooth(x, tau=50.0):
            points = x.reshape(N, D)
            points = points - np.mean(points, axis=0, keepdims=True)
            sq_dists = get_pairwise_sq_dists(points)
            mean_sq_dist = np.mean(sq_dists)
            if mean_sq_dist < 1e-12:
                return 1e10
            points = points / np.sqrt(mean_sq_dist)
            sq_dists = get_pairwise_sq_dists(points)
            sq_dists = np.maximum(sq_dists, 1e-15)
            
            m_max = np.max(sq_dists)
            exp_max = np.exp(tau * (sq_dists - m_max))
            log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
            
            m_min = np.min(sq_dists)
            exp_min = np.exp(-tau * (sq_dists - m_min))
            log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
            
            return log_sum_exp_max / log_sum_exp_min
        
        try:
            res = minimize(
                objective_smooth,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            pts = res.x.reshape(N, D)
            update_best(pts)
        except:
            pass
    
    if best_points is None:
        best_points = pts_init
        best_score = raw_score(best_points)
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
