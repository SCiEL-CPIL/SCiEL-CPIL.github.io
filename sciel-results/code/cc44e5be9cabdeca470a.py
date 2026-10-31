"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # Spectral regularizer: 
    # We want to encourage the points to be well-distributed.
    # The Laplacian of the graph where edges connect all pairs (complete graph) 
    # has eigenvalues related to the spread. 
    # A simpler proxy for "isotropy" and "no clustering" is to penalize the variance 
    # of the squared distances from the centroid if they are too uneven, 
    # or use the trace of the inverse of the covariance matrix (if full rank) 
    # to encourage the cloud to be spherical (isotropic).
    # Let's use the condition number of the covariance matrix as a penalty.
    # If the points collapse to a line or point, eigenvalues go to 0.
    # We want to maximize the product of eigenvalues (determinant) or minimize condition number.
    # Since we are minimizing, we can add a term: -log(det(Cov)) or (1/min_eig).
    # Let's use (1 / min_eig) * small_weight to prevent degeneracy, but mainly rely on constraints.
    
    def spectral_penalty(pts):
        # Covariance matrix
        cov = np.cov(pts.T)
        # Add small ridge for stability
        cov += 1e-6 * np.eye(2)
        eigvals = np.linalg.eigvalsh(cov)
        min_eig = np.min(eigvals)
        if min_eig < 1e-6:
            return 1e6
        # Penalize anisotropy: ratio of max to min eigenvalue
        # We want this ratio to be close to 1 (circle) or at least not huge (line)
        # However, the optimal solution for max/min distance ratio might not be a perfect circle.
        # The 16-point optimal configuration is likely a perturbed grid or a specific packing.
        # Let's just penalize collapse (small min eigenvalue).
        return 1.0 / min_eig

    # Objective for SLSQP:
    # We normalize the points such that the MIN distance is exactly 1.
    # Then we want to MINIMIZE the MAX distance.
    # If we can force min_dist = 1, then ratio = max_dist / 1 = max_dist.
    # Score = max_dist^2.
    # Constraints:
    # 1. min_dist >= 1 - epsilon (we want it to be at least 1, but due to numerical issues, 
    #    we might optimize a scaled version)
    # Actually, SLSQP with inequality constraints d_ij >= 1 for all i<j is a classic packing problem.
    # If we find a configuration where all d_ij >= 1, then the min_dist is >= 1.
    # To minimize the ratio, we should also constrain the max_dist to be as small as possible.
    # This is a bi-criteria problem. 
    # Standard approach: Fix min_dist = 1 (by scaling), minimize max_dist.
    # But scaling changes the geometry. 
    # Better: Minimize max_dist subject to min_dist >= 1.
    # If the optimal packing has min_dist = m and max_dist = M, we can scale it by 1/m 
    # so that min_dist becomes 1 and max_dist becomes M/m. 
    # Then we minimize this new max_dist.
    
    def objective(x):
        pts = x.reshape(N, 2)
        d_sq = get_sq_dists(pts)
        d = np.sqrt(d_sq)
        # We want to minimize the maximum distance
        # Add a small penalty for spectral degeneracy to help convergence
        # The spectral penalty is scaled down because the main goal is the distance ratio
        return np.max(d) + 1e-4 * spectral_penalty(pts)

    def constraint_min_dist(x):
        pts = x.reshape(N, 2)
        d_sq = get_sq_dists(pts)
        d = np.sqrt(d_sq)
        # Constraint: d >= 1
        # SLSQP requires g(x) >= 0 for inequality constraints
        return d - 1.0

    # Gradient of objective: max(d) is non-differentiable at ties, but we can use a smooth approx 
    # or just let SLSQP handle it with finite differences if we don't provide jac.
    # Providing jac for max is tricky. Let's use finite differences for simplicity first, 
    # or provide a subgradient.
    # Actually, for SLSQP, finite differences are often robust enough for N=16 (32 vars).
    
    # Initializations
    inits = []
    
    # 1. Grid
    x = np.arange(4) - 1.5
    y = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x, y)
    base_grid = np.column_stack([xx.ravel(), yy.ravel()])
    # Normalize grid so min dist is approx 1? 
    # Grid min dist is 1. Max dist is sqrt(3^2+3^2) = sqrt(18) ~ 4.24.
    # So ratio is ~4.24. Score ~18.
    inits.append(base_grid.ravel())
    
    # 2. Circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Min dist on circle: 2*sin(pi/16) ~ 0.39. Max dist: 2.
    # Scale to make min dist 1: factor = 1/0.39 ~ 2.56.
    # New max dist ~ 5.12. Ratio ~ 5.12.
    inits.append((circle_pts / 0.39).ravel())
    
    # 3. Random
    for _ in range(10):
        pts = np.random.uniform(-1, 1, (N, 2))
        pts = normalize(pts)
        # Scale to make min dist approx 1
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        if d_min > 1e-6:
            pts = pts / d_min
        inits.append(pts.ravel())

    best_x = None
    best_score = 1e10
    
    for x0 in inits:
        if time.time() > deadline:
            break
            
        try:
            # Define constraints
            # We have 120 pairs. This is a lot of constraints for SLSQP.
            # SLSQP might struggle with 120 inequality constraints.
            # Alternative: Use a penalty method or reduce constraints.
            # Let's try SLSQP first. If it's too slow, we'll switch.
            
            cons = [
                {'type': 'ineq', 'fun': constraint_min_dist}
            ]
            
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            final_pts = res.x.reshape(N, 2)
            raw_score = score_raw(final_pts)
            
            if raw_score < best_score:
                best_score = raw_score
                best_x = res.x
        except Exception:
            continue

    if best_x is None:
        # Fallback
        best_x = inits[0]
        best_score = score_raw(best_x.reshape(N, 2))
        
    return best_x.reshape(N, 2)

# EVOLVE-BLOCK-END
