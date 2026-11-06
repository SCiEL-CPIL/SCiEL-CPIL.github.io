"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def construct():
    """
    Find 8 points maximizing the product of pairwise distances normalized by diameter^28.
    Strategy:
    1. Define a set of diverse initial configurations (cages) to avoid the regular octagon local optimum.
    2. Use a penalty-based optimization to maximize log(product of distances) subject to diameter <= 2.
    3. Use BFGS with analytical gradients for speed and precision.
    4. Implement a multi-stage penalty increase to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = 8
    
    def get_diameter(pts):
        # pts is (8, 2)
        # Compute all pairwise distances
        # For efficiency, we can just compute max distance
        # Since n is small, O(n^2) is fine
        d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist > d:
                    d = dist
        return d

    def log_product_and_grad(pts):
        """
        Returns log(product of distances) and gradient.
        Objective: sum_{i<j} log(||p_i - p_j||)
        Gradient wrt p_k: sum_{j != k} (p_k - p_j) / ||p_k - p_j||^2
        """
        log_prod = 0.0
        grad = np.zeros_like(pts)
        
        for i in range(n):
            for j in range(i+1, n):
                diff = pts[i] - pts[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    # Handle collision
                    dist = 1e-6
                    grad[i] += diff / dist_sq
                    grad[j] -= diff / dist_sq
                else:
                    dist = np.sqrt(dist_sq)
                    log_prod += np.log(dist)
                    # Gradient of log(dist) wrt p_i is (p_i - p_j) / dist^2
                    # Gradient of log(dist) wrt p_j is (p_j - p_i) / dist^2 = -(p_i - p_j) / dist^2
                    grad[i] += diff / dist_sq
                    grad[j] -= diff / dist_sq
                    
        return log_prod, grad

    def objective_penalty(x, penalty_factor):
        """
        x is flat array of 16 coordinates.
        Objective to minimize: -log_product + penalty * max(0, diameter - 2)^2
        """
        pts = x.reshape(n, 2)
        log_prod, grad_log = log_product_and_grad(pts)
        
        d = get_diameter(pts)
        if d > 2.0:
            penalty = penalty_factor * (d - 2.0)**2
            # Gradient of penalty: 2 * penalty_factor * (d - 2) * grad_d
            # grad_d is tricky because d is max. We approximate by assuming the max pair is fixed locally
            # Or simpler: just use the penalty value and let the optimizer handle it roughly, 
            # or compute grad of the specific max pair.
            # For robustness in BFGS, we can just return the penalty value and zero gradient for the penalty part initially,
            # or better, compute gradient of the active constraint.
            # Let's find the pair with max distance
            max_dist = 0
            max_i, max_j = 0, 1
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(pts[i] - pts[j])
                    if dist > max_dist:
                        max_dist = dist
                        max_i, max_j = i, j
            
            diff = pts[max_i] - pts[max_j]
            dist = max_dist
            if dist > 1e-9:
                # grad of dist wrt p_i is diff/dist, wrt p_j is -diff/dist
                grad_penalty = np.zeros_like(pts)
                grad_penalty[max_i] += (diff / dist) * 2 * penalty_factor * (dist - 2.0)
                grad_penalty[max_j] -= (diff / dist) * 2 * penalty_factor * (dist - 2.0)
            else:
                grad_penalty = np.zeros_like(pts)
            
            total_grad = -grad_log + grad_penalty
            return -log_prod + penalty, total_grad.flatten()
        else:
            return -log_prod, (-grad_log).flatten()

    def refine(pts, penalty_start=1.0, penalty_end=1000.0, steps=50):
        """
        Multi-stage penalty optimization.
        """
        current_pts = pts.copy()
        current_score = -1e9
        
        # Calculate initial score
        def calc_score(p):
            d = get_diameter(p)
            if d == 0: return -1e9
            # Scale to diameter 2
            p_scaled = p * (2.0 / d)
            # Calculate log product
            lp, _ = log_product_and_grad(p_scaled)
            # Score is Delta / 8^8 = exp(lp) / 8^8
            # We want to maximize this.
            # log_score = lp - 8 * log(8)
            return lp - 8 * np.log(8)
        
        best_score = calc_score(current_pts)
        best_pts = current_pts.copy()
        
        for stage in range(steps):
            # Interpolate penalty
            t = stage / (steps - 1)
            # Exponential increase in penalty
            pen = penalty_start * (penalty_end / penalty_start) ** t
            
            x0 = current_pts.flatten()
            
            # Run BFGS
            try:
                res = minimize(
                    objective_penalty, 
                    x0, 
                    args=(pen,), 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                new_pts = res.x.reshape(n, 2)
                
                # Check validity
                if np.all(np.isfinite(new_pts)):
                    new_score = calc_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
                    current_pts = new_pts
            except Exception:
                pass
            
            if time.time() > deadline:
                break
                
        return best_pts, best_score

    # Define diverse initial configurations
    configs = []
    
    # 1. Regular Octagon (Baseline)
    configs.append(np.array([[np.cos(2*np.pi*k/8), np.sin(2*np.pi*k/8)] for k in range(8)]))
    
    # 2. Two Concentric Squares (inner radius r, outer 1.0)
    # We need to optimize r. Let's try a few values.
    for r in [0.3, 0.5, 0.7]:
        outer = np.array([[np.cos(2*np.pi*k/4), np.sin(2*np.pi*k/4)] for k in range(4)])
        inner = np.array([[r*np.cos(2*np.pi*k/4 + np.pi/4), r*np.sin(2*np.pi*k/4 + np.pi/4)] for k in range(4)])
        configs.append(np.vstack([outer, inner]))
        
    # 3. Square with midpoints of sides (closer to center)
    # Vertices of square, and points on edges
    s = 1.0
    sq = np.array([[-s/2, -s/2], [s/2, -s/2], [s/2, s/2], [-s/2, s/2]])
    mids = np.array([[0, -s/2], [s/2, 0], [0, s/2], [-s/2, 0]])
    configs.append(np.vstack([sq, mids]))
    
    # 4. Random Perturbations of Octagon
    np.random.seed(42)
    for _ in range(5):
        octa = np.array([[np.cos(2*np.pi*k/8), np.sin(2*np.pi*k/8)] for k in range(8)])
        noise = np.random.normal(0, 0.2, size=(8, 2))
        configs.append(octa + noise)
        
    # 5. "Butterfly" shape: Two triangles sharing a vertex? No, 8 points.
    # Maybe two squares, one rotated 45 deg, different radii.
    for r_inner in [0.5, 0.8]:
        p1 = np.array([[r_inner, 0], [-r_inner, 0], [0, r_inner], [0, -r_inner]]) # Diamond
        p2 = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) / np.sqrt(2) # Square
        configs.append(np.vstack([p1, p2]))

    # Run optimization on each config
    best_pts_global = configs[0]
    best_score_global = -1e9
    
    for i, cfg in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Normalize initial config to diameter 2
        d = get_diameter(cfg)
        if d > 1e-6:
            cfg = cfg * (2.0 / d)
            
        pts, score = refine(cfg)
        if score > best_score_global:
            best_score_global = score
            best_pts_global = pts
            
    # Final check: ensure diameter is exactly 2 for the returned points
    # The score calculation inside refine handles scaling, but the output must be valid points.
    # The task says "score is ... after scaling the set to diameter 2".
    # So we can return any valid set, the evaluator scales it.
    # However, to be safe, let's return the points scaled to diameter 2.
    d_final = get_diameter(best_pts_global)
    if d_final > 1e-6:
        best_pts_global = best_pts_global * (2.0 / d_final)
        
    # Convert to list of lists of floats
    return [[float(x), float(y)] for x, y in best_pts_global]
# EVOLVE-BLOCK-END
