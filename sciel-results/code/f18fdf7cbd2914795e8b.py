# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    t0 = time.time()
    deadline = t0 + 585.0  # Safety margin before 630s limit

    def remaining():
        return deadline - time.time()

    def normalize(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return points / norms

    def compute_score(points):
        """Compute the actual metric: (d_min / d_max)^2"""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0 or d_min == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def fib_sphere(n_pts):
        phi = np.pi * (3.0 - np.sqrt(5.0))
        pts = []
        for i in range(n_pts):
            y = 1.0 - 2.0 * i / (n_pts - 1)
            r = np.sqrt(max(0.0, 1.0 - y * y))
            th = phi * i
            pts.append([np.cos(th) * r, y, np.sin(th) * r])
        return np.array(pts)

    def gradient_descent_sphere(points, iterations=2000, lr=0.008, momentum=0.9, k_power=4.0):
        """
        Minimizes sum(1/d_ij^k) on the sphere using gradient descent with momentum.
        This is a smooth proxy for maximizing d_min.
        """
        pts = points.copy()
        v = np.zeros_like(pts)
        
        for it in range(iterations):
            if remaining() < 5.0:
                break
                
            # Compute pairwise differences and distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff ** 2, axis=2)
            
            # Avoid division by zero
            np.fill_diagonal(dists_sq, 1e-12)
            
            # Gradient of E = sum_{i<j} d_ij^{-k}
            # dE/dp_i = -k * sum_{j != i} (p_i - p_j) * d_ij^{-k-2}
            
            # We want 1/d^(k+2) = 1/(d^2)^((k+2)/2)
            exp_term = (k_power + 2) / 2.0
            dists_inv = 1.0 / (dists_sq ** exp_term)
            
            # Set diagonal to 0
            np.fill_diagonal(dists_inv, 0.0)
            
            # terms[i, j] = (p_i - p_j) * dists_inv[i, j]
            # grad_i = -k * sum_j terms[i, j]
            # sum_j w_ij (p_i - p_j) = p_i * sum_j w_ij - sum_j w_ij p_j
            sum_w = np.sum(dists_inv, axis=1, keepdims=True)
            sum_wp = np.sum(dists_inv[:, :, np.newaxis] * pts, axis=1)
            grads = -k_power * (pts * sum_w - sum_wp)
            
            # Update velocity with momentum
            v = momentum * v - lr * grads
            
            # Update positions
            pts = pts + v
            
            # Project back to sphere
            pts = normalize(pts)
            
        return pts

    def nm_refine(points, maxfev=3000):
        """Refine using Nelder-Mead on the actual objective."""
        def objective(x):
            if remaining() < 2.0:
                raise TimeoutError()
            pts = x.reshape(n, d)
            pts = normalize(pts)
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max == 0 or d_min == 0:
                return 1e6
            return - (d_min / d_max) ** 2

        try:
            res = minimize(objective, points.flatten(), method='Nelder-Mead',
                           options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 2000, 'maxfev': maxfev,
                                    'adaptive': True})
            res_pts = normalize(res.x.reshape(n, d))
            return res_pts
        except Exception:
            return points

    def sa_refine(points, n_iter=5000, t0=0.05, t1=0.001):
        """
        Simulated Annealing refinement to escape local minima.
        Uses the true score as the objective.
        """
        pts = points.copy()
        current_score = compute_score(pts)
        best_pts = pts.copy()
        best_score = current_score
        rng = np.random.RandomState(42)
        
        for it in range(n_iter):
            if remaining() < 10.0:
                break
            
            # Temperature schedule (geometric cooling)
            t = t0 * (t1 / t0) ** (it / n_iter)
            
            # Propose a move: perturb one point slightly
            idx = rng.randint(n)
            step = 0.02 * t / t0
            new_pts = pts.copy()
            new_pts[idx] += rng.randn(d) * step
            new_pts = normalize(new_pts)
            
            new_score = compute_score(new_pts)
            delta = new_score - current_score
            
            if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-10)):
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
        
        return best_pts

    def scaled_nm_restart(points, scale=0.05, n_restarts=5):
        """Try NM restarts from perturbed versions of the best point."""
        best_pts = points.copy()
        best_score = compute_score(best_pts)
        
        rng = np.random.RandomState(123)
        for _ in range(n_restarts):
            if remaining() < 15.0:
                break
            
            perturbed = normalize(best_pts + rng.randn(n, d) * scale)
            refined = nm_refine(perturbed, maxfev=1000)
            s = compute_score(refined)
            
            if s > best_score:
                best_score = s
                best_pts = refined
        
        return best_pts

    best_points = None
    best_score = -1.0

    # Initializations - more diverse
    inits = []
    
    # 1. Fibonacci Sphere
    inits.append(fib_sphere(n))
    
    # 2. Random Sphere Points
    np.random.seed(42)
    for _ in range(12):
        pts = np.random.randn(n, d)
        inits.append(normalize(pts))
    
    # 3. Perturbed Fibonacci with varied scales
    for scale in [0.05, 0.1, 0.2, 0.3]:
        pts = fib_sphere(n) + np.random.randn(n, d) * scale
        inits.append(normalize(pts))
    
    # 4. Rotated Fibonacci
    fib = fib_sphere(n)
    for _ in range(8):
        R = Rotation.from_quat(np.random.randn(4)).as_matrix()
        pts = normalize(fib @ R.T)
        inits.append(pts)
    
    # 5. Rotated + Perturbed Fibonacci
    for _ in range(8):
        R = Rotation.from_quat(np.random.randn(4)).as_matrix()
        pts = normalize(fib @ R.T) + np.random.randn(n, d) * 0.1
        inits.append(normalize(pts))

    # Strategy:
    # 1. Run gradient descent on the smooth proxy for each init with multiple configs.
    # 2. Keep the best few results.
    # 3. Run Nelder-Mead on the exact objective for the best few.
    # 4. Try SA + NM restarts from the best.
    # 5. Return the overall best.

    candidates = []

    # Expanded parameter grid - focus on effective k and lr
    lr_values = [0.005, 0.008, 0.01, 0.015, 0.02, 0.03]
    k_values = [2.0, 4.0, 8.0, 12.0, 16.0]

    for init in inits:
        if remaining() < 90.0:
            break
            
        for lr in lr_values:
            if remaining() < 60.0:
                break
                
            for k in k_values:
                if remaining() < 30.0:
                    break
                    
                # Phase 1: Coarse GD
                pts = gradient_descent_sphere(init, iterations=1500, lr=lr, momentum=0.9, k_power=k)
                
                # Phase 2: Fine GD
                pts = gradient_descent_sphere(pts, iterations=1500, lr=lr*0.1, momentum=0.9, k_power=k)
                
                s = compute_score(pts)
                candidates.append((s, pts))
                
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 8 unique-ish candidates for NM refinement
    top_candidates = []
    seen = []
    for s, pts in candidates:
        if remaining() < 40.0:
            break
            
        # Simple check to avoid duplicate starting points for NM
        is_dup = False
        for _, prev_pts in top_candidates:
            if np.max(np.abs(pts - prev_pts)) < 1e-4:
                is_dup = True
                break
        
        if not is_dup:
            top_candidates.append((s, pts))
            
        if len(top_candidates) >= 8:
            break

    # Refine top candidates with Nelder-Mead
    for s, pts in top_candidates:
        if remaining() < 20.0:
            break
            
        refined_pts = nm_refine(pts)
        refined_score = compute_score(refined_pts)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts

    # Try SA refinement from the best found so far
    if best_points is not None and remaining() > 40.0:
        sa_result = sa_refine(best_points, n_iter=3000, t0=0.03, t1=0.0005)
        sa_score = compute_score(sa_result)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_result

    # Try scaled NM restarts from the best found so far
    if best_points is not None and remaining() > 20.0:
        restarted_pts = scaled_nm_restart(best_points, scale=0.05, n_restarts=5)
        restarted_score = compute_score(restarted_pts)
        
        if restarted_score > best_score:
            best_score = restarted_score
            best_points = restarted_pts

    # Final polish with SA if time remains
    if best_points is not None and remaining() > 30.0:
        sa_result = sa_refine(best_points, n_iter=2000, t0=0.02, t1=0.0003)
        sa_score = compute_score(sa_result)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_result

    # Fallback
    if best_points is None:
        best_points = fib_sphere(n)
        best_points = normalize(best_points)

    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = fib_sphere(n)
        best_points = normalize(best_points)

    return best_points

# EVOLVE-BLOCK-END
