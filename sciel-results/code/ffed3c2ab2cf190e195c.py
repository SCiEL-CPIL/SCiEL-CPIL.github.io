# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    Uses simulated annealing with smooth objective + gradient refinement.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin before 630s kill

    def elapsed():
        return time.time() - start_time

    def remaining():
        return deadline - time.time()

    def compute_score(pts):
        """Compute (d_min/d_max)^2 exactly."""
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max <= 0:
            return 0.0
        if d_min <= 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(params, beta):
        """
        Smooth approximation of -log((d_min/d_max)^2) using log-sum-exp.
        Lower beta = smoother, higher beta = closer to true objective.
        """
        pts = params.reshape(n, d)
        # Compute all pairwise squared distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff * diff, axis=-1)
        # Mask diagonal
        idx = np.arange(n)
        sq_dists[idx, idx] = np.inf
        
        # Soft max: log(sum(exp(beta * d^2))) / beta approximates max(d^2)
        # Soft min: -log(sum(exp(-beta * d^2))) / beta approximates min(d^2)
        flat_sq = sq_dists.flatten()
        finite_mask = np.isfinite(flat_sq)
        if not np.all(finite_mask):
            flat_sq = flat_sq[finite_mask]
        
        if len(flat_sq) == 0:
            return 1e10
        
        # Soft max of squared distances
        soft_max_sq = np.log(np.sum(np.exp(beta * flat_sq))) / beta
        # Soft min of squared distances  
        soft_min_sq = -np.log(np.sum(np.exp(-beta * flat_sq))) / beta
        
        if soft_min_sq <= 0 or soft_max_sq <= 0:
            return 1e10
        
        ratio_sq = soft_min_sq / soft_max_sq
        if ratio_sq <= 0:
            return 1e10
        
        return -np.log(ratio_sq)

    def smooth_gradient(params, beta):
        """Numerical gradient of smooth objective."""
        eps = 1e-7
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += eps
            f_plus = smooth_objective(p_plus, beta)
            p_minus = params.copy()
            p_minus[i] -= eps
            f_minus = smooth_objective(p_minus, beta)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def fibonacci_sphere(n_points, radius=1.0):
        """Generate points on a sphere using Fibonacci lattice."""
        points = np.zeros((n_points, 3))
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n_points):
            z = 1 - (2 * i + 1) / n_points
            r = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * i / golden_ratio
            points[i] = radius * np.array([r * np.cos(theta), r * np.sin(theta), z])
        return points

    def random_sphere(n_points, radius=1.0, seed=None):
        """Generate random points on a sphere."""
        rng = np.random.RandomState(seed)
        points = rng.randn(n_points, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = radius * points / norms
        return points

    def random_cloud(n_points, radius=1.0, seed=None):
        """Generate random points in a ball."""
        rng = np.random.RandomState(seed)
        points = rng.randn(n_points, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Scale to be within ball
        scales = rng.uniform(0.5, 1.0, size=(n_points, 1))
        points = radius * scales * points / norms
        return points

    def normalize_points(pts):
        """Normalize points to unit scale (center at origin, max norm = 1)."""
        pts = pts.copy()
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 1e-10:
            pts = pts / max_norm
        return pts

    def simulated_anneal(init_pts, max_iters=5000, t_start=1.0, t_end=0.01, seed=None):
        """Run simulated annealing on the smooth objective."""
        rng = np.random.RandomState(seed)
        pts = init_pts.copy()
        params = pts.flatten()
        
        # Start with moderate beta, anneal to high beta
        beta_start = 5.0
        beta_end = 50.0
        
        current_score = compute_score(pts)
        best_score = current_score
        best_params = params.copy()
        
        for it in range(max_iters):
            if remaining() < 30:
                break
            
            # Anneal temperature and beta
            progress = it / max_iters
            t = t_start * (t_end / t_start) ** progress
            beta = beta_start + (beta_end - beta_start) * progress
            
            # Generate perturbation - scale with temperature
            step_size = t * 0.1
            perturbation = rng.randn(n, d) * step_size
            new_params = params + perturbation
            new_pts = new_params.reshape(n, d)
            
            # Evaluate new point
            new_score = compute_score(new_pts)
            
            # Accept if better, or with probability exp(-delta/t)
            if new_score > current_score:
                params = new_params
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_params = params.copy()
            else:
                delta = current_score - new_score
                if delta > 0 and t > 0:
                    accept_prob = np.exp(-delta / (t + 1e-10))
                    if rng.random() < accept_prob:
                        params = new_params
                        pts = new_pts
                        current_score = new_score
        
        return best_params.reshape(n, d), best_score

    def gradient_refine(pts, max_iter=200, beta=20.0):
        """Refine using gradient descent on smooth objective."""
        params = pts.flatten()
        current_obj = smooth_objective(params, beta)
        
        lr = 0.01
        for it in range(max_iter):
            if remaining() < 20:
                break
            
            grad = smooth_gradient(params, beta)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < 1e-10:
                break
            
            # Adaptive step size
            step = lr / (1 + it * 0.01)
            new_params = params - step * grad
            new_obj = smooth_objective(new_params, beta)
            
            if new_obj < current_obj:
                params = new_params
                current_obj = new_obj
            else:
                lr *= 0.5
        
        return params.reshape(n, d)

    def coordinate_descent(pts, max_iter=100):
        """Coordinate descent refinement on exact objective."""
        pts = pts.copy()
        current_score = compute_score(pts)
        
        for it in range(max_iter):
            if remaining() < 15:
                break
            
            improved = False
            for i in range(n):
                for j in range(d):
                    # Try small moves in each coordinate
                    for delta in [0.01, 0.005, 0.001, -0.01, -0.005, -0.001]:
                        new_pts = pts.copy()
                        new_pts[i, j] += delta
                        new_score = compute_score(new_pts)
                        if new_score > current_score:
                            pts = new_pts
                            current_score = new_score
                            improved = True
                            break
            if not improved:
                break
        
        return pts, current_score

    # Generate diverse initializations
    initializations = []
    
    # Fibonacci sphere
    initializations.append(fibonacci_sphere(n))
    
    # Random spheres with different seeds
    for seed in [42, 123, 456, 789, 1011, 2020, 3030, 4040, 5050, 6060]:
        initializations.append(random_sphere(n, seed=seed))
    
    # Random clouds
    for seed in [77, 88, 99]:
        initializations.append(random_cloud(n, seed=seed))
    
    # Perturbed Fibonacci
    fib = fibonacci_sphere(n)
    for perturb_seed in [1, 2, 3, 4, 5]:
        rng = np.random.RandomState(perturb_seed)
        perturbed = fib + rng.randn(n, d) * 0.2
        initializations.append(perturbed)
    
    # Structured initializations - icosahedron-like
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    initializations.append(ico)
    
    # Add 2 more points to make 14
    rng = np.random.RandomState(123)
    extra = rng.randn(2, 3)
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    initializations.append(np.vstack([ico, extra]))
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Simulated annealing from each initialization
    for i, init_pts in enumerate(initializations):
        if remaining() < 60:
            break
        
        try:
            sa_pts, sa_score = simulated_anneal(
                init_pts, 
                max_iters=3000, 
                t_start=0.5, 
                t_end=0.01,
                seed=i * 100
            )
            
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_pts
        except Exception:
            continue
    
    # Phase 2: Gradient refinement on best solution
    if best_points is not None and remaining() > 40:
        try:
            refined_pts = gradient_refine(best_points, max_iter=100, beta=30.0)
            refined_score = compute_score(refined_pts)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts
        except Exception:
            pass
    
    # Phase 3: Coordinate descent refinement
    if best_points is not None and remaining() > 30:
        try:
            cd_pts, cd_score = coordinate_descent(best_points, max_iter=50)
            if cd_score > best_score:
                best_score = cd_score
                best_points = cd_pts
        except Exception:
            pass
    
    # Phase 4: More SA iterations with higher beta if time allows
    if best_points is not None and remaining() > 60:
        try:
            sa_pts2, sa_score2 = simulated_anneal(
                best_points,
                max_iters=2000,
                t_start=0.1,
                t_end=0.001,
                seed=9999
            )
            if sa_score2 > best_score:
                best_score = sa_score2
                best_points = sa_pts2
        except Exception:
            pass
    
    # Final coordinate descent
    if best_points is not None and remaining() > 20:
        try:
            cd_pts2, cd_score2 = coordinate_descent(best_points, max_iter=30)
            if cd_score2 > best_score:
                best_score = cd_score2
                best_points = cd_pts2
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Ensure finite values and correct shape
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
    
    if best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
