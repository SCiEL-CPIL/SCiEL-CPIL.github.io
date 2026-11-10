# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Strategy:
    1. Use structured initial configurations (Fibonacci sphere, icosahedron-based).
    2. Vectorized gradient ascent on a soft objective that balances d_min and d_max.
    3. Project onto unit sphere each step.
    4. Multi-start with many random inits and refine.
    5. Use simulated annealing to escape local minima.
    """
    n = 14
    d = 3
    
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def normalize_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return points / norms
    
    def score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio
    
    def compute_soft_objective_grad(points, beta_min=50.0, beta_max=10.0):
        """
        Compute gradient of a soft objective that approximates log(d_min) - log(d_max).
        
        Uses softmax weighting:
        - For d_min: weight pairs by exp(-beta_min * d_ij), emphasizing close pairs
        - For d_max: weight pairs by exp(+beta_max * d_ij), emphasizing far pairs
        
        Returns (value, gradient) where value is the soft objective and gradient
        points in the direction to increase it.
        """
        pts = points
        n_pts = len(pts)
        
        # Compute all pairwise distances
        dists = pdist(pts)
        
        # Reshape to full matrix for easier indexing
        i_idx, j_idx = np.triu_indices(n_pts, k=1)
        dist_matrix = np.zeros((n_pts, n_pts))
        dist_matrix[i_idx, j_idx] = dists
        dist_matrix[j_idx, i_idx] = dists
        
        d_min = dists.min()
        d_max = dists.max()
        
        if d_min < 1e-10 or d_max < 1e-10:
            return 0.0, np.zeros_like(pts)
        
        # Soft weights for min and max
        w_min = np.exp(-beta_min * dists)
        w_min /= w_min.sum()
        
        w_max = np.exp(beta_max * dists)
        w_max /= w_max.sum()
        
        # Gradient of soft log(d_min): sum_i w_i * grad(log(d_i))
        # grad(log(||p_a - p_b||)) w.r.t. p_a = (p_a - p_b) / d^2
        # grad(log(||p_a - p_b||)) w.r.t. p_b = (p_b - p_a) / d^2
        
        grad = np.zeros_like(pts)
        
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            diff = pts[i] - pts[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-15:
                continue
            
            # Contribution to gradient from this pair
            # For soft min: weight w_min[k], gradient term is (p_i - p_j)/d^2 for point i
            # and (p_j - p_i)/d^2 for point j
            grad[i] += w_min[k] * diff / dist_sq
            grad[j] += w_min[k] * (-diff) / dist_sq
            
            # For soft max: we want to DECREASE d_max, so gradient is negative
            # of (p_i - p_j)/d^2 for point i
            grad[i] -= w_max[k] * diff / dist_sq
            grad[j] -= w_max[k] * (-diff) / dist_sq
        
        # Project gradient to tangent space of sphere
        for i in range(n_pts):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Value: approximate soft objective
        soft_log_min = -np.log(np.sum(w_min * np.exp(-dists)))  # This is wrong, let me fix
        # Actually, let's just return the actual ratio for tracking
        val = (d_min / d_max) ** 2
        
        return val, grad
    
    def relax_vectorized(points, n_steps=5000, lr_init=0.005, lr_decay=0.9998, momentum=0.9):
        """
        Momentum-based gradient ascent using vectorized operations.
        """
        pts = points.copy()
        velocity = np.zeros_like(pts)
        rng = np.random.default_rng(12345)
        lr = lr_init
        
        for step in range(n_steps):
            if check_deadline():
                break
            
            # Compute gradient
            val, grad = compute_soft_objective_grad(pts)
            
            # Momentum update
            velocity = momentum * velocity + lr * grad
            pts += velocity
            
            # Project to sphere
            pts = normalize_sphere(pts)
            
            lr *= lr_decay
        
        return pts
    
    def relax_repulsion_vectorized(points, n_steps=5000, power=15.0, lr=0.003, momentum=0.9):
        """
        Momentum-based gradient ascent on repulsive potential sum 1/d^power.
        Vectorized.
        """
        pts = points.copy()
        velocity = np.zeros_like(pts)
        rng = np.random.default_rng(777)
        
        for step in range(n_steps):
            if check_deadline():
                break
            
            # Compute all pairwise differences and distances
            # diff[i,j] = pts[i] - pts[j]
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            
            # Avoid division by zero
            dist_sq[dist_sq < 1e-15] = 1e-15
            
            # Gradient of sum_{i!=j} 1/||p_i - p_j||^power w.r.t. p_i
            # = power * sum_{j!=i} (p_i - p_j) / ||p_i - p_j||^(power+2)
            # = power * sum_{j!=i} (p_i - p_j) / (dist_sq^(power/2) * dist)
            # = power * sum_{j!=i} (p_i - p_j) / (dist_sq^((power+2)/2))
            
            # For each pair (i,j), contribution to grad[i] is power * (p_i - p_j) / dist^(power+2)
            # = power * diff[i,j] / (dist_sq[i,j]^((power+2)/2))
            
            exponent = (power + 2) / 2.0
            weight_matrix = np.zeros_like(dist_sq)
            np.power(dist_sq, -exponent, out=weight_matrix, where=dist_sq > 1e-15)
            
            # grad[i] = power * sum_j diff[i,j] * weight_matrix[i,j]
            grad = power * np.sum(diff * weight_matrix[:, :, np.newaxis], axis=1)
            
            # Project to tangent
            for i in range(n):
                grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
            
            # Momentum update
            velocity = momentum * velocity + lr * grad
            pts += velocity
            pts = normalize_sphere(pts)
        
        return pts
    
    def anneal_vectorized(points, n_steps=3000, power=15.0, temp_init=0.03, temp_decay=0.9995, lr=0.003):
        """Simulated annealing with repulsive forces, vectorized."""
        pts = points.copy()
        rng = np.random.default_rng(999)
        temp = temp_init
        
        for step in range(n_steps):
            if check_deadline():
                break
            
            # Compute repulsive gradient (same as relax_repulsion)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist_sq[dist_sq < 1e-15] = 1e-15
            
            exponent = (power + 2) / 2.0
            weight_matrix = np.zeros_like(dist_sq)
            np.power(dist_sq, -exponent, out=weight_matrix, where=dist_sq > 1e-15)
            
            grad = power * np.sum(diff * weight_matrix[:, :, np.newaxis], axis=1)
            
            for i in range(n):
                grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
            
            # Add noise
            noise = rng.standard_normal((n, d)) * temp
            pts += lr * grad + noise
            pts = normalize_sphere(pts)
            
            temp *= temp_decay
        
        return pts
    
    def fibonacci_sphere(n_points):
        points = []
        for i in range(n_points):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
            theta = np.pi * (1 + 5**0.5) * i
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)
    
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        return normalize_sphere(points)
    
    def icosahedron_plus_antipodal():
        """Icosahedron (12 vertices) + 2 antipodal points."""
        phi = (1 + 5**0.5) / 2
        icos = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icos = normalize_sphere(icos)
        # Add two points: try (1,1,1)/norm and (-1,-1,-1)/norm
        p1 = normalize_sphere(np.array([[1.0, 1.0, 1.0]]))[0]
        p2 = normalize_sphere(np.array([[-1.0, -1.0, -1.0]]))[0]
        return np.vstack([icos, [p1, p2]])
    
    def octahedron_plus_more():
        """Start from octahedron and add points."""
        octa = np.array([
            [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
        ], dtype=float)
        pts = list(octa)
        # Add 8 more points in a structured way
        extra_seeds = [42, 43, 44, 45, 46, 47, 48, 49]
        for s in extra_seeds:
            rng = np.random.default_rng(s)
            p = rng.standard_normal(3)
            pts.append(p / np.linalg.norm(p))
        return np.array(pts)
    
    # Generate initial configurations
    inits = []
    inits.append(fibonacci_sphere(n))
    inits.append(icosahedron_plus_antipodal())
    inits.append(octahedron_plus_more())
    
    # Many random starts
    for seed in range(40):
        inits.append(random_sphere(n, seed))
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Quick relaxation from all inits using vectorized methods
    for idx, init in enumerate(inits):
        if check_deadline():
            break
        
        # Quick repulsion relaxation (vectorized)
        relaxed = relax_repulsion_vectorized(init, n_steps=1500, power=15.0, lr=0.003)
        s = score(relaxed)
        if s > best_score:
            best_score = s
            best_points = relaxed.copy()
        
        # Soft objective relaxation
        if not check_deadline():
            relaxed2 = relax_vectorized(init, n_steps=1500, lr_init=0.005)
            s2 = score(relaxed2)
            if s2 > best_score:
                best_score = s2
                best_points = relaxed2.copy()
    
    # Phase 2: Refine the best with longer runs
    if best_points is not None and not check_deadline():
        for power in [12.0, 15.0, 20.0, 25.0]:
            if check_deadline():
                break
            refined = relax_repulsion_vectorized(best_points, n_steps=4000, power=power, lr=0.002)
            s = score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
        
        # Soft objective refinement
        if not check_deadline():
            refined = relax_vectorized(best_points, n_steps=4000, lr_init=0.003)
            s = score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
    
    # Phase 3: Annealing to escape local minima
    if best_points is not None and not check_deadline():
        for temp_init in [0.05, 0.03, 0.02, 0.01]:
            if check_deadline():
                break
            annealed = anneal_vectorized(best_points, n_steps=2500, power=15.0, temp_init=temp_init)
            s = score(annealed)
            if s > best_score:
                best_score = s
                best_points = annealed.copy()
            
            # Then refine after annealing
            if not check_deadline():
                refined = relax_repulsion_vectorized(annealed, n_steps=2500, power=15.0, lr=0.002)
                s2 = score(refined)
                if s2 > best_score:
                    best_score = s2
                    best_points = refined.copy()
    
    # Phase 4: Final fine-tuning with very small steps
    if best_points is not None and not check_deadline():
        for lr in [0.001, 0.0005, 0.0002]:
            if check_deadline():
                break
            refined = relax_repulsion_vectorized(best_points, n_steps=2500, power=20.0, lr=lr)
            s = score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
        
        # Also try soft objective fine-tuning
        if not check_deadline():
            refined = relax_vectorized(best_points, n_steps=2500, lr_init=0.001)
            s = score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
    
    # Phase 5: Multi-restart from best with perturbations
    if best_points is not None and not check_deadline():
        rng = np.random.default_rng(42)
        for trial in range(10):
            if check_deadline():
                break
            
            # Perturb the best points
            perturbed = best_points + rng.standard_normal((n, d)) * 0.1
            perturbed = normalize_sphere(perturbed)
            
            # Quick relax
            relaxed = relax_repulsion_vectorized(perturbed, n_steps=1000, power=15.0, lr=0.002)
            s = score(relaxed)
            if s > best_score:
                best_score = s
                best_points = relaxed.copy()
    
    # Fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Validate
    if not np.all(np.isfinite(best_points)) or best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
