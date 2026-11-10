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
    
    Strategy (structural change from parent):
    1. Use dodecahedron-based initialization (20 vertices, select best 14 subset).
    2. Optimize via vectorized gradient ascent on a soft minimum-distance objective.
    3. Use repulsive force model: each point experiences force from all others.
    4. Simulated annealing with random restarts to escape local minima.
    5. No scipy.optimize; pure NumPy for speed and control.
    """
    n = 14
    d = 3
    
    start_time = time.time()
    time_limit = 620.0
    safety_margin = 15.0
    
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
    
    def dodecahedron_vertices():
        """Return 20 vertices of a regular dodecahedron on unit sphere."""
        phi = (1 + 5**0.5) / 2
        verts = np.array([
            # 8 cube-like vertices
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            # 12 pentagon vertices
            [0, 1/phi, phi], [0, -1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, -phi],
            [1/phi, phi, 0], [-1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, -phi, 0],
            [phi, 0, 1/phi], [-phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, -1/phi]
        ], dtype=float)
        return normalize_sphere(verts)
    
    def select_best_subset(verts, k):
        """Select k vertices from verts that maximize minimum pairwise distance."""
        m = len(verts)
        best_subset = None
        best_min_dist = -1
        
        # Greedy selection: start with pair that has max distance, then greedily add
        # that maximizes minimum distance to existing set
        i_idx, j_idx = np.triu_indices(m, k=1)
        diffs = verts[i_idx[:, None, :]] - verts[j_idx[:, None, :]]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        
        # Start with the pair having maximum distance
        max_idx = np.argmax(dists)
        i1 = i_idx[max_idx]
        j1 = j_idx[max_idx]
        subset = [i1, j1]
        
        # Greedily add remaining k-2 points
        for _ in range(k - 2):
            best_candidate = -1
            best_min_d = -1
            for c in range(m):
                if c in subset:
                    continue
                # Minimum distance from c to current subset
                min_d = min(np.linalg.norm(verts[c] - verts[s]) for s in subset)
                if min_d > best_min_d:
                    best_min_d = min_d
                    best_candidate = c
            subset.append(best_candidate)
        
        return verts[np.array(subset)]
    
    def soft_min_objective(pts, beta=5.0):
        """
        Soft minimum distance objective.
        Returns (value, gradient) where value = sum(exp(-beta / d_ij^2)) for all pairs.
        Higher value = better (points more separated).
        """
        # Compute pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, d)
        dist_sq = np.sum(diff**2, axis=2)  # (n, n)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Soft min: sum of exp(-beta / d^2)
        # For small d, exp(-beta/d^2) -> 0; for large d, exp(-beta/d^2) -> 1
        # So we want to MAXIMIZE this sum to push points apart
        inv_d2 = 1.0 / dist_sq
        inv_d2 = np.where(np.diag_indices(n), 0.0, inv_d2)
        
        obj = np.sum(np.exp(-beta * inv_d2))
        
        # Gradient: d/dp_i sum_j exp(-beta/d_ij^2)
        # = sum_j exp(-beta/d_ij^2) * (-beta) * d/dp_i (1/d_ij^2)
        # d/dp_i (1/d_ij^2) = -2*(p_i - p_j)/d_ij^4
        # So gradient contribution from pair (i,j) to p_i:
        #   exp(-beta/d_ij^2) * (-beta) * (-2*(p_i-p_j)/d_ij^4)
        # = 2*beta*exp(-beta/d_ij^2)*(p_i-p_j)/d_ij^4
        
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                d2 = dist_sq[i, j]
                if d2 < 1e-12:
                    continue
                val = np.exp(-beta / d2)
                diff_ij = pts[i] - pts[j]
                # Contribution to grad[i] and grad[j]
                factor = 2.0 * beta * val / (d2 * d2)
                grad[i] += factor * diff_ij
                grad[j] -= factor * diff_ij
        
        return obj, grad
    
    def optimize_repulsive(pts, max_iter=2000, beta_start=5.0, beta_end=20.0, lr_start=0.05, lr_end=0.005):
        """
        Gradient ascent on soft minimum distance with adaptive beta and learning rate.
        """
        pts = pts.copy()
        n_pts = len(pts)
        
        best_score = score(pts)
        best_pts = pts.copy()
        
        for it in range(max_iter):
            if check_deadline():
                break
            
            # Anneal beta and learning rate
            t = it / max(1, max_iter - 1)
            beta = beta_start + (beta_end - beta_start) * t
            lr = lr_start * (lr_end / lr_start) ** t
            
            obj, grad = soft_min_objective(pts, beta)
            
            # Project gradient to tangent space (remove radial component)
            for i in range(n_pts):
                radial = np.dot(grad[i], pts[i]) * pts[i]
                grad[i] -= radial
            
            # Gradient ascent step
            pts = pts + lr * grad
            
            # Project back to sphere
            pts = normalize_sphere(pts)
            
            # Track best
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        
        return best_pts, best_score
    
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        return normalize_sphere(points)
    
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
    
    # Generate initial configurations
    inits = []
    
    # Dodecahedron-based init
    dodec = dodecahedron_vertices()
    best_subset = select_best_subset(dodec, n)
    inits.append(best_subset)
    
    # Fibonacci sphere
    inits.append(fibonacci_sphere(n))
    
    # Random starts
    for seed in range(40):
        inits.append(random_sphere(n, seed))
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Optimize each initial configuration
    for idx, init in enumerate(inits):
        if check_deadline():
            break
        
        opt_pts, s = optimize_repulsive(init, max_iter=1500)
        if s > best_score:
            best_score = s
            best_points = opt_pts.copy()
    
    # Phase 2: Multi-restart from best with perturbations
    if best_points is not None and not check_deadline():
        rng = np.random.default_rng(123)
        for trial in range(20):
            if check_deadline():
                break
            
            # Perturb best solution
            noise_scale = 0.1 * (1 - trial / 20) + 0.01
            perturbed = best_points + rng.standard_normal((n, d)) * noise_scale
            perturbed = normalize_sphere(perturbed)
            
            opt_pts, s = optimize_repulsive(perturbed, max_iter=800)
            if s > best_score:
                best_score = s
                best_points = opt_pts.copy()
    
    # Phase 3: Final refinement with higher beta
    if best_points is not None and not check_deadline():
        refined, s = optimize_repulsive(best_points, max_iter=1000, beta_start=15.0, beta_end=40.0, lr_start=0.02, lr_end=0.002)
        if s > best_score:
            best_score = s
            best_points = refined.copy()
    
    # Fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Validate
    if not np.all(np.isfinite(best_points)) or best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
