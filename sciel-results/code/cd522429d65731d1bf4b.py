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
    1. Start from structured configurations (icosahedron + antipodal pairs,
       Fibonacci sphere, random).
    2. Use momentum-based gradient ascent on a smooth objective:
       f = log(d_min) - log(d_max)  (maximize ratio)
       with a "soft" version using all pairwise distances.
    3. Project onto unit sphere each step.
    4. Multi-start with many random inits and refine.
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
    
    def objective_and_grad(points, beta=20.0):
        """
        Use a soft objective: minimize -log(sum exp(-beta * d_ij)) for repulsion,
        but we want to maximize d_min. Instead, use:
        f = min_i d_i  approximated by soft-min, and we want to maximize it
        while keeping d_max small.
        
        Actually, simplest: maximize log(d_min) - log(d_max).
        Gradient is tricky. Let's use: for each pair, we want all distances
        to be large (for d_min) but the max to be small (for d_max).
        
        Better: use the "energy" E = sum_{i<j} 1/d_ij^p which is minimized
        when points are well-spread. But that doesn't directly maximize d_min/d_max.
        
        Let's use: maximize log(min d) - log(max d) via subgradient.
        For d_min: gradient points in direction that increases the min distance.
        For d_max: gradient points in direction that decreases the max distance.
        """
        pts = points
        dists = pdist(pts)
        d_min = dists.min()
        d_max = dists.max()
        
        if d_min < 1e-10 or d_max < 1e-10:
            return 0.0, np.zeros_like(pts)
        
        # We want to maximize d_min and minimize d_max.
        # Gradient of d_min w.r.t. points: for the pair achieving min,
        # increase their separation. Gradient of -d_max: for the pair
        # achieving max, decrease their separation.
        
        # Soft version: use all pairs with weights.
        # Let's compute gradient for a smooth objective:
        # f = (1/n_pairs) * sum log(d_ij)  -- this is the geometric mean log
        # But we specifically want d_min/d_max.
        
        # Practical approach: use repulsion that's strong at short range
        # and weak at long range, combined with attraction to keep d_max small.
        
        # Simpler: use the objective f = d_min - lambda * d_max for some lambda,
        # but that's not scale-invariant.
        
        # Best: maximize log(d_min/d_max) = log(d_min) - log(d_max)
        # Subgradient for log(d_min): for the minimizing pair (a,b),
        #   grad_a = (p_a - p_b) / d_min, grad_b = (p_b - p_a) / d_min
        # Subgradient for -log(d_max): for the maximizing pair (c,d),
        #   grad_c = -(p_c - p_d) / d_max, grad_d = -(p_d - p_c) / d_max
        
        # But using only the extreme pairs is noisy. Use a soft version:
        # weight each pair by exp(-beta * d_ij) for the "min" part
        # and exp(+beta * d_ij) for the "max" part (to penalize large d).
        
        # Actually, let's just use a well-tested approach:
        # Maximize sum_{i<j} log(d_ij) subject to points on sphere.
        # This tends to equalize distances, which is good for d_min/d_max.
        
        # Gradient of sum log(d_ij) w.r.t. p_i:
        # sum_{j != i} (p_i - p_j) / d_ij^2 * d_ij / d_ij = sum (p_i - p_j)/d_ij
        # Wait: d/dp_i log(||p_i - p_j||) = (p_i - p_j) / ||p_i - p_j||^2
        # So grad = sum_{j != i} (p_i - p_j) / d_ij^2
        
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-15:
                    diff = np.random.randn(3) * 1e-8
                    dist_sq = np.dot(diff, diff)
                grad[i] += diff / dist_sq
        
        # Project gradient to tangent space of sphere
        for i in range(n):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        val = d_min / d_max
        return val * val, grad
    
    def relax_logsum(points, n_steps=10000, lr_init=0.01, lr_decay=0.9999, momentum=0.9):
        """
        Momentum-based gradient ascent on sum log(d_ij), projected to sphere.
        """
        pts = points.copy()
        velocity = np.zeros_like(pts)
        rng = np.random.default_rng(12345)
        lr = lr_init
        
        for step in range(n_steps):
            if check_deadline():
                break
            
            # Compute gradient of sum log(d_ij)
            grad = np.zeros_like(pts)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    diff = pts[i] - pts[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-15:
                        diff = rng.standard_normal(3) * 1e-8
                        dist_sq = np.dot(diff, diff)
                    grad[i] += diff / dist_sq
            
            # Project to tangent
            for i in range(n):
                grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
            
            # Momentum update
            velocity = momentum * velocity + lr * grad
            pts += velocity
            
            # Project to sphere
            pts = normalize_sphere(pts)
            
            lr *= lr_decay
        
        return pts
    
    def relax_repulsion(points, n_steps=10000, power=15.0, lr=0.005, momentum=0.9):
        """
        Momentum-based gradient ascent on repulsive potential sum 1/d^power.
        """
        pts = points.copy()
        velocity = np.zeros_like(pts)
        rng = np.random.default_rng(777)
        
        for step in range(n_steps):
            if check_deadline():
                break
            
            grad = np.zeros_like(pts)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    diff = pts[i] - pts[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-15:
                        diff = rng.standard_normal(3) * 1e-8
                        dist_sq = np.dot(diff, diff)
                    dist = np.sqrt(dist_sq)
                    # grad of 1/d^power w.r.t. p_i = power * (p_i - p_j) / d^(power+2)
                    grad[i] += power * diff / (dist_sq * dist)
            
            # Project to tangent
            for i in range(n):
                grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
            
            # Momentum update
            velocity = momentum * velocity + lr * grad
            pts += velocity
            pts = normalize_sphere(pts)
        
        return pts
    
    def anneal(points, n_steps=5000, power=15.0, temp_init=0.03, temp_decay=0.9995, lr=0.005):
        """Simulated annealing with repulsive forces."""
        pts = points.copy()
        rng = np.random.default_rng(999)
        temp = temp_init
        
        for step in range(n_steps):
            if check_deadline():
                break
            
            # Compute repulsive gradient
            grad = np.zeros_like(pts)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    diff = pts[i] - pts[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-15:
                        diff = rng.standard_normal(3) * 1e-8
                        dist_sq = np.dot(diff, diff)
                    dist = np.sqrt(dist_sq)
                    grad[i] += power * diff / (dist_sq * dist)
            
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
    for seed in range(60):
        inits.append(random_sphere(n, seed))
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Quick relaxation from all inits
    for idx, init in enumerate(inits):
        if check_deadline():
            break
        
        # Quick repulsion relaxation
        relaxed = relax_repulsion(init, n_steps=2000, power=15.0, lr=0.005)
        s = score(relaxed)
        if s > best_score:
            best_score = s
            best_points = relaxed.copy()
        
        # Quick log-sum relaxation
        relaxed2 = relax_logsum(init, n_steps=2000, lr_init=0.01)
        s2 = score(relaxed2)
        if s2 > best_score:
            best_score = s2
            best_points = relaxed2.copy()
    
    # Phase 2: Refine the best with longer runs
    if best_points is not None and not check_deadline():
        for power in [12.0, 15.0, 20.0]:
            if check_deadline():
                break
            refined = relax_repulsion(best_points, n_steps=5000, power=power, lr=0.003)
            s = score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
        
        # Log-sum refinement
        if not check_deadline():
            refined = relax_logsum(best_points, n_steps=5000, lr_init=0.005)
            s = score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
    
    # Phase 3: Annealing to escape local minima
    if best_points is not None and not check_deadline():
        for temp_init in [0.05, 0.03, 0.01]:
            if check_deadline():
                break
            annealed = anneal(best_points, n_steps=3000, power=15.0, temp_init=temp_init)
            s = score(annealed)
            if s > best_score:
                best_score = s
                best_points = annealed.copy()
            
            # Then refine after annealing
            if not check_deadline():
                refined = relax_repulsion(annealed, n_steps=3000, power=15.0, lr=0.003)
                s2 = score(refined)
                if s2 > best_score:
                    best_score = s2
                    best_points = refined.copy()
    
    # Phase 4: Final fine-tuning with very small steps
    if best_points is not None and not check_deadline():
        for lr in [0.001, 0.0005, 0.0002]:
            if check_deadline():
                break
            refined = relax_repulsion(best_points, n_steps=3000, power=20.0, lr=lr)
            s = score(refined)
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
