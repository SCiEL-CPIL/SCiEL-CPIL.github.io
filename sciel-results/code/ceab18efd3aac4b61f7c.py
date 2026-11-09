import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 150s limit
    
    best_points = None
    best_radius = np.inf
    
    # Helper to compute exact covering radius for scoring
    def compute_covering_radius(pts):
        try:
            # Check for degenerate cases (coincident points, etc.)
            if np.any(np.linalg.norm(pts, axis=1) < 1e-10):
                return np.inf
            
            # Check if all points are in a hemisphere (invalid for covering)
            # A simple heuristic: if the convex hull does not contain the origin, 
            # the points are in a hemisphere. 
            # However, for N=50, it's unlikely to be in a hemisphere if well distributed.
            # We rely on the hull computation.
            
            hull = ConvexHull(pts)
            max_rad = 0.0
            
            for simplex in hull.simplices:
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                normal = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                c_s = normal / norm
                
                # Calculate angular distances from circumcenter to vertices
                d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                
                max_d_circ = max(d1, d2, d3)
                
                # Check edges
                e12 = v1 + v2
                e12_norm = np.linalg.norm(e12)
                if e12_norm > 1e-10:
                    m12 = e12 / e12_norm
                    d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
                else:
                    d_e12 = 0.0
                
                e23 = v2 + v3
                e23_norm = np.linalg.norm(e23)
                if e23_norm > 1e-10:
                    m23 = e23 / e23_norm
                    d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                else:
                    d_e23 = 0.0

                e31 = v3 + v1
                e31_norm = np.linalg.norm(e31)
                if e31_norm > 1e-10:
                    m31 = e31 / e31_norm
                    d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                else:
                    d_e31 = 0.0

                max_d_edge = max(d_e12, d_e23, d_e31)

                # Determine local radius
                if max_d_circ <= np.pi / 2.0:
                    local_rad = max_d_circ
                else:
                    local_rad = max_d_edge

                if local_rad > max_rad:
                    max_rad = local_rad
            return max_rad
        except Exception:
            return np.inf

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return pts / norms

    def repulsive_energy(pts, sigma=0.5):
        """
        Computes a surrogate energy that approximates the covering radius.
        We use a sum of pairwise repulsive forces. Minimizing this tends to
        spread points evenly. The 'soft-max' or 'kernel' approach makes it differentiable.
        Here we use a simple inverse distance kernel which is effective for TSP-like spreading.
        """
        # Vectorized pairwise distances
        # pts shape (N, 3)
        # dist2[i,j] = ||p_i - p_j||^2
        dist2 = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2 * (pts @ pts.T)
        np.fill_diagonal(dist2, 1.0) # Avoid division by zero, doesn't matter as we exclude diag
        dist = np.sqrt(np.maximum(dist2, 1e-12))
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Energy: Sum of 1/d^k for all pairs. 
        # Using k=2 (Coulomb) or higher. 
        # To make it relate to covering radius, we want to maximize the min distance.
        # Minimizing sum(1/d^2) pushes close points apart.
        energy = np.sum(1.0 / dist2[mask])
        
        # Gradient computation
        # d/dp_i (1/||p_i - p_j||^2) = -2 (p_i - p_j) / ||p_i - p_j||^4
        # Sum over j != i
        # grad[i] = sum_{j!=i} -2 (p_i - p_j) / dist2[i,j]^2
        
        # dist2 is (N, N). We need to sum over j for each i.
        # term[i,j] = (p_i - p_j) / dist2[i,j]^2
        # Let's compute this vectorized.
        
        # dist4 = dist2^2
        dist4 = dist2**2
        dist4[~mask] = 1.0 # Avoid div by zero on diag
        
        # weights[i,j] = 1 / dist4[i,j]
        weights = 1.0 / dist4
        weights[~mask] = 0.0
        
        # grad[i] = -2 * sum_j (p_i - p_j) * weights[i,j]
        # = -2 * (p_i * sum_j weights[i,j] - sum_j p_j * weights[i,j])
        
        sum_weights = np.sum(weights, axis=1) # (N,)
        # sum_p_j_w = weights.T @ pts  -> (N, 3)
        sum_p_j_w = weights.T @ pts
        
        grad = -2.0 * (pts * sum_weights[:, None] - sum_p_j_w)
        
        return energy, grad

    def riemannian_momentum_descent(pts, lr=0.01, momentum=0.9, steps=2000):
        """
        Perform gradient descent on the sphere using Riemannian momentum.
        """
        pts = normalize(pts)
        velocity = np.zeros_like(pts)
        
        for step in range(steps):
            # Check time
            if time.time() - start_time > time_limit - 5.0:
                break
                
            energy, grad = repulsive_energy(pts)
            
            # Project gradient to tangent space: grad - (grad . p) p
            dots = np.sum(pts * grad, axis=1, keepdims=True)
            grad_tangent = grad - dots * pts
            
            # Update velocity with momentum
            velocity = momentum * velocity + lr * grad_tangent
            
            # Move point
            new_pts = pts + velocity
            new_pts = normalize(new_pts)
            
            # Optional: Check for improvement or just keep going
            pts = new_pts
            
        return pts

    # Phase 1: Multistart Optimization
    # We run multiple restarts with different initial configurations.
    # 1. Fibonacci sphere
    # 2. Random uniform points
    # 3. Points from a slightly perturbed Fibonacci sphere
    
    initializations = []
    
    # Init 1: Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib_pts)
    
    # Init 2: Random
    for _ in range(3):
        rand_pts = np.random.randn(N, 3)
        rand_pts = normalize(rand_pts)
        initializations.append(rand_pts)
        
    # Init 3: Perturbed Fibonacci
    perturbed = fib_pts + np.random.normal(0, 0.5, (N, 3))
    perturbed = normalize(perturbed)
    initializations.append(perturbed)

    # For each initialization, run the optimizer
    for init_pts in initializations:
        if time.time() - start_time > time_limit - 10.0:
            break
            
        # Run Riemannian Gradient Descent
        # We use a decaying learning rate schedule implicitly by just running fixed steps
        # or we can do multiple passes with decreasing LR.
        # Let's do a few passes with different LRs.
        
        current_pts = init_pts.copy()
        
        # Pass 1: Coarse
        current_pts = riemannian_momentum_descent(current_pts, lr=0.05, momentum=0.8, steps=1000)
        # Pass 2: Fine
        current_pts = riemannian_momentum_descent(current_pts, lr=0.01, momentum=0.9, steps=1000)
        # Pass 3: Ultra-fine
        current_pts = riemannian_momentum_descent(current_pts, lr=0.002, momentum=0.95, steps=1000)
        
        # Evaluate
        rad = compute_covering_radius(current_pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = current_pts.copy()
            
        # If we found a very good one, we can stop early or continue
        # Best known is ~18.3 deg = 0.319 rad
        if best_radius < 0.320:
             # We are close to the known best, try to squeeze more
             pass
             
    # Final safety check: if best_points is None, return Fibonacci
    if best_points is None:
        return fib_pts
        
    return best_points

# EVOLVE-BLOCK-END
