"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import random


def construct():
    # Simulated annealing approach to minimize logarithmic energy on the unit sphere
    
    # Initialize with golden-angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure points are on unit sphere
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    
    def compute_energy(points):
        """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|)"""
        n = len(points)
        energy = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d < 1e-15:
                    return 1e10  # Very high penalty for coincident points
                energy += math.log(1.0 / d)
        return energy
    
    def random_point():
        """Generate a random point on the unit sphere"""
        # Use Gaussian distribution normalized
        v = np.random.randn(3)
        v /= np.linalg.norm(v)
        return v
    
    def move_point(points, idx, scale):
        """
        Perform a Metropolis-Hastings move on point idx.
        scale controls the size of the perturbation.
        Returns new points array and whether the move was accepted.
        """
        new_pts = points.copy()
        
        # Generate a random direction for perturbation
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        
        # Perturb the point
        perturbation = scale * direction
        new_pts[idx] = points[idx] + perturbation
        
        # Normalize to unit sphere
        norm = np.linalg.norm(new_pts[idx])
        if norm < 1e-15:
            new_pts[idx] = random_point()
        else:
            new_pts[idx] /= norm
        
        return new_pts
    
    def compute_energy_fast(points):
        """Compute energy using vectorized operations for speed"""
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        
        # Avoid log(0) by masking diagonal and very small distances
        np.fill_diagonal(dists, 1.0)
        dists = np.maximum(dists, 1e-15)
        
        # Energy is sum over i<j of log(1/d) = -sum over i<j of log(d)
        # Use upper triangle only
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        return -np.sum(np.log(d))
    
    # Simulated annealing parameters
    T_start = 5.0
    T_end = 0.001
    n_iterations = 50000
    cool_factor = (T_end / T_start) ** (1.0 / n_iterations)
    
    current_energy = compute_energy_fast(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    random.seed(42)
    np.random.seed(42)
    
    for it in range(n_iterations):
        # Geometrically informed cooling schedule
        T = T_start * (cool_factor ** it)
        
        # Choose which point to move - bias towards points with high local energy contribution
        idx = np.random.randint(N)
        
        # Scale of perturbation decreases with temperature
        scale = 0.1 * (T / T_start) ** 0.5 + 0.01
        
        new_pts = move_point(pts, idx, scale)
        new_energy = compute_energy_fast(new_pts)
        
        # Metropolis acceptance criterion
        delta_E = new_energy - current_energy
        if delta_E < 0:
            # Accept if energy decreases
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        else:
            # Accept with probability exp(-delta_E / T)
            if T > 1e-10:
                prob = math.exp(-delta_E / T)
                if random.random() < prob:
                    pts = new_pts
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_pts = pts.copy()
    
    # Final deterministic polish pass using gradient-based refinement
    # Use simple iterative gradient descent with line search
    pts = best_pts.copy()
    
    def gradient_descent_polish(points, max_iters=10000):
        """Refine points using gradient descent on the sphere"""
        pts = points.copy()
        lr = 0.01
        
        for it in range(max_iters):
            # Compute gradient of energy with respect to each point
            grads = np.zeros_like(pts)
            
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < 1e-15:
                        continue
                    # d/dxi log(1/|xi-xj|) = -d/dxi log|xi-xj| = -(xi-xj)/|xi-xj|^2
                    # But we're on the sphere, so we need the tangential component
                    grad_i = -(pts[i] - pts[j]) / (d * d)
                    # Project gradient onto tangent space at pts[i]
                    proj = grad_i - np.dot(grad_i, pts[i]) * pts[i]
                    grads[i] += proj
            
            # Update points with gradient descent
            new_pts = pts - lr * grads
            
            # Project back to unit sphere
            norms = np.linalg.norm(new_pts, axis=1)
            norms = np.maximum(norms, 1e-15)
            new_pts = new_pts / norms[:, np.newaxis]
            
            # Check convergence
            if np.max(np.abs(new_pts - pts)) < 1e-12:
                break
            
            pts = new_pts
            # Decay learning rate
            lr *= 0.999
        
        return pts
    
    # Perform polish
    polished_pts = gradient_descent_polish(best_pts)
    
    # Check if polished is better
    polished_energy = compute_energy_fast(polished_pts)
    if polished_energy < best_energy:
        best_pts = polished_pts
        best_energy = polished_energy
    
    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    best_pts = best_pts / norms[:, np.newaxis]
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them
                best_pts[j] = best_pts[j] + 1e-8 * np.random.randn(3)
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
