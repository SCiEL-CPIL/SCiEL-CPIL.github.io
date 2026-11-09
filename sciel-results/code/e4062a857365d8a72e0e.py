"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Physics-based relaxation: treat circles as repulsive particles with hard-core
    constraints, initialized on a hexagonal lattice, and optimized via gradient
    ascent on a soft-penalty energy function.
    
    The energy function penalizes:
    1. Overlaps between circles (soft penalty for pairs closer than 2r)
    2. Circles extending beyond the unit circle boundary
    
    We use a simulated annealing-like approach with adaptive step sizes.
    """
    np.random.seed(42)
    
    # Initialize on a hexagonal lattice within the unit disk
    def hex_lattice_init(n):
        """Generate n points on a hexagonal lattice clipped to unit disk."""
        # Start with a dense hex lattice and pick the best n points
        spacing = 0.06  # initial guess
        points = []
        
        # Generate hex lattice points
        for row in range(-10, 11):
            for col in range(-10, 11):
                x = (col + 0.5 * (row % 2)) * spacing
                y = row * spacing * np.sqrt(3) / 2
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) < n:
            # Increase density
            spacing *= 0.9
            return hex_lattice_init(n)
        
        # Sort by distance from center and pick the closest n
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)
        return points[idx[:n]].copy()
    
    # Initial configuration
    pts = hex_lattice_init(N)
    
    # Compute initial radius
    def compute_radius(pts):
        """Compute the largest valid radius for given center positions."""
        if len(pts) < 2:
            return min(1 - np.max(np.hypot(pts[:, 0], pts[:, 1])), 0.5)
        
        # Minimum distance between any pair
        from scipy.spatial.distance import pdist
        min_dist = np.min(pdist(pts))
        
        # Maximum distance from origin
        max_r = np.max(np.hypot(pts[:, 0], pts[:, 1]))
        
        return min(min_dist / 2.0, 1.0 - max_r)
    
    r0 = compute_radius(pts)
    
    # Energy function: we want to maximize r, so minimize -r
    # Use a soft penalty formulation
    def energy(pts, r_target):
        """
        Compute a soft energy that penalizes:
        - Pairs with distance < 2*r_target (overlap)
        - Points with |p| > 1 - r_target (outside container)
        
        Returns (energy, gradient)
        """
        n = len(pts)
        E = 0.0
        grad = np.zeros_like(pts)
        
        # Pairwise overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Coincident: push apart randomly
                    grad[i] += np.random.randn(2) * 0.1
                    grad[j] -= np.random.randn(2) * 0.1
                    E += 10.0
                    continue
                
                # Soft penalty: quadratic for dist < 2*r_target
                if dist < 2 * r_target:
                    overlap = 2 * r_target - dist
                    penalty = 100.0 * overlap * overlap
                    E += penalty
                    # Gradient: dE/ddist * dd/pts
                    # dE/ddist = -200 * overlap
                    # dd/pts[i] = d / dist
                    force = 200.0 * overlap / dist
                    grad[i] += force * d
                    grad[j] -= force * d
        
        # Boundary penalties
        for i in range(n):
            r = np.linalg.norm(pts[i])
            if r > 1 - r_target:
                violation = r - (1 - r_target)
                penalty = 50.0 * violation * violation
                E += penalty
                # Gradient
                force = 100.0 * violation / max(r, 1e-10)
                grad[i] -= force * pts[i]
        
        return E, grad
    
    # Optimize using a simple gradient-based approach with annealing
    best_r = r0
    best_pts = pts.copy()
    
    n_iter = 3000
    T = 0.01  # temperature
    lr = 0.001  # learning rate
    
    for it in range(n_iter):
        # Compute current radius
        r_curr = compute_radius(pts)
        
        if r_curr > best_r:
            best_r = r_curr
            best_pts = pts.copy()
        
        # Use current radius as target for energy
        r_target = r_curr
        
        # Compute energy and gradient
        E, grad = energy(pts, r_target)
        
        # Gradient ascent (we want to maximize r, so we move in direction that increases r)
        # The energy is a penalty, so we want to minimize it -> gradient descent on E
        # But we also want to increase r, so let's use a combined approach
        
        # Actually, let's use a different strategy: 
        # Use the gradient of the energy to fix violations, then nudge points to increase r
        
        # Step 1: Fix violations using gradient descent on energy
        step = -lr * grad
        
        # Add small random perturbation (Langevin)
        noise = np.random.randn(*pts.shape) * T
        step += noise
        
        # Apply step
        pts_new = pts + step
        
        # Ensure points stay within a reasonable region
        pts_new = np.clip(pts_new, -1.5, 1.5)
        
        # Check if new configuration is valid and has better radius
        r_new = compute_radius(pts_new)
        
        # Accept if better or with some probability (simulated annealing)
        if r_new > r_curr:
            pts = pts_new
        else:
            # Accept with probability exp(-(r_curr - r_new) / T)
            delta = r_curr - r_new
            if delta < 1e-10:
                pts = pts_new
            else:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    pts = pts_new
        
        # Anneal
        T *= 0.999
        lr *= 0.9995
        
        # Early stopping if no improvement for a while
        if it % 100 == 0 and it > 0:
            # Check convergence
            pass
    
    # Final refinement: try to improve the best configuration
    pts = best_pts.copy()
    
    # Local optimization: for each point, try small perturbations
    for _ in range(500):
        r_curr = compute_radius(pts)
        improved = False
        
        for i in range(N):
            # Try small random moves
            for _ in range(5):
                delta = np.random.randn(2) * 0.001
                pts_new = pts.copy()
                pts_new[i] += delta
                r_new = compute_radius(pts_new)
                
                if r_new > r_curr:
                    pts = pts_new
                    r_curr = r_new
                    improved = True
                    break
        
        if not improved:
            break
    
    final_r = compute_radius(pts)
    if final_r > best_r:
        best_r = final_r
        best_pts = pts.copy()
    
    return best_pts

# EVOLVE-BLOCK-END
