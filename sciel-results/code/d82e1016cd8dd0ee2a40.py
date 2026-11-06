"""Rewritten program: Fibonacci disk initialization + repulsion-based relaxation."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START

def construct():
    """
    Construct N=300 circle centers in the unit disk to maximize the minimum distance
    (and thus the common circle radius). Uses Fibonacci disk initialization followed
    by a repulsion-based relaxation with deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Fibonacci disk initialization
    def fibonacci_disk(n, radius=1.0):
        """Generate n points in a disk of given radius using Fibonacci spiral."""
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        i = np.arange(n)
        # Radius distribution: sqrt(i/n) * radius for uniform density
        r = radius * np.sqrt((i + 0.5) / n)
        theta = i * golden_angle
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y])
    
    # Compute score: largest common radius
    def compute_score(points):
        """Compute the largest common radius for given points."""
        n = len(points)
        if n < 2:
            return 0.0
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        # Distance from center to container boundary
        norms = np.sqrt(np.sum(points ** 2, axis=1))
        min_boundary = np.min(1.0 - norms)
        return min(min_dist / 2.0, min_boundary)
    
    # Initialize with Fibonacci disk
    points = fibonacci_disk(N)
    best_points = points.copy()
    best_score = compute_score(points)
    
    # Repulsion-based relaxation
    # We want to maximize the minimum distance between all pairs and from boundary
    # Use a force-based approach: each point repels others, and boundary repels points
    
    def repulsion_step(points, step_size, iters=100):
        """Perform repulsion-based relaxation."""
        n = len(points)
        for it in range(iters):
            # Compute pairwise distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Force from other points: repulsion proportional to 1/d^2 for close pairs
            # Use a softened version to avoid singularities
            force = np.zeros_like(points)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    d = dists[i, j]
                    if d < 1e-10:
                        # Random small displacement
                        force[i] += np.random.randn(2) * 0.01
                    else:
                        # Repulsion force: stronger when closer
                        f = 1.0 / (d * d + 1e-6)
                        force[i] += f * diff[i, j]
            
            # Boundary repulsion: push points inward if too close to boundary
            norms = np.sqrt(np.sum(points ** 2, axis=1))
            for i in range(n):
                if norms[i] > 0.95:  # Close to boundary
                    # Push inward
                    force[i] -= (points[i] / (norms[i] + 1e-10)) * (norms[i] - 0.95) * 2.0
            
            # Update positions with adaptive step size
            points = points + step_size * force
            
            # Project back into disk if needed
            norms = np.sqrt(np.sum(points ** 2, axis=1))
            mask = norms > 1.0
            if np.any(mask):
                points[mask] = points[mask] / norms[mask, np.newaxis] * 0.999
            
            # Check deadline
            if time.time() > deadline - 5.0:
                break
            
            # Track best
            score = compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        
        return points
    
    # Multi-scale relaxation: start with larger steps, then refine
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for step_size in step_sizes:
        if time.time() > deadline - 10.0:
            break
        points = repulsion_step(points, step_size, iters=200)
    
    # Final polish: very small steps
    if time.time() < deadline - 5.0:
        points = repulsion_step(points, 0.0001, iters=500)
    
    # Ensure we return the best found
    final_score = compute_score(points)
    if final_score > best_score:
        best_points = points.copy()
    
    return best_points

# EVOLVE-BLOCK-END
