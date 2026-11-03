"""Packing 300 equal circles in a unit circle - force-directed optimization."""
import numpy as np
from scipy.spatial import Delaunay
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Stage 1: Initial hexagonal lattice configuration
    def hex_lattice_init(n_points):
        """Generate initial hexagonal lattice points in unit disk."""
        # Estimate spacing for n_points in unit disk
        # Area of unit disk = pi, area per point ≈ pi/n_points
        # Hexagonal spacing: area per point = spacing^2 * sqrt(3)/2
        spacing = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
        
        points = []
        k = int(1.0 / spacing) + 3
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if x*x + y*y <= (1.0 - spacing/2)**2:
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have too many, select the best N
        if len(points) > n_points:
            # Sort by distance from center, take closest N
            dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            idx = np.argsort(dist, kind='stable')[:n_points]
            points = points[idx]
        
        # If we have too few, add random points
        if len(points) < n_points:
            remaining = n_points - len(points)
            random_pts = np.random.uniform(-0.9, 0.9, size=(remaining, 2))
            points = np.vstack([points, random_pts])
        
        return points[:n_points]
    
    points = hex_lattice_init(N)
    
    # Stage 2: Force-directed optimization
    def compute_forces(pts, repulsion_strength=1.0, attraction_strength=0.1):
        """Compute forces for force-directed optimization."""
        n = len(pts)
        forces = np.zeros_like(pts)
        
        # Repulsive forces between all pairs (soft repulsion)
        for i in range(n):
            for j in range(i+1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    dist = 1e-10
                    diff = np.array([1e-5, 0])
                
                # Strong repulsion when close
                force_mag = repulsion_strength / (dist**2)
                force_dir = diff / dist
                forces[i] += force_dir * force_mag
                forces[j] -= force_dir * force_mag
        
        # Attractive force toward center (weak, to keep points in disk)
        for i in range(n):
            r = np.linalg.norm(pts[i])
            if r > 0.1:
                forces[i] -= attraction_strength * pts[i] / r
        
        # Constraint force: push points back inside disk
        for i in range(n):
            r = np.linalg.norm(pts[i])
            if r > 0.95:
                # Push back toward center
                forces[i] -= 5.0 * (r - 0.95) * pts[i] / r
        
        return forces
    
    def compute_radius(pts):
        """Compute the packing radius."""
        n = len(pts)
        min_pair_dist = np.inf
        min_edge_dist = np.inf
        
        for i in range(n):
            # Distance to boundary
            r = np.linalg.norm(pts[i])
            edge_dist = 1.0 - r
            if edge_dist < min_edge_dist:
                min_edge_dist = edge_dist
            
            for j in range(i+1, n):
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < min_pair_dist:
                    min_pair_dist = dist
        
        return min(min_pair_dist / 2, min_edge_dist)
    
    # Optimize using gradient descent with adaptive step size
    current_radius = compute_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    # Use a smoother objective: maximize min distance
    def smooth_min_dist(pts, temperature=0.01):
        """Soft-min of pair distances using log-sum-exp."""
        n = len(pts)
        dists = []
        for i in range(n):
            for j in range(i+1, n):
                dists.append(np.linalg.norm(pts[i] - pts[j]))
            dists.append(1.0 - np.linalg.norm(pts[i]))
        
        dists = np.array(dists)
        # Soft min: -log-sum-exp(-dists/temperature) * temperature
        max_d = np.max(dists)
        log_sum = np.log(np.sum(np.exp(-(dists - max_d) / temperature)))
        soft_min = max_d + temperature * log_sum
        return soft_min
    
    # Gradient-based optimization
    def gradient_step(pts, lr=0.001, temperature=0.01):
        """Perform one gradient step to maximize soft-min distance."""
        n = len(pts)
        grad = np.zeros_like(pts)
        
        # Compute gradient of soft-min
        dists = []
        indices = []
        for i in range(n):
            for j in range(i+1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    dists.append(dist)
                    indices.append((i, j, d, dist))
            # Boundary distance
            r = np.linalg.norm(pts[i])
            if r < 1.0:
                dists.append(1.0 - r)
                indices.append((i, -1, pts[i], r))
        
        dists = np.array(dists)
        max_d = np.max(dists)
        weights = np.exp(-(dists - max_d) / temperature)
        weights /= np.sum(weights)
        
        for w, (i, j, d, dist) in zip(weights, indices):
            if j == -1:
                # Boundary: gradient of (1 - |p_i|)
                if dist > 1e-10:
                    grad[i] += w * (-pts[i] / dist)
            else:
                # Pair distance: gradient of |p_i - p_j|
                if dist > 1e-10:
                    grad_dir = d / dist
                    grad[i] += w * grad_dir
                    grad[j] -= w * grad_dir
        
        return pts + lr * grad
    
    # Run optimization for a fixed number of iterations
    max_iter = 500
    lr = 0.005
    temperature = 0.05
    
    for iteration in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Adaptive learning rate
        if iteration % 100 == 0:
            lr *= 0.8
            temperature *= 0.9
        
        points = gradient_step(points, lr, temperature)
        
        # Keep points in disk
        for i in range(N):
            r = np.linalg.norm(points[i])
            if r > 0.999:
                points[i] *= 0.999 / r
        
        # Check improvement
        current_radius = compute_radius(points)
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    points = best_points
    
    # Stage 3: Local refinement with tighter tolerances
    # Use Newton-like method for local optimization
    for polish_iter in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Compute all pair distances and find critical pairs
        n = len(points)
        critical_pairs = []
        min_dist = np.inf
        
        for i in range(n):
            for j in range(i+1, n):
                d = points[i] - points[j]
                dist = np.linalg.norm(d)
                if dist < min_dist:
                    min_dist = dist
                    critical_pairs = [(i, j)]
                elif abs(dist - min_dist) < 1e-6:
                    critical_pairs.append((i, j))
        
        # Also check boundary
        for i in range(n):
            r = np.linalg.norm(points[i])
            edge_dist = 1.0 - r
            if edge_dist < min_dist:
                min_dist = edge_dist
                critical_pairs = [(i, -1)]
        
        # Apply small perturbations to critical pairs
        if critical_pairs:
            for i, j in critical_pairs[:10]:  # Limit to avoid O(n^2) blowup
                if j == -1:
                    # Push point away from boundary
                    r = np.linalg.norm(points[i])
                    if r > 0:
                        points[i] -= 0.001 * points[i] / r
                else:
                    d = points[i] - points[j]
                    dist = np.linalg.norm(d)
                    if dist > 1e-10:
                        points[i] += 0.001 * d / dist
                        points[j] -= 0.001 * d / dist
        
        # Keep in disk
        for i in range(N):
            r = np.linalg.norm(points[i])
            if r > 0.999:
                points[i] *= 0.999 / r
    
    final_radius = compute_radius(points)
    if final_radius > best_radius:
        best_points = points
        best_radius = final_radius
    
    return best_points
