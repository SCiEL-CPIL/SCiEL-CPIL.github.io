import numpy as np
from scipy.spatial import cKDTree
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 seconds deadline
    
    # Initial configuration using Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization parameters
    learning_rate = 0.01
    num_iterations = 10000
    
    # Potential energy function and its gradient
    def potential_energy(points):
        # Compute pairwise distances
        dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0)
        # Repulsive potential
        U = np.sum(1.0 / (dists * dists))
        return U
    
    def gradient(points):
        # Compute gradients of the potential energy
        dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, 1.0)
        # Gradient of 1/d^2 is -2*(pi - pj)/d^4
        grad = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i != j:
                    diff = points[i] - points[j]
                    d2 = np.dot(diff, diff)
                    grad[i] += -2.0 * diff / (d2 * d2)
        return grad
    
    # Optimization loop
    for it in range(num_iterations):
        if time.time() > deadline - 10.0:  # Safety margin
            break
        
        # Compute gradient
        grad = gradient(points)
        
        # Update points
        points -= learning_rate * grad
        
        # Normalize points to the unit sphere
        norms = np.linalg.norm(points, axis=1)
        points /= norms[:, np.newaxis]
    
    # Evaluate covering radius
    def compute_covering_radius(points):
        # Create a dense grid of test points on the sphere
        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2 * np.pi, 200)
        theta, phi = np.meshgrid(theta, phi)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        test_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Find nearest neighbor for each test point
        tree = cKDTree(points)
        dists, _ = tree.query(test_points)
        
        # Convert Euclidean distances to angular distances
        angular_dists = 2 * np.arcsin(dists / 2)
        
        # Return maximum angular distance in degrees
        return np.max(angular_dists) * 180 / np.pi
    
    # Return the optimized points
    return points
