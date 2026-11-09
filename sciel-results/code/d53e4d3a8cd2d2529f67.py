import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 600
    
    def calculate_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def project_to_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return points / norms

    def optimize_energy(points, iterations=5000, lr=0.01, beta=2.0):
        """
        Minimize repulsive energy E = - sum 1/d_ij^beta on the sphere.
        This helps distribute points evenly.
        """
        points = project_to_sphere(points.copy())
        n = len(points)
        
        for i in range(iterations):
            if time.time() - start_time > timeout - 5:
                break
                
            # Calculate distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(n, dtype=bool)
            dist_matrix[~mask] = 1e9
            
            # Energy gradient
            # E = - sum_{i<j} d_ij^-beta
            # dE/dp_i = - sum_{j != i} (-beta) * d_ij^(-beta - 2) * (p_i - p_j) / d_ij
            #          = beta * sum_{j != i} d_ij^(-beta - 2) * (p_i - p_j) / d_ij
            #          = beta * sum_{j != i} d_ij^(-beta - 1) * (p_i - p_j) / d_ij^2? No.
            # Let u = p_i - p_j. d = ||u||.
            # term = d^-beta.
            # d(term)/dp_i = -beta * d^(-beta-1) * (d(d)/dp_i)
            # d(d)/dp_i = (p_i - p_j) / d
            # d(term)/dp_i = -beta * d^(-beta-1) * (p_i - p_j) / d = -beta * d^(-beta-2) * (p_i - p_j)
            # Since we minimize E, and E = - sum..., grad E = - sum grad(-term) = sum beta * d^(-beta-2) * (p_i - p_j)
            # Wait, E = - sum d^-beta.
            # grad_p_i ( - d^-beta ) = - (-beta) d^(-beta-1) * grad_p_i(d)
            # = beta d^(-beta-1) * (p_i - p_j)/d
            # = beta d^(-beta-2) (p_i - p_j)
            
            grad = np.zeros_like(points)
            for k in range(n):
                p_k = points[k]
                # distances from k to all j
                d_kj = dist_matrix[k]
                # mask self
                d_kj[k] = 1e9
                # factors
                # sum_{j != k} beta * d_kj^-beta-2 * (p_k - p_j)
                # Note: (p_k - p_j) is row k of diff
                diff_k = diff[k]
                # We need to weight each vector (p_k - p_j) by d_kj^-beta-2
                # d_kj is 1D array, diff_k is 2D
                weights = np.power(d_kj, -beta - 2)
                weights[k] = 0
                grad_k = np.sum(diff_k * weights[:, np.newaxis], axis=0)
                grad[k] = beta * grad_k
            
            # Project to tangent space
            for k in range(n):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k
            
            # Update: We want to DECREASE energy.
            # So we move AGAINST the gradient.
            points = points - lr * grad
            points = project_to_sphere(points)
            
            # Decay learning rate
            if i % 100 == 0:
                lr *= 0.95
                
        return points

    def polish_min_dist(points, iterations=2000, lr=0.005):
        """
        Final polish to strictly maximize min distance.
        """
        points = project_to_sphere(points.copy())
        n = len(points)
        
        for i in range(iterations):
            if time.time() - start_time > timeout - 2:
                break
                
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(n, dtype=bool)
            dist_matrix[~mask] = np.inf
            d_min = np.min(dist_matrix)
            
            min_pairs = np.argwhere(dist_matrix == d_min)
            
            grad = np.zeros_like(points)
            for idx in min_pairs:
                i, j = idx[0], idx[1]
                vec = points[i] - points[j]
                d = np.linalg.norm(vec)
                if d > 1e-8:
                    unit = vec / d
                    grad[i] += unit
                    grad[j] -= unit
            
            for k in range(n):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k
            
            points = points + lr * grad # Ascend on min dist
            points = project_to_sphere(points)
            
            if i > 500:
                lr *= 0.999
                
        return points

    n = 14
    best_score = 0.0
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Fibonacci
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = np.linspace(-1, 1, n, endpoint=False) + 0.5/n
    r = np.sqrt(1 - y**2)
    theta = phi * np.arange(n)
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    inits.append(np.column_stack((x, y, z)))
    
    # 2. Random
    np.random.seed(42)
    inits.append(np.random.randn(n, 3))
    
    # 3. Random with different seed
    np.random.seed(123)
    inits.append(np.random.randn(n, 3))
    
    # 4. More random seeds for diversity
    for seed in [7, 99, 2024]:
        np.random.seed(seed)
        inits.append(np.random.randn(n, 3))

    for init in inits:
        if time.time() - start_time > timeout - 20:
            break
            
        # Phase 1: Spread out using energy minimization
        p1 = optimize_energy(init, iterations=2000, lr=0.02, beta=4.0)
        
        # Phase 2: Polish min distance
        p2 = polish_min_dist(p1, iterations=3000, lr=0.005)
        
        score = calculate_score(p2)
        if score > best_score:
            best_score = score
            best_points = p2

    if best_points is None:
        best_points = np.random.randn(n, 3)
        
    return best_points
