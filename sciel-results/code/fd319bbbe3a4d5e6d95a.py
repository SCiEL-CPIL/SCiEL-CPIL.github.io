import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    best_points = None
    best_score = 0.0
    
    # Multiple restarts to avoid local minima
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize with hexagonal lattice + jitter
        # Estimate spacing
        # Area = pi. N circles. Area per circle ~ pi/N.
        # Hex packing density ~ 0.9069.
        # Effective area per circle ~ (pi/N) / 0.9069.
        # Side of hex cell ~ sqrt(effective_area / (sqrt(3)/2)).
        # Simpler: just use a grid and scale.
        
        # Generate hex points
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        # Adjust spacing to fit N points roughly
        # Try to generate enough points
        points = []
        k = 10
        while len(points) < N:
            for i in range(-k, k+1):
                for j in range(-k, k+1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if np.hypot(x, y) <= 1.0:
                        points.append([x, y])
            k += 1
            if k > 20: break
            
        points = np.array(points[:N])
        
        # Add jitter
        np.random.seed(restart)
        points += np.random.normal(0, spacing * 0.1, points.shape)
        
        # Ensure inside
        radii = np.linalg.norm(points, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            points[mask] = points[mask] / radii[mask, np.newaxis] * 0.99
            
        # Optimize
        k_temp = 50.0  # Inverse temperature for soft-min
        eta = 0.01     # Step size
        
        # Adaptive step size or fixed?
        # Let's try fixed with damping.
        
        for step in range(2000):
            if time.time() > deadline - 2.0:
                break
                
            # Compute distances
            # d[i, j] = |c_i - c_j|
            # Use broadcasting
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            
            # Mask diagonal
            np.fill_diagonal(d, np.inf)
            
            # Boundary distances
            radii = np.linalg.norm(points, axis=1)
            b_dist = 1.0 - radii
            
            # Compute weights for soft-min
            # We want to maximize min(d, b_dist)
            # Surrogate: -1/k * log( sum exp(-k * val) )
            
            # Internal weights
            # w_ij = exp(-k * d_ij)
            # To avoid underflow/overflow, subtract max?
            # d is usually small (0.05). k=50. k*d = 2.5. exp(-2.5) is fine.
            # If d is 0, exp(0)=1.
            
            w_int = np.exp(-k_temp * d)
            np.fill_diagonal(w_int, 0)
            
            # Boundary weights
            w_bnd = np.exp(-k_temp * b_dist)
            
            # Sum
            S = np.sum(w_int) + np.sum(w_bnd)
            
            if S == 0:
                break
                
            # Gradient
            # grad_i = 1/S * [ sum_j w_ij * (c_i - c_j)/d_ij - w_bi * c_i/|c_i| ]
            
            # Term 1: sum_j w_ij * (c_i - c_j)/d_ij
            # diff is (c_i - c_j)
            # d is |diff|
            # Avoid div by 0
            d_safe = np.where(d > 1e-10, d, 1.0)
            
            # weights * unit vectors
            # w_int[i,j] * diff[i,j] / d_safe[i,j]
            force_int = np.sum(w_int[:, :, np.newaxis] * diff / d_safe[:, :, np.newaxis], axis=1)
            
            # Term 2: w_bi * c_i/|c_i|
            radii_safe = np.where(radii > 1e-10, radii, 1.0)
            force_bnd = w_bnd[:, np.newaxis] * (points / radii_safe[:, np.newaxis])
            
            grad = (force_int - force_bnd) / S
            
            # Update
            points += eta * grad
            
            # Project to disk
            radii = np.linalg.norm(points, axis=1)
            mask = radii > 1.0
            if np.any(mask):
                points[mask] = points[mask] / radii[mask, np.newaxis]
                
            # Calculate current score
            d_min = np.min(d)
            b_min = np.min(b_dist)
            score = min(d_min / 2.0, b_min)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
            # Adaptive step size?
            # If score decreased significantly, reduce eta
            # Simple: just keep eta small.
            
    return best_points
