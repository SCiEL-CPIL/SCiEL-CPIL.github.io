import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Initial configuration: Random points in the cube
    # Or a jittered grid for better start
    # Let's use a jittered 4x4x4 grid subset
    # 4x4x4 = 64 points. We need 50.
    # We can just generate 50 random points, it's often robust.
    np.random.seed(42)
    points = np.random.uniform(-0.5, 0.5, (N, 3))
    
    # To be safer, let's try a structured start
    # 5x5x5 grid = 125 points.
    # We can select 50 points that are well spread.
    # Actually, random is fine if the optimizer is good.
    
    best_points = points.copy()
    best_score = _calculate_score(points)
    
    # Parameters
    beta = 20.0
    step_size = 0.01
    
    for iteration in range(100000):
        if time.time() > deadline:
            break
            
        # Calculate distances
        # d[i, j] is distance between i and j
        # Efficient calculation:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
        dist = np.sqrt(dist_sq)
        
        # Smooth min approximation
        # We want to maximize the minimum distance.
        # Equivalent to minimizing -soft_min(dist)
        # soft_min = -1/beta * log(sum(exp(-beta * dist)))
        # Gradient of soft_min w.r.t c_i:
        # sum_j (dist_ij * exp(-beta * dist_ij) / sum_k exp(-beta * dist_ik)) * (c_i - c_j) / dist_ij
        # = sum_j (exp(-beta * dist_ij) / sum_k exp(-beta * dist_ik)) * (c_i - c_j)
        
        exp_term = np.exp(-beta * dist)
        np.fill_diagonal(exp_term, 0.0)
        
        # Weights w_ij = exp(-beta * d_ij) / sum_k exp(-beta * d_ik)
        # Note: sum over k != i
        denom = np.sum(exp_term, axis=1, keepdims=True)
        # Avoid division by zero if all distances are huge (unlikely)
        denom[denom == 0] = 1.0
        
        weights = exp_term / denom
        
        # Gradient of the loss (which is -soft_min) is - sum_j w_ij * (c_i - c_j)
        # We want to minimize loss, so we move in direction of -grad_loss = sum_j w_ij * (c_i - c_j)
        # This is a repulsion force.
        
        # Force on i = sum_j weights[i, j] * (points[i] - points[j])
        # points[i] is (N, 3), points[j] is (N, 3)
        # diff is (N, N, 3)
        # force = sum_j weights[i, j, None] * diff[i, j]
        force = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        
        # Update
        points += step_size * force
        
        # Project to cube [-0.5, 0.5]
        points = np.clip(points, -0.5, 0.5)
        
        # Calculate score
        score = _calculate_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
        # Annealing: reduce step size, increase beta
        if iteration % 1000 == 0:
            step_size *= 0.9
            beta *= 1.2
            # Add small noise to escape local minima
            if np.random.rand() < 0.1:
                points += np.random.uniform(-0.01, 0.01, (N, 3))
                points = np.clip(points, -0.5, 0.5)
                
    return best_points

def _calculate_score(points):
    # min_{i<j} |c_i - c_j| / 2
    # min_i min_k (0.5 - |c_ik|)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distance
    min_wall = 0.5 - np.max(np.abs(points), axis=1)
    min_wall_dist = np.min(min_wall)
    
    r = min(min_dist / 2.0, min_wall_dist)
    return r
