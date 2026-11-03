import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Initialization
    s = 0.062
    points_list = []
    k = int(1.0 / s) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            if x*x + y*y < 1.0:
                points_list.append([x, y])
    points = np.array(points_list)
    
    while len(points) < N:
        s *= 0.95
        points_list = []
        k = int(1.0 / s) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = i * s + (j % 2) * s / 2.0
                y = j * s * np.sqrt(3.0) / 2.0
                if x*x + y*y < 1.0:
                    points_list.append([x, y])
        points = np.array(points_list)
        if time.time() > deadline:
            break

    if len(points) > N:
        dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
        indices = np.argsort(dists_to_boundary)[::-1][:N]
        points = points[indices]
    else:
        while len(points) < N:
            p = np.random.rand(2) * 2 - 1
            while np.linalg.norm(p) > 1:
                p = np.random.rand(2) * 2 - 1
            points = np.vstack([points, p])
        points = points[:N]

    best_points = points.copy()
    best_score = 0.0
    
    # Soft-max parameters
    beta = 20.0 # Temperature
    alpha = 0.05 # Step size
    mu = 0.5 # Momentum
    velocity = np.zeros_like(points)
    
    while time.time() < deadline - 2.0:
        # Compute distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Constraints:
        # 1. Pair distances: d_ij / 2
        # 2. Boundary distances: 1 - |c_i|
        
        # We want to maximize the minimum of these.
        # Let's form a vector of all "radius candidates".
        # Pair candidates: dists / 2. There are N*(N-1)/2 of them.
        # Boundary candidates: 1 - norms. There are N of them.
        
        # For efficiency, we can compute the gradient directly without storing all pair distances if we use the soft-max trick carefully,
        # but N=300 is small enough to store the NxN matrix.
        
        pair_radii = dists / 2.0
        boundary_radii = 1.0 - np.linalg.norm(points, axis=1)
        
        # Soft-max of the combined set?
        # It's easier to do two separate soft-maxes and combine?
        # No, the min is over the union.
        # max(min(A), min(B)) is not min(max(A), max(B)).
        # But we want to maximize the global minimum.
        
        # Let's use the soft-max of the entire set of constraints.
        # Total constraints: N*(N-1)/2 + N.
        # This is ~45000 + 300 = 45300 values.
        # Computing exp(beta * values) is fine.
        
        # Flatten pair radii (upper triangle)
        # Actually, we can just use the full NxN matrix and ignore diagonal.
        # But we need to be careful not to double count.
        # Let's just use the full NxN matrix of pair radii, and the N boundary radii.
        # The soft-max will naturally weight the smallest ones.
        # The gradient will sum contributions from all pairs.
        
        # To save memory/time, we can compute the soft-max weights for pairs and boundaries separately?
        # No, the min is global.
        
        # Let's construct the full vector of radii.
        # This might be slow if we do it naively.
        # Let's stick to the active set approach but with a small epsilon, or the soft-max on the full set.
        
        # Given N=300, the full set is small.
        
        # 1. Pair part
        # We want grad of soft_max(pair_radii)
        # But we also have boundary.
        
        # Let's do this:
        # R_pair = soft_max(pair_radii)
        # R_bnd = soft_max(boundary_radii)
        # R_total = soft_max([R_pair, R_bnd])? No.
        
        # The objective is min( min_pairs, min_boundary ).
        # This is equivalent to min( all_pair_radii, all_boundary_radii ).
        
        # So we can just concatenate the pair radii and boundary radii into one vector.
        
        # Flatten pair radii (excluding diagonal)
        mask = ~np.eye(N, dtype=bool)
        pair_flat = pair_radii[mask]
        
        # Concatenate
        all_radii = np.concatenate([pair_flat, boundary_radii])
        
        # Soft-max
        # To avoid overflow, subtract max
        max_r = np.max(all_radii)
        exp_vals = np.exp(beta * (all_radii - max_r))
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        # Now compute gradient
        # The gradient of soft_max is sum( weight_k * grad(V_k) )
        # V_k are the radii.
        
        grads = np.zeros_like(points)
        
        # Pair gradients
        # We need to map the weights back to the NxN matrix
        pair_weights = weights[:len(pair_flat)]
        pair_weight_matrix = np.zeros((N, N))
        pair_weight_matrix[mask] = pair_weights
        
        # For each pair (i,j) with weight w_ij:
        # grad_i += w_ij * (points[i] - points[j]) / dists[i,j]
        # grad_j += w_ij * (points[j] - points[i]) / dists[i,j]
        
        # Vectorized:
        # diff is points[i] - points[j]
        # We want sum_j w_ij * diff_ij / dists_ij
        
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        weighted_diff = diff * (pair_weight_matrix / safe_dists)[:, :, np.newaxis]
        
        # Sum over j for each i
        grads += np.sum(weighted_diff, axis=1)
        
        # Boundary gradients
        bnd_weights = weights[len(pair_flat):]
        # For point i with weight w_i:
        # grad_i += w_i * (-points[i] / |points[i]|)
        
        norms = np.linalg.norm(points, axis=1)
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        boundary_grads = -points / safe_norms[:, np.newaxis]
        boundary_grads *= bnd_weights[:, np.newaxis]
        
        grads += boundary_grads
        
        # Update with momentum
        velocity = mu * velocity + alpha * grads
        points += velocity
        
        # Project to unit circle
        new_norms = np.linalg.norm(points, axis=1)
        outside = new_norms > 1.0
        if np.any(outside):
            points[outside] *= (1.0 / new_norms[outside])[:, np.newaxis]
            velocity[outside] = 0.0 # Reset velocity for outside points? Or just let it be.
            
        # Calculate true score
        min_pair = np.min(dists) / 2.0
        min_bnd = np.min(boundary_radii)
        current_score = min(min_pair, min_bnd)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        # Decay alpha
        alpha *= 0.999
        
    return best_points
