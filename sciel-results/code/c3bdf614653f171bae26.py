"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Finds a packing of 50 spheres in a unit cube [-0.5, 0.5]^3.
    Uses Repulsive Force Optimization (Force-Directed Layout) with Binary Search on Radius.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer

    def check_feasibility(R, seed, max_iters=20000):
        """
        Checks if a packing of N spheres with radius R is feasible.
        Uses iterative force-directed relaxation.
        Returns (is_feasible, best_config, min_dist).
        """
        rng = np.random.default_rng(seed)
        
        # Initialize positions randomly within the valid region for radius R
        # If R is too large, the valid region might be empty, but we handle that by clamping.
        # Better: Initialize in [-0.5, 0.5] and let forces push them in.
        # But starting valid is better.
        lower = -0.5 + R
        upper = 0.5 - R
        
        if lower > upper:
            # R is too large, impossible
            return False, None, 0.0
            
        X = rng.uniform(lower, upper, size=(N, 3))
        
        best_min_dist = 0.0
        best_X = X.copy()
        
        # Learning rate decay
        lr_start = 0.1
        lr_end = 0.001
        lr_decay = (lr_end / lr_start) ** (1.0 / max_iters)
        lr = lr_start
        
        for t in range(max_iters):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Calculate forces
            # Pairwise forces
            # Diffs: (N, N, 3)
            diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            
            # Avoid division by zero
            dists_safe = np.where(dists > 1e-8, dists, 1.0)
            
            # Force magnitude: Repulsion if dist < 2R
            # F = k * (2R - dist) if dist < 2R else 0
            # Direction: (X_j - X_i) / dist
            # We want force on i from j.
            # Let's compute force matrix F (N, N, 3) where F[i,j] is force on i from j.
            
            # Overlap amount
            overlap = np.maximum(0, 2 * R - dists)
            
            # Force vector from j to i is -diffs[i,j] / dists[i,j] * overlap[i,j]
            # Wait, diffs[i,j] = X[i] - X[j]. So vector from j to i is X[i]-X[j].
            # Repulsion pushes i away from j. So force is in direction X[i]-X[j].
            # Force on i = sum_j ( (X[i]-X[j]) / |X[i]-X[j]| * (2R - |X[i]-X[j]|) )
            
            # Vectorize:
            # unit_vecs = diffs / dists_safe[:, :, np.newaxis]
            # forces = unit_vecs * overlap[:, :, np.newaxis]
            # Net force on i is sum over j of forces[i,j]
            
            # Note: diagonal is 0, so it doesn't matter.
            unit_vecs = diffs / dists_safe[:, :, np.newaxis]
            # Mask diagonal to 0 to be safe, though overlap is 0 there anyway if dist>0
            np.fill_diagonal(overlap, 0)
            
            force_vectors = unit_vecs * overlap[:, :, np.newaxis]
            net_forces = np.sum(force_vectors, axis=1)
            
            # Wall forces
            # For each dimension, if X[i,k] > 0.5 - R, push left (negative).
            # If X[i,k] < -0.5 + R, push right (positive).
            # Violation = max(0, X[i,k] - (0.5 - R)) for right wall
            # Force = -violation * k_wall
            # Let's use same strength as sphere-sphere.
            
            # Right wall (x=0.5):
            # dist_to_wall = 0.5 - X
            # violation = max(0, R - dist_to_wall) = max(0, R - (0.5 - X)) = max(0, X - (0.5 - R))
            # Force direction: -1 (towards center)
            
            # Left wall (x=-0.5):
            # dist_to_wall = X - (-0.5) = X + 0.5
            # violation = max(0, R - (X + 0.5)) = max(0, (R - 0.5) - X)
            # Force direction: +1
            
            # Generalize for all 3 axes.
            # X shape (N, 3)
            
            # Right walls (positive side)
            viol_right = np.maximum(0, X - (0.5 - R))
            force_right = -viol_right
            
            # Left walls (negative side)
            viol_left = np.maximum(0, (-0.5 + R) - X)
            force_left = viol_left
            
            net_forces += (force_right + force_left)
            
            # Update positions
            X += lr * net_forces
            
            # Clip to cube to ensure they don't fly out too far
            X = np.clip(X, -0.5, 0.5)
            
            # Calculate current min distance
            # We can do this every few steps to save time, or every step if fast enough.
            # N=50, O(N^2) is 2500 ops, very fast.
            if t % 10 == 0:
                min_d = np.inf
                # Pairwise
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(X[i] - X[j])
                        if d < min_d:
                            min_d = d
                # Walls
                for i in range(N):
                    for k in range(3):
                        d_wall = 0.5 - abs(X[i][k])
                        if d_wall < min_d:
                            min_d = d_wall
                
                min_dist_half = min_d / 2.0
                if min_dist_half > best_min_dist:
                    best_min_dist = min_dist_half
                    best_X = X.copy()
                    
            # Decay learning rate
            lr *= lr_decay
            
        # Final check
        min_d_final = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_X[i] - best_X[j])
                if d < min_d_final:
                    min_d_final = d
        for i in range(N):
            for k in range(3):
                d_wall = 0.5 - abs(best_X[i][k])
                if d_wall < min_d_final:
                    min_d_final = d_wall
                    
        final_min_dist = min_d_final / 2.0
        is_feasible = (final_min_dist >= R - 1e-7)
        
        return is_feasible, best_X, final_min_dist

    # Binary Search for R
    low_R = 0.12
    high_R = 0.14
    best_config = None
    best_score = 0.0
    
    # Coarse search
    # We will run a few iterations.
    # To improve robustness, we can run multiple seeds for each R.
    
    num_binary_steps = 25
    for step in range(num_binary_steps):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        mid_R = (low_R + high_R) / 2.0
        
        # Try 2 different seeds for this R to increase chance of finding a feasible pack
        feasible = False
        best_config_for_R = None
        best_score_for_R = 0.0
        
        for seed in [step * 100, step * 100 + 1]:
            is_feas, config, score = check_feasibility(mid_R, seed, max_iters=5000)
            if is_feas:
                feasible = True
                if score > best_score_for_R:
                    best_score_for_R = score
                    best_config_for_R = config
            # If not feasible, we might still want to keep the best config found?
            # No, for binary search, we need to know if R is feasible.
            # If one seed finds it feasible, R is feasible.
        
        if feasible:
            low_R = mid_R
            if best_score_for_R > best_score:
                best_score = best_score_for_R
                best_config = best_config_for_R
        else:
            high_R = mid_R
            
    if best_config is None:
        # Fallback
        best_config = np.zeros((N, 3))
        
    return best_config

# EVOLVE-BLOCK-END
