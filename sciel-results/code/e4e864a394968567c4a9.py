"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # 5s safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check centers are inside the square [-0.5, 0.5]^2
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    # Distance to walls
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding self)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    r = min(min_wall, min_pair / 2.0)
    return r

def _generate_hex_grid():
    """Generate a hexagonal lattice packing as initial configuration."""
    best_centers = None
    best_r = 0.0
    
    # Try different row counts
    for n_rows in range(10, 25):
        n_cols = int(np.ceil(N / n_rows))
        
        # Hexagonal spacing: horizontal = s, vertical = s * sqrt(3)/2
        # With alternating offset of s/2
        # We need to fit in [-0.5, 0.5]^2
        
        # Let's compute the required spacing
        # Width: (n_cols - 1) * s + 2*r <= 1  =>  s + 2*r <= 1/(n_cols-1) if n_cols > 1
        # Actually, for hex: the width is (n_cols - 1) * s + 2*r
        # The height is (n_rows - 1) * s * sqrt(3)/2 + 2*r
        
        # We want to maximize r subject to:
        # (n_cols - 1) * s + 2*r <= 1
        # (n_rows - 1) * s * sqrt(3)/2 + 2*r <= 1
        # s >= 2*r (non-overlapping in hex: nearest neighbor distance = s)
        
        # From hex geometry, nearest neighbor distance = s
        # So we need s >= 2*r
        
        # Let's solve: maximize r
        # s = 2*r (tight packing)
        # (n_cols - 1) * 2*r + 2*r <= 1  =>  2*r * n_cols <= 1  =>  r <= 1/(2*n_cols)
        # (n_rows - 1) * 2*r * sqrt(3)/2 + 2*r <= 1  =>  r * ((n_rows-1)*sqrt(3) + 2) <= 1  =>  r <= 1/((n_rows-1)*sqrt(3) + 2)
        
        r1 = 1.0 / (2.0 * n_cols)
        r2 = 1.0 / ((n_rows - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        s = 2 * r
        
        # Build the grid
        centers = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x = -0.5 + r + j * s + (i % 2) * s / 2
                y = -0.5 + r + i * s * np.sqrt(3) / 2
                centers.append([x, y])
            if len(centers) >= N:
                break
        
        centers = np.array(centers[:N])
        
        # Check if all are in bounds
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            # Adjust: try to center the grid
            # Recompute with centering
            xs = centers[:, 0]
            ys = centers[:, 1]
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            centers[:, 0] -= cx
            centers[:, 1] -= cy
            
            # Now check if it fits
            if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
                continue
        
        r_actual = _compute_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers

def _simulated_annealing(centers, max_iter=20000, T_start=0.05, T_end=0.001):
    """
    Simulated Annealing to explore the configuration space.
    Objective: maximize minimum distance (radius).
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    current_r = best_r
    
    tree = cKDTree(centers)
    
    no_improve = 0
    T = T_start
    
    for it in range(max_iter):
        if _time_left() < 20.0:
            break
            
        # Update temperature (geometric cooling)
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        # Select a random subset of points to move
        # Moving fewer points is more efficient and helps local refinement
        n_move = 1 + (it % 3) # 1, 2, or 3 points
        
        # Pick random indices
        if n_move == 1:
            idxs = np.array([np.random.randint(N)])
        else:
            idxs = np.random.choice(N, size=n_move, replace=False)
            
        trial = centers.copy()
        
        # Generate perturbations
        # Scale step size with temperature
        step_size = T * 0.5
        for idx in idxs:
            # Random direction and magnitude
            angle = np.random.uniform(0, 2 * np.pi)
            # Gaussian-like step
            mag = np.random.randn() * step_size
            trial[idx, 0] += mag * np.cos(angle)
            trial[idx, 1] += mag * np.sin(angle)
            
        # Clip to boundaries
        trial[:, 0] = np.clip(trial[:, 0], -0.5, 0.5)
        trial[:, 1] = np.clip(trial[:, 1], -0.5, 0.5)
        
        # Evaluate
        r_trial = _compute_radius(trial)
        
        # Acceptance criterion
        if r_trial > current_r:
            # Always accept improvement
            centers = trial
            current_r = r_trial
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Accept worse solution with probability exp((r_trial - current_r) / T)
            # Note: r is positive, so delta is negative
            delta = r_trial - current_r
            # Avoid division by zero or overflow
            if T > 1e-12:
                prob = np.exp(delta / T)
            else:
                prob = 0.0
                
            if np.random.rand() < prob:
                centers = trial
                current_r = r_trial
                no_improve += 1
            else:
                no_improve += 1
                
        # Early stopping if no improvement for a long time and T is low
        if no_improve > 1000 and T < 0.01:
            break
            
    return best_centers, best_r

def _projected_gradient_refine(centers, max_iter=5000, lr=0.001, momentum=0.9):
    """
    High-precision refinement using projected gradient descent.
    Objective: Maximize sum of squared pairwise distances (encourages uniform spread)
    Subject to: All points inside square, and pairwise distances >= 2*r (where r is current min radius).
    
    Actually, a simpler and effective approach for the final stage is to maximize the minimum distance directly
    using a smooth approximation or by solving the convex-ish subproblems.
    
    However, the cell finding suggests: "Projected gradient descent with momentum maximizes sum of squared pairwise distances... constraint violation penalty pushes apart close pairs".
    
    Let's implement:
    Objective F(x) = sum_{i<j} |x_i - x_j|^2 - Lambda * sum_{i<j} max(0, 2*r_target - |x_i - x_j|)^2 - Mu * sum_i wall_penalty_i
    
    But r_target is changing.
    
    Alternative: Just maximize the "energy" E = sum_{i<j} 1/|x_i - x_j|^2 (Repulsive potential)
    This naturally pushes points apart. We just need to ensure they stay in the box.
    This is the standard "Thomson problem" in 2D.
    
    Let's use the Repulsive Potential approach with gradient descent.
    """
    centers = centers.copy()
    velocity = np.zeros_like(centers)
    
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    # We want to maximize the minimum distance.
    # Maximizing sum of inverse squared distances is a good proxy.
    # Gradient of -sum 1/d^2 is sum 2*(x_i - x_j)/d^4
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
            
        # Compute gradient of the repulsive potential
        # U = - sum_{i<j} 1/d_ij^2
        # grad U_i = sum_{j != i} 2 * (x_i - x_j) / d_ij^4
        
        # For 200 points, O(N^2) is 40k ops, which is fast.
        # Use vectorization for speed.
        
        # Pairwise differences
        # To avoid N^2 memory blowup for N=200 (40k floats is fine), we can do it directly.
        
        # diff[i, j] = x_i - x_j
        # d_sq[i, j] = ||x_i - x_j||^2
        
        # We compute the force on each point i from all j.
        # Force_i = sum_j 2 * (x_i - x_j) / d_ij^4
        
        # Efficient computation:
        # x_i (N, 1, 2) - x_j (1, N, 2) -> (N, N, 2)
        # This creates a 200x200x2 array, which is 80k floats. Very fast.
        
        xi = centers[:, np.newaxis, :] # (N, 1, 2)
        xj = centers[np.newaxis, :, :] # (1, N, 2)
        
        diff = xi - xj # (N, N, 2)
        d_sq = np.sum(diff * diff, axis=2) # (N, N)
        
        # Avoid division by zero
        d_sq[range(N), range(N)] = 1.0 # Diagonal is 0, set to 1 to avoid inf, mask later
        
        # Force term: 2 * diff / d_sq^2
        # We need d_sq^4 in denominator?
        # U = - 1/d^2 = - 1/d_sq
        # dU/dx_i = - d(1/d_sq)/dx_i = - (-1/d_sq^2) * d(d_sq)/dx_i = (1/d_sq^2) * 2*(x_i - x_j)
        # So Force = 2 * diff / d_sq^2
        
        inv_d_sq_sq = 1.0 / (d_sq * d_sq) # (N, N)
        
        # Weighted sum of diffs
        # Force_i = sum_j 2 * diff[i, j] * inv_d_sq_sq[i, j]
        # Multiply diff (N, N, 2) by inv_d_sq_sq (N, N, 1)
        weighted_diff = diff * inv_d_sq_sq[:, :, np.newaxis]
        
        # Sum over j (axis 1)
        grad = 2.0 * np.sum(weighted_diff, axis=1) # (N, 2)
        
        # Wall constraints:
        # Add a strong repulsive force from walls to keep points inside.
        # Soft wall potential: 100 * (0.5 - |x|)^2 if near edge?
        # Or simpler: if x > 0.5 - eps, push left.
        # Let's use a smooth wall potential: W(x) = 100 * max(0, |x| - 0.5 + 0.05)^2
        # Gradient: 200 * (|x| - 0.5 + 0.05) * sign(x) if |x| > 0.45
        
        margin = 0.05
        wall_force = np.zeros_like(centers)
        for k in range(2):
            # Right/Top wall
            mask_r = centers[:, k] > 0.5 - margin
            if np.any(mask_r):
                dist_to_wall = centers[mask_r, k] - (0.5 - margin)
                wall_force[mask_r, k] -= 200.0 * dist_to_wall # Push left/down
            # Left/Bottom wall
            mask_l = centers[:, k] < -0.5 + margin
            if np.any(mask_l):
                dist_to_wall = (-0.5 + margin) - centers[mask_l, k]
                wall_force[mask_l, k] += 200.0 * dist_to_wall # Push right/up
        
        grad += wall_force
        
        # Update velocity with momentum
        velocity = momentum * velocity + lr * grad
        
        # Update positions
        centers += velocity
        
        # Project back to square
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Check improvement
        r_trial = _compute_radius(centers)
        if r_trial > best_r:
            best_r = r_trial
            best_centers = centers.copy()
            
        # Decay learning rate
        if it % 100 == 0:
            lr *= 0.95
            
    return best_centers, best_r

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # 1. Start with a good hexagonal grid
    centers = _generate_hex_grid()
    r = _compute_radius(centers)
    
    # 2. Global exploration with Simulated Annealing
    # Budget: ~80 seconds
    if _time_left() > 40.0:
        centers, r = _simulated_annealing(centers, max_iter=50000)
        
    # 3. Local refinement with Projected Gradient Descent (Repulsive Potential)
    # Budget: ~50 seconds
    if _time_left() > 10.0:
        centers, r = _projected_gradient_refine(centers, max_iter=10000)
        
    # Final verification and clipping
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
