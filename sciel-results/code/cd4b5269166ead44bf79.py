import numpy as np

N = 300

# EVOLVE-BLOCK-START
def construct():
    # Approach: Simulated Annealing with Random Initialization
    # This is distinct from the Sunflower Spiral + Hill Climbing approach.
    
    # 1. Initialization: Random points inside the unit circle
    # Use rejection sampling or polar coordinates
    # Polar: r = sqrt(U(0,1)), theta = U(0, 2pi)
    rng = np.random.default_rng(42) # Fixed seed for reproducibility, though random is fine
    radii = np.sqrt(rng.random(N))
    thetas = 2.0 * np.pi * rng.random(N)
    points = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    
    def calc_score(pts):
        # Calculate the minimum radius r
        # Boundary constraint
        dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(dists_boundary)
        
        # Pairwise constraint
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)

    current_score = calc_score(points)
    
    # 2. Simulated Annealing
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    step_size = 0.05
    max_iter = 8000
    no_improve = 0
    best_score = current_score
    best_points = points.copy()
    
    for it in range(max_iter):
        # Pick a random point to move
        idx = rng.integers(N)
        
        # Generate a random perturbation
        # Gaussian noise
        noise = rng.normal(0, step_size, 2)
        new_points = points.copy()
        new_points[idx] += noise
        
        # Ensure the moved point stays within the container (optional, but helps stability)
        # Actually, let the score function handle boundary violations by lowering the score.
        # But if it goes way out, the score drops a lot.
        
        new_score = calc_score(new_points)
        
        # Acceptance criterion
        if new_score >= current_score:
            points = new_points
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Metropolis criterion
            delta = new_score - current_score
            # exp(delta / T)
            # Since delta is negative, exp is < 1
            if T > 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    points = new_points
                    current_score = new_score
            no_improve += 1
            
        # Cool down
        T *= cooling_rate
        # Adapt step size: reduce it as we cool
        step_size *= 0.999
        
        if T < T_min:
            break
            
    # 3. Local Polish (optional, but helps fine-tuning)
    # We can run a few iterations of a simple local search to tighten the best solution found
    # This is similar to the seed's hill climbing but applied to the SA result
    # To strictly follow "do not reuse current designs", I should avoid the exact same hill climbing logic.
    # However, a simple gradient-free local refinement is standard.
    # Let's do a simple "push apart" local refinement for a few steps.
    
    pts = best_points.copy()
    for _ in range(200):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_dist_val = np.min(dists)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        
        dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary_val = np.min(dists_boundary)
        min_boundary_idx = np.argmin(dists_boundary)
        
        if min_dist_val / 2.0 < min_boundary_val:
            vec = pts[i] - pts[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                direction = vec / norm
                pts[i] += direction * 0.005
                pts[j] -= direction * 0.005
        else:
            idx = min_boundary_idx
            vec = -pts[idx]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                direction = vec / norm
                pts[idx] += direction * 0.005
                
        # Project back
        norms = np.linalg.norm(pts, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            pts[outside] /= norms[outside, np.newaxis]
            
    final_score = calc_score(pts)
    if final_score > best_score:
        return pts
    else:
        return best_points

# EVOLVE-BLOCK-END
