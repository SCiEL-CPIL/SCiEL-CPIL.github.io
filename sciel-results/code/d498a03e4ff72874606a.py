"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    best_score = 0.0
    best_points = None
    
    # Helper to calculate the score (minimum radius)
    def get_score(pts):
        # Pairwise distances
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        # Mask diagonal
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        # Boundary distance
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_pair, min_boundary)

    # Initial configuration: Hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            init_points = points[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98

    # Simulated Annealing Setup
    # We want to maximize the minimum distance. 
    # Energy function: We use a potential that penalizes close pairs heavily.
    # E = sum(1/d^2) for d < threshold, plus boundary penalty.
    # Actually, simpler: just use the "score" directly as a proxy for "goodness" 
    # and accept worse solutions probabilistically.
    # But SA usually minimizes energy. Let's minimize -score? No, -score is non-smooth.
    # Let's use a smooth energy function: E = sum(1/d^2) for pairs where d < 2*r_est 
    # and boundary violations.
    
    # A better energy function for packing: 
    # E = sum_{i<j} (1/d_ij^2) + sum_i (1/r_i^2) if outside? 
    # Let's stick to a standard repulsive force model.
    # Force on i from j: (1/d^2) * unit_vector(i-j)
    
    # We will run multiple restarts.
    current_pts = init_points.copy()
    current_score = get_score(current_pts)
    best_score = current_score
    best_points = current_pts.copy()
    
    # SA Parameters
    T_start = 0.05
    T_end = 0.0001
    alpha = 0.995  # Cooling rate
    # Number of iterations depends on time budget
    # Estimate time per iteration. N=50, so 1225 pairs. Vectorized numpy should be fast.
    
    # Pre-compute indices for pairs to speed up force calculation? 
    # For N=50, full matrix is fine.
    
    def compute_forces(pts):
        # Returns force vector for each point
        forces = np.zeros_like(pts)
        # Pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_sq = np.sum(diff * diff, axis=2)
        # Avoid division by zero
        d_sq[np.diag_indices_from(d_sq)] = np.inf
        d = np.sqrt(d_sq)
        
        # Repulsive force magnitude: 1/d^2
        # Only apply if d is small enough to be relevant? 
        # For packing, we want to push them apart.
        # Let's use a soft repulsion that is always active but strong when close.
        # F = 1/d^2
        inv_d_sq = 1.0 / d_sq
        # Force vector = inv_d_sq * diff
        # Sum over neighbors
        forces = np.sum(inv_d_sq[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary constraint
        # If a point is outside, pull it back hard.
        r = np.linalg.norm(pts, axis=1)
        outside = r > 1.0
        if np.any(outside):
            # Force towards center
            forces[outside] -= pts[outside] * 10.0 # Strong pull
            
        return forces

    # Main SA Loop
    # We will do a few restarts with different initial temperatures or perturbations
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            pts = init_points.copy()
        else:
            # Perturb best solution
            pts = best_points + np.random.randn(N, 2) * 0.05
        
        T = T_start * (1.0 + restart * 0.5) # Slightly higher T for later restarts? Or just randomize.
        # Actually, let's just run a standard SA schedule.
        T = T_start
        
        # Estimate iterations based on time
        # Let's do a fixed number of iterations per restart, say 5000, and check time.
        max_iters = 5000
        
        for i in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Compute forces
            F = compute_forces(pts)
            
            # Propose move
            # Move each particle by a small random step + force step
            # step_size = 0.01 + T * 0.1 ?
            step_size = 0.01
            new_pts = pts + F * 0.001 + np.random.randn(N, 2) * step_size
            
            # Check validity (hard constraint)
            # If any point is far outside (e.g., > 1.1), reject immediately
            if np.any(np.linalg.norm(new_pts, axis=1) > 1.1):
                T *= alpha
                continue
                
            new_score = get_score(new_pts)
            curr_score = get_score(pts)
            
            # Acceptance probability
            # We want to maximize score.
            # If new_score > curr_score, accept.
            # Else, accept with prob exp((new_score - curr_score) / T)
            delta = new_score - curr_score
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                pts = new_pts
                curr_score = new_score
                if curr_score > best_score:
                    best_score = curr_score
                    best_points = pts.copy()
                    
            T *= alpha
            if T < T_end:
                break

    return best_points
# EVOLVE-BLOCK-END
