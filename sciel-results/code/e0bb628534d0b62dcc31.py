"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Place 50 equal non-overlapping circles in the unit square [-0.5, 0.5]^2
    to maximize the common radius.
    
    Uses Simulated Annealing (SA) to explore the configuration space.
    1. Initialize with random points.
    2. Iteratively perturb points.
    3. Accept moves based on Metropolis criterion to maximize the minimum radius.
    4. Cool the temperature to converge to a local optimum.
    """
    
    def score(points):
        """Calculate the maximum radius for a given configuration."""
        # Pairwise distances
        # Using broadcasting for vectorized calculation
        # diff shape: (N, N, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Minimum pairwise distance
        # Exclude diagonal (self-distance)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distances
        # Distance to nearest wall
        # x is in [-0.5, 0.5], so dist to wall is 0.5 - |x|
        dist_x = 0.5 - np.abs(points[:, 0])
        dist_y = 0.5 - np.abs(points[:, 1])
        min_boundary_dist = np.min(np.minimum(dist_x, dist_y))
        
        # Radius is half the min pairwise dist, limited by boundary dist
        # If a point is outside, dist_x or dist_y becomes negative, min_boundary_dist < 0
        # This naturally penalizes being outside.
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        # Ensure r is not negative for stability, though it shouldn't be if we handle it
        if r < 0:
            return -1.0 # Heavy penalty for being outside or overlapping significantly? 
                        # Actually, if min_pair_dist is 0 (coincident), r=0.
                        # If outside, min_boundary < 0, so r < 0.
                        # We want to maximize r.
        return r

    def anneal(initial_points, max_iters=5000, T_start=0.05, T_end=0.001, alpha=0.999):
        """Run Simulated Annealing."""
        current_points = initial_points.copy()
        current_score = score(current_points)
        best_points = current_points.copy()
        best_score = current_score
        
        T = T_start
        
        for i in range(max_iters):
            # Generate a perturbation
            # Perturb a random subset of points to allow local moves
            # Or perturb all points with small noise.
            # Let's try perturbing all points with Gaussian noise scaled by T
            noise = np.random.normal(0, T, size=(N, 2))
            proposed_points = current_points + noise
            
            # Clip to square to keep them valid? 
            # If we clip, the "score" function handles the boundary constraint.
            # However, clipping changes the geometry. 
            # Better to let them go outside and let the score penalize it.
            # But if they go too far, it's inefficient.
            # Let's clip to [-0.5, 0.5] to ensure we don't waste time on far-away points,
            # but the score function will still penalize being on the edge.
            # Actually, if we clip, a point at 0.6 becomes 0.5. 
            # The distance to boundary becomes 0. 
            # This is a valid move if it improves the score relative to the previous state.
            
            proposed_points = np.clip(proposed_points, -0.5, 0.5)
            
            proposed_score = score(proposed_points)
            
            # Metropolis acceptance
            if proposed_score > current_score:
                # Always accept better solutions
                current_points = proposed_points
                current_score = proposed_score
                
                if current_score > best_score:
                    best_points = current_points.copy()
                    best_score = current_score
            else:
                # Accept worse solutions with probability exp((new - old) / T)
                # Here we maximize score, so delta = proposed - current < 0
                # Prob = exp((proposed_score - current_score) / T)
                delta = proposed_score - current_score
                if delta < 0:
                    # If T is very small, this probability is ~0
                    p = np.exp(delta / T)
                    if np.random.rand() < p:
                        current_points = proposed_points
                        current_score = proposed_score
                
            # Cool down
            T *= alpha
            
            # Optional: If T is very small, we might be stuck. 
            # Restarting with a slightly perturbed best solution can help.
            if i % 1000 == 0 and i > 0:
                # Small kick to escape local minima
                kick = np.random.normal(0, 0.01, size=(N, 2))
                current_points = best_points + kick
                current_points = np.clip(current_points, -0.5, 0.5)
                current_score = score(current_points)
                
        return best_points

    # Multiple restarts to find a better global optimum
    np.random.seed(42)
    
    best_overall = None
    best_overall_score = -1.0
    
    # Run several independent annealing processes
    for restart in range(5):
        # Initialize with random points
        # To improve initial score, we can use a grid or random
        # Random is fine for SA if it runs long enough
        init_points = np.random.uniform(-0.4, 0.4, size=(N, 2))
        
        # Run annealing
        # Time budget: 120s total. 5 restarts -> 24s each.
        # 5000 iterations is fast with vectorized numpy.
        # Let's increase iterations if time permits, but keep it safe.
        # 5000 iters * 5 restarts = 25k iters.
        # Each iter does O(N^2) distance calc. N=50, N^2=2500.
        # 25k * 2500 = 62.5M ops. This is very fast in Python/NumPy.
        
        result = anneal(init_points, max_iters=10000, T_start=0.03, T_end=0.0001, alpha=0.9995)
        
        s = score(result)
        if s > best_overall_score:
            best_overall_score = s
            best_overall = result
            
    return best_overall

# EVOLVE-BLOCK-END
