"""Improved circle packing for N=300 in unit circle using Simulated Annealing."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the packing score r for given points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        # Check all points are inside unit circle
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii >= 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        return min(min_pair, min_boundary)

    def initialize_random():
        """Initialize points randomly inside the unit circle."""
        # Rejection sampling to get uniform distribution in circle
        points = np.zeros((N, 2))
        for i in range(N):
            while True:
                x, y = np.random.uniform(-1, 1, 2)
                if x*x + y*y <= 1.0:
                    points[i] = [x, y]
                    break
        return points

    def simulated_annealing(pts, deadline):
        """
        Simulated Annealing to maximize the minimum distance.
        """
        current_pts = pts.copy()
        current_score = compute_score(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score
        
        # Temperature parameters
        T = 0.05
        T_min = 0.001
        cooling_rate = 0.99995
        
        # Move size
        move_size = 0.1
        
        it = 0
        while time.time() < deadline:
            it += 1
            
            # Select a random point to move
            idx = np.random.randint(N)
            orig_pt = current_pts[idx].copy()
            
            # Generate a random move
            # Scale move size with temperature?
            # Or keep it constant and adjust acceptance?
            # Let's keep move size adaptive based on T
            step = np.random.randn(2) * (move_size * T)
            
            new_pt = orig_pt + step
            
            # Project inside unit circle
            r = np.hypot(new_pt[0], new_pt[1])
            if r > 1.0:
                new_pt = new_pt / r
            
            current_pts[idx] = new_pt
            
            new_score = compute_score(current_pts)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_pts = current_pts.copy()
            else:
                # Revert
                current_pts[idx] = orig_pt
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                # Reheat or restart?
                # Let's just keep T at T_min to continue local search
                T = T_min
                
        return best_pts, best_score

    # Main execution
    best_pts = None
    best_score = 0.0
    
    # Try a few random initializations if time permits
    # But SA is stochastic, so one long run is often better than many short ones
    # However, starting from a better configuration helps.
    
    # Strategy: 
    # 1. Start with random initialization.
    # 2. Run SA for the remaining time.
    
    pts = initialize_random()
    best_pts, best_score = simulated_annealing(pts, deadline)
    
    # Final check
    final_score = compute_score(best_pts)
    if final_score > best_score:
        best_score = final_score
        
    # Ensure validity
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_pts)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_pts = best_pts * scale
        
    return best_pts
# EVOLVE-BLOCK-END
