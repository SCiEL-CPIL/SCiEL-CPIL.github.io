import numpy as np
import time
from scipy.spatial.distance import cdist

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def compute_radius(pts):
        """Compute the maximum radius for given points."""
        if len(pts) != N:
            return 0.0
        # Check bounds
        if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5):
            return 0.0
        if np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        # Pairwise distances
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Boundary distances
        min_boundary = np.min(np.minimum(
            np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
            np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
        ))
        
        return min(min_pair, min_boundary)
    
    def make_initial_config():
        """Create a good initial configuration."""
        # Start with 7x7 grid, remove center, add 2 points near center
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        
        # Remove center point (index 24 for 7x7)
        center_idx = 24
        grid = np.delete(grid, center_idx, axis=0)  # Now 48 points
        
        # Add 2 points near center with slight offset
        offset = 0.06
        pts = np.vstack([
            grid,
            [[-offset, 0]],
            [[offset, 0]]
        ])
        
        # Jitter slightly to break symmetry
        np.random.seed(42)
        pts += np.random.randn(N, 2) * 0.001
        pts = np.clip(pts, -0.5 + 0.01, 0.5 - 0.01)
        
        return pts
    
    def simulated_annealing(pts, max_time):
        """Run simulated annealing to optimize the packing."""
        current_pts = pts.copy()
        current_r = compute_radius(current_pts)
        best_pts = current_pts.copy()
        best_r = current_r
        
        # Temperature schedule
        T = 0.05
        T_min = 1e-8
        cooling_rate = 0.9995
        step_scale = 0.03
        
        while time.time() < max_time:
            # Generate a random move
            idx = np.random.randint(N)
            delta = np.random.randn(2) * step_scale
            new_pts = current_pts.copy()
            new_pts[idx] += delta
            
            # Keep in bounds
            new_pts[idx, 0] = np.clip(new_pts[idx, 0], -0.5, 0.5)
            new_pts[idx, 1] = np.clip(new_pts[idx, 1], -0.5, 0.5)
            
            new_r = compute_radius(new_pts)
            
            # Acceptance criterion
            if new_r >= current_r:
                current_pts = new_pts
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
            else:
                # Metropolis acceptance
                delta_r = current_r - new_r
                if delta_r < T:
                    prob = np.exp(-delta_r / T)
                    if np.random.random() < prob:
                        current_pts = new_pts
                        current_r = new_r
            
            # Cool down
            T = max(T * cooling_rate, T_min)
            
            # Occasionally try larger moves
            if np.random.random() < 0.05:
                step_scale = 0.01 + 0.05 * np.random.random()
        
        return best_pts, best_r
    
    # Run multiple restarts
    best_pts = None
    best_r = 0.0
    
    # First run: good initial config
    pts0 = make_initial_config()
    r0 = compute_radius(pts0)
    
    # Run SA on the initial config
    deadline_for_sa = min(deadline, start_time + 100.0)
    pts_sa, r_sa = simulated_annealing(pts0, deadline_for_sa)
    
    if r_sa > best_r:
        best_r = r_sa
        best_pts = pts_sa
    
    # If time remains, try a few more restarts with different inits
    while time.time() < deadline - 10.0:
        # Random init
        np.random.seed(int(time.time() * 1000) % 10000)
        pts_rand = np.random.rand(N, 2) - 0.5
        pts_rand = np.clip(pts_rand, -0.45, 0.45)
        
        # Short SA run
        remaining = deadline - time.time() - 5.0
        if remaining < 5.0:
            break
        
        pts_try, r_try = simulated_annealing(pts_rand, min(deadline, time.time() + remaining))
        
        if r_try > best_r:
            best_r = r_try
            best_pts = pts_try
    
    # Final local polish on best solution
    if time.time() < deadline - 5.0 and best_pts is not None:
        # Do a fine-tuning pass with very small steps
        T = 1e-6
        step_scale = 0.005
        current_pts = best_pts.copy()
        current_r = best_r
        
        end_polish = min(deadline, time.time() + 8.0)
        while time.time() < end_polish:
            idx = np.random.randint(N)
            delta = np.random.randn(2) * step_scale
            new_pts = current_pts.copy()
            new_pts[idx] += delta
            new_pts[idx, 0] = np.clip(new_pts[idx, 0], -0.5, 0.5)
            new_pts[idx, 1] = np.clip(new_pts[idx, 1], -0.5, 0.5)
            
            new_r = compute_radius(new_pts)
            
            if new_r >= current_r:
                current_pts = new_pts
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
            else:
                delta_r = current_r - new_r
                if delta_r < T:
                    prob = np.exp(-delta_r / T)
                    if np.random.random() < prob:
                        current_pts = new_pts
                        current_r = new_r
        
        if best_pts is None:
            best_pts = current_pts
    
    # Ensure best_pts is valid
    if best_pts is None:
        best_pts = pts0
    
    return best_pts
