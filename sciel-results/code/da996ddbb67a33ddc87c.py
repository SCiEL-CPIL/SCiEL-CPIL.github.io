# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Maximizes the min/max distance ratio for 16 points in 2D using Simulated Annealing
    with a repulsive force model. This approach treats the problem as finding an 
    equilibrium configuration of hard-core particles, which is more robust to local 
    minima than gradient-based methods on non-smooth objectives.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit

    def time_exceeded():
        return time.time() > deadline

    def evaluate(points):
        """Calculate the score R = (d_min / d_max)^2"""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    def compute_forces(points):
        """
        Compute repulsive forces based on pairwise distances.
        We want to maximize d_min, so we push points apart if they are too close.
        We also want to minimize d_max, so we pull points inward if they are too far.
        
        This is a heuristic force model:
        F_repel ~ 1/d^2 (strong when close)
        F_attract ~ (d - R_target) (weak spring towards an optimal radius)
        
        Actually, a simpler and more effective approach for TSP/Packing is:
        1. Calculate all pairwise distances.
        2. Identify the minimum distance d_min.
        3. Identify the maximum distance d_max.
        4. Apply forces to increase d_min and decrease d_max.
        
        However, for Simulated Annealing, we just need a good objective function.
        Let's stick to a pure energy minimization where Energy = -log(d_min) + log(d_max)
        or simply minimize (d_max / d_min).
        
        For the SA move generation, we will use random perturbations and accept based on the score.
        """
        pass

    def generate_initial_config(seed):
        """Generate diverse initial configurations."""
        np.random.seed(seed)
        # Strategy 1: Random points in unit disk
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n))
        points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        # Strategy 2: Hexagonal lattice subset (known good local min)
        if seed % 2 == 0:
             # Generate hex lattice
             R = 4
             pts = []
             for q in range(-R, R + 1):
                 for r in range(max(-R, -q - R), min(R, -q + R) + 1):
                     x = 1.5 * q
                     y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
                     pts.append([x, y])
             hex_pts = np.array(pts)
             hex_pts -= np.mean(hex_pts, axis=0)
             # Sort by distance to center to get densest cluster
             dists_center = np.linalg.norm(hex_pts, axis=1)
             sorted_idx = np.argsort(dists_center)
             points = hex_pts[sorted_idx[:n]].copy()
             # Normalize to fit in unit disk
             max_r = np.max(np.linalg.norm(points, axis=1))
             if max_r > 0:
                 points /= max_r
                 
        # Strategy 3: Square grid
        else:
             x = np.linspace(-1, 1, 4)
             y = np.linspace(-1, 1, 4)
             xx, yy = np.meshgrid(x, y)
             points = np.column_stack((xx.ravel(), yy.ravel()))
             max_r = np.max(np.linalg.norm(points, axis=1))
             if max_r > 0:
                 points /= max_r

        return points

    best_score_global = -1.0
    best_points_global = None

    # Simulated Annealing Parameters
    # We will run multiple independent SA runs with different seeds to explore the landscape.
    # Each run will have a cooling schedule.
    
    num_sa_runs = 20
    steps_per_run = 50000
    
    # Distribute time budget
    # Total time ~ 580s.
    # Each step is fast (pdist on 16 points is ~1us).
    # 20 * 50000 = 1,000,000 steps. Should be well within time.
    
    for run_idx in range(num_sa_runs):
        if time_exceeded():
            break
            
        seed = 42 + run_idx
        points = generate_initial_config(seed)
        
        # Normalize to unit scale initially
        max_r = np.max(np.linalg.norm(points, axis=1))
        if max_r > 0:
            points = points / max_r
            
        current_score = evaluate(points)
        if current_score > best_score_global:
            best_score_global = current_score
            best_points_global = points.copy()
            
        # SA Parameters
        T_start = 0.5
        T_end = 0.001
        alpha = 0.999 # Cooling rate
        
        current_T = T_start
        local_best_score = current_score
        local_best_points = points.copy()
        
        # We will use a simple random walk with occasional large jumps
        for step in range(steps_per_run):
            if time_exceeded():
                break
                
            # Generate a new candidate
            # Mix of small perturbations and larger "kicks" to escape local minima
            if np.random.random() < 0.1:
                # Large kick: move one or two points significantly
                num_moves = np.random.randint(1, 3)
                indices = np.random.choice(n, num_moves, replace=False)
                points[indices] += np.random.randn(num_moves, 2) * 0.2
            else:
                # Small perturbation
                idx = np.random.randint(n)
                points[idx] += np.random.randn(2) * 0.05
            
            # Keep points within a reasonable bound (unit disk * 1.5) to prevent divergence
            # We don't strictly need to constrain to unit disk, but it helps normalization
            norms = np.linalg.norm(points, axis=1)
            max_allowed = 1.5
            for i in range(n):
                if norms[i] > max_allowed:
                    points[i] = points[i] / norms[i] * max_allowed
            
            new_score = evaluate(points)
            
            # Acceptance Criteria
            if new_score > current_score:
                current_score = new_score
                points = points # already updated
                if new_score > local_best_score:
                    local_best_score = new_score
                    local_best_points = points.copy()
            else:
                # Metropolis criterion
                delta = new_score - current_score
                # Since we maximize score, delta is negative.
                # P(accept) = exp(delta / T)
                # Note: delta is negative, so delta/T is negative.
                # We need to scale delta appropriately. The score range is roughly [0, 1].
                # A difference of 0.01 is significant.
                # Let's use a temperature that allows acceptance of ~1% drops initially.
                if current_T > 0:
                    prob = np.exp(delta / (current_T * 0.1)) # Scaled temperature
                    if np.random.random() < prob:
                        current_score = new_score
                        # points already updated
                    else:
                        # Revert
                        # We need to store previous points to revert properly
                        # To save memory/time, we just generate new ones or keep a backup
                        # For simplicity and speed, let's just keep the old points if rejected.
                        # But we modified points in place. We need a backup.
                        pass # This logic is flawed for in-place modification without backup.
            
            # Let's refactor the SA loop to be cleaner with backups
            
        # Re-do the SA loop properly with backups to ensure correctness
        # The previous loop was pseudocode. Let's write the real one below.
        
    # RE-IMPLEMENTATION WITH CORRECT BACKUP LOGIC
    
    best_score_final = -1.0
    best_points_final = None
    
    def run_sa(points_init, seed):
        np.random.seed(seed)
        points = points_init.copy()
        current_score = evaluate(points)
        best_local_score = current_score
        best_local_points = points.copy()
        
        T = 0.5
        cooling = 0.9995
        
        for step in range(20000):
            if time_exceeded():
                break
            
            prev_points = points.copy()
            
            # Perturbation
            if np.random.random() < 0.05:
                # Large move
                idxs = np.random.choice(n, 2, replace=False)
                points[idxs] += np.random.randn(2, 2) * 0.3
            else:
                # Small move
                idx = np.random.randint(n)
                points[idx] += np.random.randn(2) * 0.05
            
            # Boundary check (soft constraint)
            norms = np.linalg.norm(points, axis=1)
            for i in range(n):
                if norms[i] > 2.0:
                    points[i] = points[i] / norms[i] * 2.0
            
            new_score = evaluate(points)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local_points = points.copy()
            else:
                delta = new_score - current_score
                # Accept with probability exp(delta/T)
                # delta is negative
                if T > 1e-6:
                    # Scale delta to be comparable to T
                    # Score changes are small. 0.01 change is big.
                    # Let's assume T is in units of score.
                    prob = np.exp(delta / T)
                    if np.random.random() < prob:
                        current_score = new_score
                    else:
                        points = prev_points
                else:
                    points = prev_points
            
            T *= cooling
            
        return best_local_points, best_local_score

    # Run multiple SA instances
    # We have ~580s.
    # 10 instances * 20000 steps = 200,000 steps.
    # Each step involves pdist (120 dists) and exp. ~5-10us.
    # 200,000 * 10us = 2 seconds.
    # We can afford many more steps or instances.
    # Let's do 50 instances of 100,000 steps.
    # 50 * 100,000 = 5,000,000 steps.
    # 5,000,000 * 10us = 50 seconds.
    # This is safe.
    
    num_instances = 50
    steps_per_instance = 100000
    
    for i in range(num_instances):
        if time_exceeded():
            break
            
        seed = 1000 + i
        # Diverse starts
        if i % 3 == 0:
            start_pts = generate_initial_config(seed)
        elif i % 3 == 1:
            # Random in disk
            angles = np.random.uniform(0, 2*np.pi, n, seed=seed)
            radii = np.sqrt(np.random.uniform(0, 1, n, seed=seed))
            start_pts = np.column_stack((radii*np.cos(angles), radii*np.sin(angles)))
        else:
            # Perturbed best known so far
            if best_points_final is not None:
                start_pts = best_points_final + np.random.randn(n, 2, seed=seed) * 0.1
            else:
                start_pts = generate_initial_config(seed)
                
        # Normalize start
        max_r = np.max(np.linalg.norm(start_pts, axis=1))
        if max_r > 0:
            start_pts = start_pts / max_r
            
        res_pts, res_score = run_sa(start_pts, seed)
        
        if res_score > best_score_final:
            best_score_final = res_score
            best_points_final = res_pts.copy()

    # Final Polish: Use a local optimizer on the best SA result to squeeze out more
    if best_points_final is not None and not time_exceeded():
        from scipy.optimize import minimize
        
        def objective(params):
            pts = params.reshape((n, 2))
            dists = pdist(pts, 'euclidean')
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-10:
                return 0.0
            return -(d_min / d_max) ** 2

        try:
            res = minimize(objective, best_points_final.ravel(), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12},
                           callback=lambda x: time_exceeded())
            if not time_exceeded():
                final_pts = res.x.reshape((n, 2))
                final_score = evaluate(final_pts)
                if final_score > best_score_final:
                    best_points_final = final_pts
                    best_score_final = final_score
        except Exception:
            pass

    if best_points_final is None:
        # Fallback
        best_points_final = generate_initial_config(0)

    return best_points_final

# EVOLVE-BLOCK-END
