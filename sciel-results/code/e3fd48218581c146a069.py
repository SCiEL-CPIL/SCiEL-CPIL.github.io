import numpy as np
import time
import math
from scipy.spatial.distance import cdist

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Grid resolution: 60x60 gives 3600 positions.
    # 200 points out of 3600.
    # Distance between grid points is 1/59.
    # If we place points on grid, min distance is at least 1/59 ~ 0.0169 (if adjacent).
    # We want to maximize min dist.
    # Using a grid allows fast distance calculation via integer coordinates.
    
    grid_size = 60
    grid_step = 1.0 / (grid_size - 1)
    
    # Precompute all grid points in [-0.5, 0.5]
    coords = np.linspace(-0.5, 0.5, grid_size)
    grid_points = np.array([(x, y) for y in coords for x in coords])
    n_points = len(grid_points)
    
    # Map from (row, col) to index
    # index = row * grid_size + col
    
    # Objective: Maximize r = min(min_pair_dist/2, min_wall_dist)
    # Since we are on a grid, wall dist is fixed for each point.
    # We can precompute wall distances for each grid point.
    wall_dists = 0.5 - np.abs(grid_points)
    min_wall_per_point = np.min(wall_dists, axis=1)
    
    # Initial configuration: random assignment of N points to distinct grid spots
    # Or a structured start (hexagonal-like) mapped to nearest grid point.
    
    def get_radius_from_indices(indices):
        """
        indices: array of shape (N,) with values in [0, n_points)
        Returns the radius r for this configuration.
        """
        pts = grid_points[indices]
        # Wall constraint
        min_wall = np.min(min_wall_per_point[indices])
        
        # Pairwise constraint
        # Use cdist for simplicity, but for speed we might want to optimize this.
        # With N=200, cdist is 200x200, which is fast.
        dist_matrix = cdist(pts, pts)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        
        r = min(min_wall, min_pair / 2.0)
        return r

    def sa_search(indices, time_limit):
        """
        Simulated Annealing on the set of occupied grid indices.
        Move: Pick a random point, move it to a random other grid spot (swap or insert).
        Actually, since points are distinct, we can pick a point i and a target index j.
        If j is occupied by k, swap i and k. If j is empty, move i to j (and free old spot).
        But maintaining "empty" spots requires tracking.
        Simpler: Pick i, pick j. If j is occupied by k, swap. If not, just move?
        No, we must maintain exactly N points.
        So: Pick i. Pick j. If j == i, skip.
        If j is occupied by k (k != i), swap positions of i and k.
        If j is NOT occupied, this implies we are moving i to j and... wait, if we just move i to j,
        we need to ensure we don't duplicate.
        
        Let's stick to swaps only to maintain the set size and distinctness easily?
        No, swaps restrict the search space significantly (permutation only).
        We need to be able to move a point to a new empty spot.
        
        Let's maintain a boolean array `occupied` of size n_points.
        State: `indices` (array of N indices), `occupied` (bool array).
        
        Move:
        1. Pick random point index i in [0, N).
        2. Pick random target grid index t in [0, n_points).
        3. If t is occupied by point k (k != i):
           Swap indices[i] and indices[k]. Update occupied.
        4. If t is NOT occupied:
           Move indices[i] to t. Free old spot. Update occupied.
        5. If t is occupied by i: No change (or skip).
        
        To efficiently check if t is occupied and by whom, we can use an inverse mapping array `inv_map` of size n_points,
        where inv_map[t] = point_index if occupied, else -1.
        """
        current_indices = indices.copy()
        current_inv_map = np.full(n_points, -1, dtype=np.int32)
        for i in range(N):
            current_inv_map[current_indices[i]] = i
            
        current_radius = get_radius_from_indices(current_indices)
        best_radius = current_radius
        best_indices = current_indices.copy()
        
        T = 0.001 # Initial temperature
        T_min = 1e-6
        cooling_rate = 0.9995
        
        rng = np.random.default_rng(42)
        
        while time.time() < start_time + time_limit and T > T_min:
            # Pick a random point
            i = rng.integers(0, N)
            # Pick a random target grid spot
            t = rng.integers(0, n_points)
            
            if t == current_indices[i]:
                T *= cooling_rate
                continue
                
            k = current_inv_map[t]
            
            if k == -1:
                # Target is empty. Move i to t.
                old_t = current_indices[i]
                
                # Calculate delta radius? 
                # Calculating full radius is O(N^2). 
                # With N=200, N^2 = 40000. 
                # 145s / 40000 ops ~ 3600 ops per second? No, numpy is fast.
                # Let's estimate: 1e5 evaluations per second? 
                # 145s * 1e5 = 1.45e7 evaluations.
                # This is plenty for SA.
                
                # Tentative move
                current_indices[i] = t
                current_inv_map[old_t] = -1
                current_inv_map[t] = i
                
                new_radius = get_radius_from_indices(current_indices)
                
                if new_radius > current_radius:
                    current_radius = new_radius
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_indices = current_indices.copy()
                else:
                    # Accept with probability exp((new - current) / T)
                    delta = new_radius - current_radius
                    if delta < 0:
                        accept_prob = math.exp(delta / T)
                        if rng.random() > accept_prob:
                            # Reject: revert
                            current_indices[i] = old_t
                            current_inv_map[t] = -1
                            current_inv_map[old_t] = i
                            current_radius = get_radius_from_indices(current_indices) # Restore exact value? 
                            # Actually we saved current_radius before move. But if we rejected, current_radius should be old value.
                            # Wait, if we rejected, the state is reverted, so current_radius is the old one.
                            # We didn't update current_radius in the "if new > curr" block if it was worse.
                            # So current_radius is still the value before the move. Correct.
            else:
                # Target is occupied by k. Swap i and k.
                old_i_t = current_indices[i]
                old_k_t = current_indices[k]
                
                current_indices[i] = old_k_t
                current_indices[k] = old_i_t
                current_inv_map[old_i_t] = k
                current_inv_map[old_k_t] = i
                
                new_radius = get_radius_from_indices(current_indices)
                
                if new_radius > current_radius:
                    current_radius = new_radius
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_indices = current_indices.copy()
                else:
                    delta = new_radius - current_radius
                    if delta < 0:
                        accept_prob = math.exp(delta / T)
                        if rng.random() > accept_prob:
                            # Revert
                            current_indices[i] = old_i_t
                            current_indices[k] = old_k_t
                            current_inv_map[old_i_t] = i
                            current_inv_map[old_k_t] = k
                            # current_radius remains unchanged (old value)
            
            T *= cooling_rate
            
        return best_indices, best_radius

    # Initialization:
    # 1. Random
    # 2. Grid-based hexagon approximation
    # Let's try a few initializations.
    
    best_overall_radius = 0.0
    best_overall_indices = None
    
    # Init 1: Random
    init_indices = rng.choice(n_points, size=N, replace=False)
    
    # Time budget for SA
    # We have ~140s.
    # Let's run SA for the remaining time.
    # But we might want to try multiple restarts if it's fast.
    # Given the cost of get_radius (O(N^2)), let's see how fast it is.
    
    # Quick test of speed
    t0 = time.time()
    _ = get_radius_from_indices(init_indices)
    t1 = time.time()
    time_per_eval = t1 - t0
    # print(f"Time per eval: {time_per_eval}")
    
    # If time_per_eval is very small, we can do many iterations.
    # If it's slow, we need to be careful.
    
    # Let's run one long SA chain.
    sa_time_limit = deadline - time.time() - 5.0 # 5s safety margin
    
    if sa_time_limit > 1.0:
        best_indices, best_radius = sa_search(init_indices, sa_time_limit)
        
        if best_radius > best_overall_radius:
            best_overall_radius = best_radius
            best_overall_indices = best_indices
    
    # If we have time, maybe try a second restart with different init?
    # For now, stick to one robust run.
    
    if best_overall_indices is None:
        # Fallback
        best_overall_indices = init_indices
        
    # Convert indices to coordinates
    centers = grid_points[best_overall_indices]
    
    # Final polish?
    # The grid solution is discrete. A local continuous optimization could improve it slightly.
    # But the grid step is 1/59 ~ 0.017. The optimal radius is ~0.036.
    # So we are likely 2-3 grid steps away from optimal spacing.
    # Continuous polish could recover the "fractional" parts.
    
    if time.time() < deadline - 2.0:
        # Simple gradient step or Nelder-Mead on the best solution?
        # Let's use a simple gradient ascent on the soft-min or just max-min.
        # Actually, let's just use the grid solution. It's structurally different and robust.
        # Adding a polish might be good but risky if it breaks validity.
        
        # Let's try a very light L-BFGS polish if time permits.
        from scipy.optimize import minimize
        
        def objective(x):
            c = x.reshape(-1, 2)
            if np.any(np.abs(c) > 0.5 + 1e-9):
                return 1e6
            wall_dists = 0.5 - np.abs(c)
            min_wall = np.min(wall_dists)
            dist_matrix = cdist(c, c)
            np.fill_diagonal(dist_matrix, np.inf)
            min_pair = np.min(dist_matrix)
            r = min(min_wall, min_pair / 2.0)
            return -r
        
        x0 = centers.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-10})
            if res.success or res.fun < -best_overall_radius:
                # Check validity
                c_new = res.x.reshape(-1, 2)
                if np.all(np.abs(c_new) <= 0.5 + 1e-9) and np.all(np.isfinite(c_new)):
                    r_new = -res.fun
                    if r_new > best_overall_radius:
                        centers = c_new
                        best_overall_radius = r_new
        except Exception:
            pass

    return centers

# EVOLVE-BLOCK-END
