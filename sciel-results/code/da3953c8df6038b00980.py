import numpy as np
import time
from scipy.optimize import minimize

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def is_valid(xy, r, tol=1e-9):
        n = len(r)
        if np.any(r < -tol): return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol): return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol): return False
        
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    def get_score(xy, r):
        return np.sum(r)

    def polish(xy_init, r_init, time_budget=10.0):
        """
        Local continuous optimization.
        We fix the topology (which circles are neighbors) established by the initial config.
        We optimize radii and positions to maximize sum(r) subject to non-overlap and boundary constraints.
        We use a penalty method or sequential quadratic programming logic simplified.
        Here we use a simple iterative projection method:
        1. Try to increase radii.
        2. If overlap occurs, push centers apart or shrink radii.
        """
        xy = xy_init.copy()
        r = r_init.copy()
        n = len(r)
        
        # We want to maximize sum(r).
        # Constraint: dist(i,j) >= r_i + r_j
        # Constraint: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        
        # Simple heuristic polish:
        # Iterate: 
        # 1. Calculate max possible radius for each circle given current positions and neighbors.
        # 2. If current radius < max possible, increase it.
        # 3. If overlap, resolve by moving centers (if space) or reducing radii.
        
        for step in range(200):
            if time.time() > start_time + time_budget:
                break
            
            # 1. Determine max radius for each circle based on boundaries and neighbors
            max_r = np.zeros(n)
            for i in range(n):
                # Boundary limit
                b_lim = min(xy[i, 0], 1 - xy[i, 0], xy[i, 1], 1 - xy[i, 1])
                max_r[i] = b_lim
                
                # Neighbor limit
                for j in range(n):
                    if i == j: continue
                    dx = xy[i, 0] - xy[j, 0]
                    dy = xy[i, 1] - xy[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    # dist >= r_i + r_j => r_i <= dist - r_j
                    lim = dist - r[j]
                    if lim < max_r[i]:
                        max_r[i] = lim
            
            # 2. Update radii towards max_r
            # We don't want to jump instantly to max_r as it might cause numerical issues 
            # if neighbors are also growing. We take a step.
            target_r = max_r
            # Ensure positive
            target_r = np.maximum(target_r, 0.01)
            
            # If we can increase, do so.
            # Simple approach: set r to max_r, then resolve overlaps by shrinking.
            r_new = target_r.copy()
            
            # 3. Resolve overlaps caused by the new radii
            # We do a few iterations of overlap resolution
            for _ in range(5):
                changed = False
                for i in range(n):
                    for j in range(i+1, n):
                        dx = xy[i, 0] - xy[j, 0]
                        dy = xy[i, 1] - xy[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        min_dist = r_new[i] + r_new[j]
                        
                        if dist < min_dist - 1e-9:
                            # Overlap exists. 
                            # Strategy: Shrink the smaller circle more, or split the difference.
                            # To maximize sum, we should shrink both proportionally to their radii?
                            # Or just shrink enough to separate.
                            overlap = min_dist - dist
                            # Reduce radii to remove overlap
                            # r_i' + r_j' = dist
                            # Let's reduce r_i and r_j such that the sum reduction is 'overlap'
                            # Distribute reduction proportional to current radii
                            total_r = r_new[i] + r_new[j]
                            if total_r < 1e-9: continue
                            red_i = overlap * (r_new[i] / total_r)
                            red_j = overlap * (r_new[j] / total_r)
                            
                            r_new[i] -= red_i
                            r_new[j] -= red_j
                            changed = True
                            
                            # Also try to push centers apart if possible
                            # If dist > 0, push i away from j and j away from i
                            if dist > 1e-9:
                                push = overlap * 0.5
                                # Check if pushing stays in bounds
                                # This is complex, so we just rely on radius shrinking for safety
                                # A more advanced polish would move centers.
                                # Let's try a small move.
                                f_vec = np.array([dx, dy]) / dist
                                # Move i by -f_vec * push/2, j by f_vec * push/2 ?
                                # No, that increases distance by push.
                                # But we must check bounds.
                                # For simplicity and robustness in this short polish, 
                                # we stick to radius adjustment which is always safe 
                                # as long as we don't go negative.
            
            r = r_new
            # Ensure non-negative
            r = np.maximum(r, 0.0)
            
            # Check if we improved
            # If no changes happened in the last few steps, we might be at a local max
            # But since we always try to grow to max_r, we should keep improving until stable.
            
        return xy, r

    def solve_mwis_grid(grid_size, time_budget):
        """
        1. Create a grid of candidate circles.
        2. For each cell, define a candidate circle centered there with a specific radius.
           To make it a proper MWIS, we need weights. 
           Weight = radius.
           But radius depends on neighbors. 
           Approximation: Assume all circles have the same radius 'r0' for the MWIS step.
           Then all weights are equal. We just need Max Independent Set.
           But we want to maximize SUM of radii. 
           
           Alternative: 
           Generate candidates with varying radii? 
           Let's stick to fixed radius candidates for the grid, select 32 non-overlapping ones.
           Then polish.
           
           To allow different radii, we can generate candidates with radii from a set {r1, r2, r3...}.
           However, a simpler and effective heuristic for "Sum of Radii" is:
           1. Find a dense packing of small circles.
           2. Merge/grow? No, N is fixed.
           
           Let's use a Grid of potential centers.
           We want to pick 32 centers such that they are far apart.
           This is equivalent to finding 32 points that maximize the minimum distance (maximin), 
           which roughly correlates with maximizing the sum of radii in a homogeneous packing.
           
           Greedy Maximin Selection:
           1. Start with empty set.
           2. Pick a point that maximizes the minimum distance to existing points.
           3. Repeat until 32 points.
           
           This is a known heuristic for circle packing.
        """
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        
        # Candidate pool: A grid of points
        # Higher resolution for better precision
        # 20x20 = 400 points. 
        # We need to pick 32.
        
        # To speed up, we can use a coarser grid first, then refine.
        # But 400 points is small enough for O(N^2) operations in Python loops if optimized?
        # 400*400 = 160k checks. 32 iterations. 32 * 160k = 5M ops. 
        # In Python, this might be slow if not vectorized.
        
        # Let's vectorize the maximin step.
        
        # Generate grid
        side = 20
        x = np.linspace(0.5/side, 1 - 0.5/side, side)
        y = np.linspace(0.5/side, 1 - 0.5/side, side)
        xx, yy = np.meshgrid(x, y)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])
        
        # We will select 32 points from this grid.
        selected = []
        remaining = candidates.copy()
        
        # Initial point: Center? Or random?
        # Random start to break symmetry
        start_idx = rng.randint(0, len(remaining))
        p = remaining[start_idx]
        selected.append(p)
        remaining = np.delete(remaining, start_idx, axis=0)
        
        for _ in range(N - 1):
            if time.time() > start_time + time_budget:
                break
                
            if len(remaining) == 0:
                break
                
            # Calculate distance from each remaining point to all selected points
            # remaining: (M, 2)
            # selected: (K, 2)
            
            # We want min_dist_to_selected for each remaining point.
            # dist_matrix: (M, K)
            # d_ij = || r_i - s_j ||
            
            # Vectorized distance calculation
            # r[:, None, :] - s[None, :, :]
            diffs = remaining[:, None, :] - np.array(selected)[None, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            
            # min distance for each remaining point
            min_dists = np.min(dists, axis=1)
            
            # Pick the point with max min_dist
            best_idx = np.argmax(min_dists)
            p = remaining[best_idx]
            selected.append(p)
            remaining = np.delete(remaining, best_idx, axis=0)
            
        if len(selected) < N:
            # Fallback if grid was too sparse or time ran out
            # Fill with random points
            while len(selected) < N:
                p = rng.uniform(0.1, 0.9, size=2)
                selected.append(p)
                
        xy = np.array(selected)
        
        # Initial radii: 
        # Set to half the min distance to nearest neighbor / 2? 
        # Or just a small value, let polish grow them.
        r = np.ones(N) * 0.05
        
        return xy, r

    best_result = None
    best_score = -np.inf
    
    # Run multiple instances of the Greedy Maximin + Polish strategy
    # The greedy choice depends on the starting point and tie-breaking.
    # We can randomize the grid slightly or the start point.
    
    n_trials = 20
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
            
        # Vary the grid resolution or start point for diversity
        # For simplicity, we just use different random seeds for the start point
        # by passing a time-based seed or trial-based seed
        
        # Modify solve_mwis_grid to accept a seed
        # But the function above uses time. Let's inline the logic or pass seed.
        
        # Let's just call a modified version
        # Actually, the current solve_mwis_grid uses time for seed. 
        # To ensure diversity, we can add a small random offset to the grid or start point.
        
        # Let's implement a local version with a specific seed
        def solve_mwis_grid_seeded(seed, grid_side=20):
            rng = np.random.RandomState(seed)
            
            # Add small jitter to grid to avoid perfect symmetry issues
            # Or just use standard grid
            x = np.linspace(0.5/grid_side, 1 - 0.5/grid_side, grid_side)
            y = np.linspace(0.5/grid_side, 1 - 0.5/grid_side, grid_side)
            xx, yy = np.meshgrid(x, y)
            candidates = np.column_stack([xx.ravel(), yy.ravel()])
            
            # Random start
            start_idx = rng.randint(0, len(candidates))
            p = candidates[start_idx]
            selected = [p]
            remaining = np.delete(candidates, start_idx, axis=0)
            
            for _ in range(N - 1):
                if len(remaining) == 0: break
                diffs = remaining[:, None, :] - np.array(selected)[None, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=2))
                min_dists = np.min(dists, axis=1)
                best_idx = np.argmax(min_dists)
                p = remaining[best_idx]
                selected.append(p)
                remaining = np.delete(remaining, best_idx, axis=0)
                
            if len(selected) < N:
                while len(selected) < N:
                    p = rng.uniform(0.1, 0.9, size=2)
                    selected.append(p)
                    
            xy = np.array(selected)
            r = np.ones(N) * 0.05
            return xy, r

        xy, r = solve_mwis_grid_seeded(42 + trial, grid_side=15 + (trial % 3) * 5)
        
        # Polish
        # Budget for polish: 5 seconds per trial
        # Total trials: 20. 20 * 5 = 100s. Plus overhead.
        # Deadline is 145s. Safe.
        
        xy_pol, r_pol = polish(xy, r, time_budget=5.0)
        
        # Check validity
        if not is_valid(xy_pol, r_pol):
            # Repair
            for scale in np.arange(1.0, 0.5, -0.05):
                r_test = r_pol * scale
                if is_valid(xy_pol, r_test):
                    r_pol = r_test
                    break
            else:
                continue
        
        score = get_score(xy_pol, r_pol)
        if score > best_score:
            best_score = score
            best_result = (xy_pol.copy(), r_pol.copy())
            
    if best_result is None:
        # Fallback grid
        side = 6
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
            r[i] = 0.5/side - 0.01
        return np.column_stack([xy, r])

    xy, r = best_result
    # Final check
    if not is_valid(xy, r):
         for scale in np.arange(1.0, 0.1, -0.01):
             r_test = r * scale
             if is_valid(xy, r_test):
                 r = r_test
                 break
                 
    return np.column_stack([xy, r])

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print("Score:", np.sum(res[:, 2]))
    print("Valid:", is_valid(res[:, :2], res[:, 2]))
