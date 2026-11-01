"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(centers):
    """Compute the radius score for a set of centers."""
    # Clip to ensure validity
    c = np.clip(centers, -0.5, 0.5)
    # Min distance to boundary
    dist_boundary = np.min(0.5 - np.abs(c), axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Min pairwise distance
    # Using scipy's cKDTree for efficiency
    from scipy.spatial import cKDTree
    tree = cKDTree(c)
    # Query k=2 to get the nearest neighbor for each point
    # The distance to the nearest neighbor is the 2nd value (index 1) in the result
    # But we need the min over all pairs.
    # Actually, cKDTree.query(c, k=2) returns distances to the 2 nearest neighbors (including self).
    # So distances[:, 1] are the distances to the nearest neighbor for each point.
    # The min of these is the min pairwise distance.
    try:
        dists, _ = tree.query(c, k=2)
        min_pair_dist = np.min(dists[:, 1])
    except Exception:
        # Fallback to brute force if scipy fails or something is wrong
        n = len(c)
        min_d = 1e9
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(c[i] - c[j])
                if d < min_d:
                    min_d = d
        min_pair_dist = min_d
    
    r = min(min_boundary, min_pair_dist / 2.0)
    return r


def _hex_init(n):
    """Generate a hexagonal packing initial configuration."""
    # Approximate hexagonal packing in a square
    # The optimal density is pi / (2 * sqrt(3)) ~ 0.9069
    # For N=50, area per circle ~ 1/50.
    # Radius r ~ sqrt(1 / (50 * 0.9069 * 4/pi)) ? No, simpler:
    # Let's just generate a grid with hex offset.
    
    # Start with a rough estimate of spacing
    # If we pack N circles in area 1, density ~ 0.9
    # Area per circle ~ 1/50 = 0.02
    # Hex cell area = r^2 * 2 * sqrt(3)
    # 0.02 = r^2 * 3.464 => r^2 = 0.00577 => r = 0.076
    # Spacing d = 2*r = 0.152
    
    # Let's try to fit a hex grid.
    # Rows alternate offset.
    # Row height = d * sqrt(3)/2
    # Col width = d
    
    # We need to find a grid size that fits at least N points.
    # Let's try increasing grid size until we have enough points.
    
    best_centers = None
    best_score = 0
    
    # Try a range of spacings
    # The optimal radius is ~0.07137. So spacing ~0.1427.
    # Let's search around this.
    
    for d in np.linspace(0.13, 0.16, 20):
        row_h = d * np.sqrt(3) / 2
        # Number of rows that fit in height 1
        # y from -0.5 to 0.5
        # y_i = -0.5 + r + i * row_h ... 
        # Actually, let's just generate points on a hex lattice and pick the best N.
        
        # Generate a large enough grid
        # x from -0.5 to 0.5
        # y from -0.5 to 0.5
        
        # Step in x is d
        # Step in y is row_h
        # Offset for odd rows is d/2
        
        # Let's generate points
        points = []
        
        # Number of rows
        # y = -0.5 + margin + i * row_h
        # Let's just iterate i
        i = 0
        y = -0.5 + d/2 # Start a bit inside
        while y < 0.5:
            x_start = -0.5 + d/2 + (d/2 if i % 2 else 0)
            j = 0
            x = x_start
            while x < 0.5:
                points.append([x, y])
                x += d
                j += 1
            y += row_h
            i += 1
        
        if len(points) < N:
            continue
            
        points = np.array(points)
        
        # We have more than N points. We need to select N points to maximize min distance.
        # This is a hard problem.
        # Heuristic: pick points that are spread out.
        # Or, just take the first N? No, that biases to corner.
        # Better: Randomly select N from the grid, or use a greedy farthest-point sampling?
        # Given time constraints, let's just try to center the grid.
        
        # Center of mass of points
        com = np.mean(points, axis=0)
        points -= com
        
        # Now clip to square? No, we want them inside.
        # The grid generation was within bounds roughly.
        # Let's just take the N points closest to the center?
        # Or random subset?
        
        # Let's try a few random subsets
        best_local_score = 0
        best_local_centers = None
        
        for _ in range(10):
            idx = np.random.choice(len(points), N, replace=False)
            subset = points[idx]
            # Clip to ensure inside
            subset = np.clip(subset, -0.5, 0.5)
            s = _score(subset)
            if s > best_local_score:
                best_local_score = s
                best_local_centers = subset
        
        if best_local_score > best_score:
            best_score = best_local_score
            best_centers = best_local_centers.copy()
            
    if best_centers is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_centers


def _simulate_anneal(centers, t_start, t_end, n_iter, time_limit):
    """Run simulated annealing to improve the packing."""
    c = centers.copy()
    current_score = _score(c)
    best_score = current_score
    best_c = c.copy()
    
    t = t_start
    cooling = (t_end / t_start) ** (1.0 / n_iter)
    
    start_time = time.time()
    
    for i in range(n_iter):
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Random perturbation
        # Move one or a few points
        # Let's move a random point by a random amount
        idx = np.random.randint(N)
        step = t * 10 # Scale step with temperature?
        # Actually, step size should be related to the scale of the problem.
        # Let's use a fixed step size that shrinks.
        step = 0.05 * (t / t_start) ** 0.5
        
        delta = np.random.uniform(-step, step, 2)
        
        old_pos = c[idx].copy()
        c[idx] += delta
        
        # Keep inside square
        c[idx] = np.clip(c[idx], -0.5, 0.5)
        
        new_score = _score(c)
        
        # Acceptance criterion
        if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / t):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        else:
            c[idx] = old_pos
        
        t *= cooling
        
    return best_c, best_score


def construct():
    start_time = time.time()
    time_limit = 140  # 120s limit, keep 20s margin
    
    # 1. Generate initial hexagonal configuration
    centers = _hex_init(N)
    current_score = _score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # 2. Run Simulated Annealing
    # We can run multiple restarts if time permits
    t_start = 0.05
    t_end = 1e-7
    n_iter = 50000
    
    # Estimate time per iteration
    # _score is O(N^2) or O(N log N). N=50 is small.
    # 50000 iterations should be fast.
    
    while time.time() - start_time < time_limit - 5:
        # Run one SA pass
        # Use a fraction of the remaining time
        elapsed = time.time() - start_time
        remaining = time_limit - 5 - elapsed
        if remaining <= 0:
            break
            
        # How many iterations can we do?
        # Let's just run a fixed number and check time
        # If it takes too long, we'll break inside the loop.
        
        # Adjust n_iter based on remaining time?
        # For now, fixed 50k.
        
        new_centers, new_score = _simulate_anneal(best_centers, t_start, t_end, n_iter, remaining)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers
        
        # If we improved, we might want to restart from the new best?
        # Or continue from the current state?
        # Usually, restarting from the best found so far is good.
        
        # Check if we are close to the theoretical max?
        # Best known: 0.071377103865
        if best_score > 0.07137:
            # We are very close, try to refine with smaller steps?
            # Or just stop?
            # Let's continue to see if we can beat it slightly.
            pass
            
    return best_centers
# EVOLVE-BLOCK-END
