"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog
import time

N = 26

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Initial configuration: a 5x5 grid + 1 extra, take first 26
    # Actually, let's use a better initial layout.
    # 5x5 grid gives 25 circles. We need 26.
    # Let's try a 5x5 grid with one circle in the center? No, that would overlap.
    # Better: use a hexagonal packing or a jittered grid.
    # For now, start with a 5x5 grid of 25 circles, and add one more?
    # The task says "Pack 26 circles".
    # A 5x5 grid has 25 cells. The 26th circle would have to be smaller or we need a different layout.
    # Let's use a 5x5 grid for the first 25, and place the 26th in the center of one of the cells,
    # or better, use a layout that naturally fits 26.
    # A common approach is to start with a grid and then optimize.
    # Let's initialize with a 5x5 grid (25 circles) and one circle in the middle of the square,
    # but that will overlap.
    # Alternative: 5x5 grid, but the 26th circle is very small initially?
    # Or, use a 6x5 grid? 6*5=30. Take first 26.
    # Let's use a 6x5 grid.
    side_x = 6
    side_y = 5
    # Total 30 positions. We take the first 26.
    # This might be uneven.
    # Let's try a more symmetric approach.
    # How about 26 circles in a 5x5 grid, with the center cell split into 2? No.
    # Let's just use a 5x5 grid for 25 circles, and add the 26th at (0.5, 0.5) with a tiny radius,
    # then let the optimizer move it.
    
    # Actually, a better initial guess is a hexagonal packing.
    # But for simplicity, let's start with a jittered grid.
    
    # Let's use 5x5 grid for 25 circles, and the 26th circle at the center of the square.
    # This will cause initial overlap, but the optimizer should fix it.
    
    def initial_layout():
        positions = []
        radii = []
        # 5x5 grid
        for i in range(5):
            for j in range(5):
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                positions.append((x, y))
                radii.append(0.5 / 5.0 - 1e-3)
        # 26th circle at center
        positions.append((0.5, 0.5))
        radii.append(0.5 / 5.0 - 1e-3)
        return np.array(positions), np.array(radii)
    
    # Alternative: use a 6x5 grid, take first 26
    def initial_layout_grid():
        positions = []
        radii = []
        side_x = 6
        side_y = 5
        for i in range(side_x):
            for j in range(side_y):
                x = (i + 0.5) / side_x
                y = (j + 0.5) / side_y
                positions.append((x, y))
                radii.append(0.5 / max(side_x, side_y) - 1e-3)
        return np.array(positions[:N]), np.array(radii[:N])
    
    # Let's use the 6x5 grid approach, it's simpler and avoids the center overlap issue initially.
    pos, rad = initial_layout_grid()
    
    # Now, we will optimize using a simple gradient-based method or an iterative radius adjustment.
    # Given the time constraint and the hint about LP, let's try a simpler approach first:
    # Iteratively adjust positions and radii to maximize sum of radii.
    
    # However, the hint says "Graph-based LP radius optimization... should yield tighter packings".
    # Let's implement a simplified version of that.
    
    # Step 1: Optimize positions to maximize minimum distance between circles.
    # This is a non-convex problem. We can use gradient ascent on the minimum distance.
    
    # Let's use a simple gradient ascent with random restarts if time permits.
    
    best_score = np.sum(rad)
    best_pos = pos.copy()
    best_rad = rad.copy()
    
    # Gradient ascent on positions to maximize the minimum pairwise distance
    # and the distance to the boundaries.
    
    def min_distance(pos, rad):
        # Compute minimum distance between any two circles (center distance - r_i - r_j)
        # and distance to boundary (min(x - r, 1 - x - r, y - r, 1 - y - r))
        n = len(pos)
        min_dist = np.inf
        # Boundary distances
        for i in range(n):
            d = min(pos[i][0] - rad[i], 1 - pos[i][0] - rad[i], 
                     pos[i][1] - rad[i], 1 - pos[i][1] - rad[i])
            min_dist = min(min_dist, d)
        # Pairwise distances
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j]) - rad[i] - rad[j]
                min_dist = min(min_dist, d)
        return min_dist
    
    # This is slow for n=26. Let's vectorize.
    def min_distance_fast(pos, rad):
        n = len(pos)
        # Boundary
        boundary_dists = np.minimum(
            np.minimum(pos[:, 0] - rad, 1 - pos[:, 0] - rad),
            np.minimum(pos[:, 1] - rad, 1 - pos[:, 1] - rad)
        )
        min_boundary = np.min(boundary_dists)
        
        # Pairwise
        # Compute all pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Subtract radii
        r_sum = rad[:, np.newaxis] + rad[np.newaxis, :]
        pairwise_dists = dists - r_sum
        # Set diagonal to inf
        np.fill_diagonal(pairwise_dists, np.inf)
        min_pairwise = np.min(pairwise_dists)
        
        return min(min_boundary, min_pairwise)
    
    # Gradient of min distance is tricky because it's a min of many functions.
    # We can use a soft-min approximation: -log(sum(exp(-k * d_i))) / k
    # But for now, let's try a simpler approach: local search.
    
    # Actually, let's try a different strategy:
    # 1. Fix positions, solve for maximum radii using LP.
    # 2. Adjust positions to improve the LP solution.
    
    # The LP for maximum radii given positions:
    # Maximize sum(r_i)
    # Subject to:
    #   r_i + r_j <= ||p_i - p_j|| for all i < j
    #   r_i <= p_i_x, r_i <= 1 - p_i_x, r_i <= p_i_y, r_i <= 1 - p_i_y for all i
    #   r_i >= 0
    
    def solve_lp(pos):
        n = len(pos)
        # Variables: r_0, ..., r_{n-1}
        # Objective: maximize sum(r_i) -> minimize -sum(r_i)
        c = -np.ones(n)
        
        # Inequality constraints: A_ub @ r <= b_ub
        A_ub = []
        b_ub = []
        
        # Boundary constraints:
        # r_i <= p_i_x  ->  r_i <= p_i_x
        # r_i <= 1 - p_i_x -> r_i <= 1 - p_i_x
        # r_i <= p_i_y -> r_i <= p_i_y
        # r_i <= 1 - p_i_y -> r_i <= 1 - p_i_y
        for i in range(n):
            # r_i <= p_i_x
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(pos[i][0])
            
            # r_i <= 1 - p_i_x
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(1 - pos[i][0])
            
            # r_i <= p_i_y
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(pos[i][1])
            
            # r_i <= 1 - p_i_y
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(1 - pos[i][1])
        
        # Pairwise constraints:
        # r_i + r_j <= ||p_i - p_j||
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(d)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0, None)] * n
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return -res.fun  # sum of radii
        else:
            return 0.0
    
    # Now, let's do a local search on positions.
    # We'll use a simple hill-climbing approach:
    # 1. Solve LP for current positions.
    # 2. Try small random perturbations of positions.
    # 3. If the LP objective improves, accept the perturbation.
    
    current_pos = pos.copy()
    current_score = solve_lp(current_pos)
    best_score = current_score
    best_pos = current_pos.copy()
    
    step_size = 0.01
    n_iter = 1000
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Try a random perturbation
        new_pos = current_pos.copy()
        # Perturb one or two circles
        idx = np.random.randint(0, N)
        new_pos[idx][0] += np.random.uniform(-step_size, step_size)
        new_pos[idx][1] += np.random.uniform(-step_size, step_size)
        
        # Clamp to [0,1]
        new_pos = np.clip(new_pos, 0, 1)
        
        new_score = solve_lp(new_pos)
        
        if new_score > current_score:
            current_pos = new_pos
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pos = current_pos.copy()
        else:
            # Occasionally accept worse moves to escape local optima
            if np.random.rand() < 0.1:
                current_pos = new_pos
                current_score = new_score
        
        # Decrease step size occasionally
        if it % 100 == 0:
            step_size *= 0.9
    
    # Final LP solve to get optimal radii for best positions
    final_score = solve_lp(best_pos)
    
    # Get the radii from the LP solution
    n = len(best_pos)
    c = -np.ones(n)
    A_ub = []
    b_ub = []
    
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(best_pos[i][0])
        
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(1 - best_pos[i][0])
        
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(best_pos[i][1])
        
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(1 - best_pos[i][1])
    
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None)] * n
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        radii = res.x
    else:
        # Fallback to equal radii
        radii = np.full(N, 0.05)
    
    # Construct the result
    result = np.column_stack((best_pos, radii))
    
    # Ensure circles are inside the unit square
    # Clip radii to be non-negative and not exceed boundary distances
    for i in range(N):
        max_r = min(best_pos[i][0], 1 - best_pos[i][0], best_pos[i][1], 1 - best_pos[i][1])
        if result[i][2] > max_r:
            result[i][2] = max_r
    
    return result

# EVOLVE-BLOCK-END
